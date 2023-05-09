import torch

import triton
import triton.language as tl


# Blocked MatMul of (M, K) by a (K, N) matrix:
#for m in range(0, M, BLOCK_SIZE_M):    # Do in parallel
#    for n in range(0, N, BLOCK_SIZE_N):    # Do in parallel
#        axx = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):    # Do in parallel
#            a = A[m: m + BLOCK_SIZE_M, k: k + BLOCK_SIZE_K]
#            b = B[k: k + BLOCK_SIZE_K, n: n + BLOCK_SIZE_N]
#            acc += dot(a, b)
#        C[m: m + BLOCK_SIZE_M, n: n + BLOCK_SIZE_N] = acc
#
# Each iteration of the doubly-nested for-loop is performed by a dedicated
# Triton program instance.
#
#

### Compute kernel ###
# Above kernel is simple to implement in Triton, but the main difficulty comes
# from the computation of the memory locations at which blocks of A and B must
# be read in the inner loop. For that, we need multi-dimensional pointer
# arithmetics.
#
# Pointer arithmetics:
# For a row-major 2D tensor `X`, the memory location of `X[i, j]` is given by
# `&X[i, j] = X + i * stride_xi + j * stride_xj`. Therefore, blocks of pointers
# for `A[m: m + BLOCK_SIZE_M, k: k + BLOCK_SIZE_K] and 
# B[k: k + BLOCK_SIZE_K, n: n + BLOCK_SIZE_N]` can be defined in pseudo-code
# as:
#
#   A[m: m + BLOCK_SIZE_M, k: k + BLOCK_SIZE_K] =
#       a_ptr + (m: m + BLOCK_SIZE_M)[:, None] * A.stride(0) + (k: k + BLOCK_SIZE_K)[None, :] * A.stride(1);
#   B[k: k + BLOCK_SIZE_K, n: n + BLOCK_SIZE_N] =
#       b_ptr + (k: k + BLOCK_SIZE_K)[:, None] * B.stride(0) + (n: n + BLOCK_SIZE_N)[None, :] * B.stride(1);
#
# And now we can define the pointers for blocks of A and B in Triton as the
# following:
#   offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#   offs_bn = (pid_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#   offs_k = tl.arange(0, BLOCK_SIZE_K)
#   a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#
# Note that the modulo handles the case where `M` is not a multiple of
# `BLOCK_SIZE_M` or `N` is not a multiple of `BLOCK_SIZE_N`, in which case we
# pad the data. For the `K` dimension, we handle that later using masking.
#
# Now we have the inner loop updates for the pointers:
#
#   a_ptrs += BLOCK_SIZE_K * stride_ak;
#   b_ptrs += BLOCK_SIZE_K * stride_bk;
#
# L2 Cache Optimizations:
# Each program computes a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of `C`. However,
# remember that the order of computation of these blocks matter since it
# affects L2 cache hit rate of our program. And unfortunately, a simple row-
# major ordering like this is not going to cut it:
#
#   pid = triton.program_id(0);
#   grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M;
#   grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N;
#   pid_m = pid / grid_n;
#   pid_n = pid % grid_n;
#
# For example: 9 x 9 blocks MatMul
#   - Row-major ordering:
#       - A loads row of 9 blocks -> (1, 9)
#       - B loads all 81 blocks -> (9, 9)
#       - C accumulates row of 9 blocks -> (1, 9)
#   - Grouped ordering:
#       - A loads rows of 27 blocks -> (3, 9)
#       - B loads columns of 27 blocks -> (9, 3)
#       - C accumulates 9 blocks -> (3, 3)
#   - Instead of loading 90 blocks to compute 9 blocks, we load 54 blocks.
#
# In grouped ordering, we launch blocks in an order that promotes data
# reuse. We "super-group" blocks into groups of `GROUP_M` rows before
# switching to the next columns:
#
## Program ID.
#pid = tl.program_id(axis=0)
## Number of program ids along the M axis.
#num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
## Number of program ids along the N axis.
#num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
## Number of the programs in the group.
#num_pid_in_group = GROUP_SIZE_M * num_pid_n
## ID of the group this program is in.
#group_id = pid // num_pid_in_group
## Row-ID of the first program in the group.
#first_pid_m = group_id * GROUP_SIZE_M
## If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller.
#group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
## *Within groups*, programs are ordered in a column-major order
## Row-id of the program in the *launch grid*
#pid_m = first_pid_m + (pid % group_size_m)
## Col-id of the program in the *launch grid*
#pid_n = (pid % num_pid_in_group) // group_size_m
#
# In practice, this boosts our perf by more than 10%.


## Final Result ##
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices.
    a_ptr, b_ptr, c_ptr,
    # Matrix dims.
    M, N, K,
    # The stride variable represents how much to increase the ptr when moving
    # by 1 element in a particular dimension. Eg. `stride_am` is how much to
    # increase `a_ptr` by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters.
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for Matmul.

    Kernel for computing the matmul C = A x B. A has shape (M, K), B has shape
    (K, N) and C has shape (M, N).
    """
    # Map program ids `pid` to the block of C it should compute. This is done
    # in a grouped ordering to promote L2 data reuse.
    # Note: This is at block granularity
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n   # GROUP_SIZE_M is `num_pid_m`
    group_id = pid // num_pid_in_group  # pids assigned going down C columns
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B. We will advance this
    # pointer as we move in the K direction and accumulate.
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers.
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers.
    # Note: This is assigning pointers within blocks.
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate to compute a block of the C matrix. We will accumulate into a
    # `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K
        # dimension. If it's out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Note that the 2.0.0 Triton breaks with this mask, need the nightly
        # version since the tutorials on the docs follows the newest branch.
        # For simplicitly masking isn't applied here, so if `K` is not a
        # multiple of `BLOCK_SIZE_K`, this will access out of bounds error or
        # (worse) incorrect results.
        #a = tl.load(a_ptrs)
        #b = tl.load(b_ptrs)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here while the accumulator is
    # still in FP32.
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# Can fused leaky relu by providing it as an `ACTIVATION` meta-parameter in
# `matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    """Helper function for matmul kernel.
    """
    # Check constraints.
    #assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    b = b.T
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty ((M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider",
        line_vals=["cublas", "triton"],
        line_names=["cuBLAS", "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def main():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    benchmark.run(
        show_plots=False,
        print_data=True,
        save_path="/scratch/ssd004/scratch/mchoi/triton_compiler/workspace/benchmarks",
    )


if __name__ == "__main__":
    main()
