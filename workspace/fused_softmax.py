import torch

import triton
import triton.language as tl


@torch.jit.script
def naive_softmax(x):
    """Naive softmax.

    Does 5MN + 2M loads and 3MN + 2N stores for some
    M x N tensor. The maximum element of the tensor is subtracted to avoid
    overflows. Softmax is invariant to this shift.
    """
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


# We want to fused these softmax operations so we don't do so much memory
# movement between kernel calls. Theoretically, we'd like to read only MN
# elements of x once and write back MN elements as the output for a ~4x
# theoretical speedup. `torch.jit.script` tries to do the kernel fusion
# automatically, but it's far from ideal.


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """Faster fused softmax kernel.
    
    Loads a row of the input matrix X, normalizes it, then writes it back to
    the output Y. Note that blocks in triton must be powers of 2, so we need to
    internally pad each row and guard the memory operations properly if we want
    to handle arbitrary input shapes.
    """
    # Rows of softmax are independent, parallelize across that.
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance
    # 1 row.
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of 2 greater than n_cols, so we can fit
    # each row in a single block.
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > n_cols.
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    # Subtract max value for numerical stability.
    row_minus_max = row - tl.max(row, axis=0)
    # Note that the exponentiation in Triton is fast but approximate, think
    # __expf in CUDA.
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM.
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x):
    """Fused softmax kernel helper function.
    """
    n_rows, n_cols = x.shape
    # The block size is the smallest power of 2 greater than the number of
    # columns in `x`.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per
    # row by increasing the number of warps (`num_warps`) over which each row
    # is distributed. You can auto-tune this feature so you don't have to come
    # up with manual heuristics.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output tensor.
    y = torch.empty_like(x)
    # Enqueue kernel. 1D launch grid, where we have one kernel instance per row
    # of the input matrix.
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # Arg names to use as an x-axis for plot.
        x_vals=[128 * i for i in range(2, 100)],    # Possible values for `x_name`.
        line_arg="provider",
        line_vals=[
            "triton",
            "torch-native",
            "torch-jit",
        ],
        line_names=[
            "Triton",
            "Torch (native)",
            "Torch (jit)",
        ],
        styles=[("blue", "-"), ("green", "-"), ("green", "--")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M":4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    percentiles = [0.5, 0.2, 0.8]
    if provider == "torch-native":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.softmax(x, axis=1), percentiles=percentiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: softmax(x), percentiles=percentiles)
    if provider == "torch-jit":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: naive_softmax(x), percentiles=percentiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def main():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    print(y_torch)
    print(y_triton)
    print(
        f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(y_torch - y_triton))}'
    )
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)

    benchmark.run(
        print_data=True,
        show_plots=False,
        save_path="/scratch/ssd004/scratch/mchoi/triton_compiler/workspace/benchmarks",
    )


if __name__ == "__main__":
    main()
