import os

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def matmul_broken(
    A_ptr, B_ptr, C_ptr,
    m: tl.constexpr, k: tl.constexpr, n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Define tile of pointers for A.
    a_start = pid * BLOCK_M
    a_offs_m = tl.arange(0, BLOCK_M) + a_start
    a_offs_n = tl.arange(0, k)
    a_tile = a_offs_m[:, None] * k + a_offs_n[None, :]

    # Load A.
    A = tl.load(A_ptr + a_tile)

    for offset in tl.static_range(n // BLOCK_N):
        # Define tile of pointers for B.
        b_start = offset * BLOCK_N
        b_offs_n = tl.arange(0, BLOCK_N) + b_start
        b_offs_k = tl.arange(0, k)
        b_tile = b_offs_n[:, None] * k + b_offs_k[None, :]
        B = tl.load(B_ptr + b_tile)

        # Define tile of pointers for C.
        c_start_m = pid * BLOCK_M
        c_start_n = offset * BLOCK_N
        c_offs_m = tl.arange(0, BLOCK_M) + c_start_m
        c_offs_n = tl.arange(0, BLOCK_N) + c_start_n
        c_tile = c_offs_m[:, None] * n + c_offs_n[None, :] * 1

        c_block = tl.dot(A, tl.trans(B))
        tl.store(C_ptr + c_tile, c_block)


@triton.jit
def matmul_fixed(
    A_ptr, B_ptr, C_ptr,
    m: tl.constexpr, k: tl.constexpr, n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Define tile of pointers for A.
    a_start = pid * BLOCK_M
    a_offs_m = tl.arange(0, BLOCK_M) + a_start
    a_offs_n = tl.arange(0, k)
    a_tile = a_offs_m[:, None] * k + a_offs_n[None, :]

    # Load A.
    A = tl.load(A_ptr + a_tile)

    for offset in tl.static_range(n // BLOCK_N):
        # Define tile of pointers for B.
        b_start = offset * BLOCK_N
        b_offs_n = tl.arange(0, BLOCK_N) + b_start
        b_offs_k = tl.arange(0, k)
        b_tile = b_offs_n[None, :] * k + b_offs_k[:, None]
        B = tl.load(B_ptr + b_tile)

        # Define tile of pointers for C.
        c_start_m = pid * BLOCK_M
        c_start_n = offset * BLOCK_N
        c_offs_m = tl.arange(0, BLOCK_M) + c_start_m
        c_offs_n = tl.arange(0, BLOCK_N) + c_start_n
        c_tile = c_offs_m[:, None] * n + c_offs_n[None, :] * 1

        c_block = tl.dot(A, B)
        tl.store(C_ptr + c_tile, c_block)


def transpose_test(m, n, k, BLOCK_M, BLOCK_N):
    """Do A @ B.T, with shapes (m, k) and (n, k), m == n.
    """
    torch.manual_seed(0)
    A = torch.randn((m, k), dtype=torch.float32, device="cuda")
    B = torch.randn((n, k), dtype=torch.float32, device="cuda")
    C_triton_broken = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    C_triton_fixed = torch.zeros((m, n), dtype=torch.float32, device="cuda")

    # m and n should technically be equal.
    assert m % BLOCK_M == 0
    assert n % BLOCK_N == 0
    assert m // BLOCK_M == n // BLOCK_N

    # Triton transpose kernel.
    grid = (m // BLOCK_M,)
    matmul_broken[grid](
        A, B, C_triton_broken,
        m, k, n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    matmul_fixed[grid](
        A, B, C_triton_fixed,
        m, k, n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # Torch transpose kernel.
    C_torch = A @ B.T

    # Check that torch and fixed matmul are equal
    torch.allclose(C_triton_fixed, C_torch, atol=3e-2)

    print(C_triton_broken)
    print(C_triton_fixed)
    print(C_torch)

    diff = (torch.abs(C_triton_broken - C_triton_fixed) < 0.06)

    # NOTE: Diff is only correct in columnar stripes, every 4. This is because
    #       `tl.trans(...)` only transposes blocks in a `tl.dot`. It
    #       also seems that `tl.trans(...)` only works within a `tl.dot` too,
    #       I guess since they form one LLVM/MLIR intrinsic (see paper).

    breakpoint()


def main():
    transpose_test(128, 128, 64, 64, 64)


if __name__ == "__main__":
    main()
