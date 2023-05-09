import os

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def _print(prefix, data):
    tl.static_print(prefix, data)


@triton.jit
def _print_tensor(prefix, var):
    tl.device_print(prefix, var)


DEBUG = 1
if DEBUG == 1 and os.environ["TRITON_DEBUG"] == 0:
    raise Exception("Please set TRITON_DEBUG=1 to debug.")
DEBUG_TENSOR_SHAPE_1 = (32, 256)
DEBUG_TENSOR_SHAPE_2 = (1024, 64)
DEBUG_TENSOR_SHAPE_3 = (1024, 1024)
DEBUG_TENSOR_DTYPE_1 = torch.float32
DEBUG_TENSOR_DTYPE_2 = torch.float32
DEBUG_TENSOR_DTYPE_3 = torch.float32
DEBUG_PRINT_FORS = 1


def self_attn_fwd(
    Q, K, V,
):
    qk = torch.einsum("nd,Nd->nN", Q, K)
    P = torch.nn.functional.softmax(qk, dim=-1)
    O = torch.einsum("nN,Nd->nd", P, V)
    return O


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,    # Inputs.
    O_ptr, l_ptr, m_ptr,    # Output and statistics.
    N: tl.constexpr, d: tl.constexpr,     # Dimension constants.
    BR_BLOCK_SIZE: tl.constexpr, BC_BLOCK_SIZE: tl.constexpr,   # Block sizes.
    TR_NUM_BLOCKS: tl.constexpr, TC_NUM_BLOCKS: tl.constexpr,   # Number of blocks.
):
    pid = tl.program_id(axis=0)

    # Load the Qi tile into SRAM.
    q_start = pid * BR_BLOCK_SIZE
    q_offs_n = tl.arange(0, BR_BLOCK_SIZE) + q_start
    q_offs_d = tl.arange(0, d)
    q_tile = q_offs_n[:, None] * d + q_offs_d[None, :]
    qi = tl.load(Q_ptr + q_tile)
    _print("qi.shape: ", qi.shape)

    # Initialize the output buffer.
    Oi = tl.zeros((BR_BLOCK_SIZE, d), dtype=tl.float32)

    # Load li and mi tiles, which share the same tile specifications. Physically
    # these are rows, but logically treat as columns.
    lm_start = pid * BR_BLOCK_SIZE
    lm_tile = tl.arange(0, BR_BLOCK_SIZE) + lm_start
    li = tl.load(l_ptr + lm_tile)
    mi = tl.load(m_ptr + lm_tile)

    # Loop across column blocks, per row (pid).
    for offset in range(TC_NUM_BLOCKS):
        # Load the Kj.T tile.
        k_start = offset * BC_BLOCK_SIZE
        k_offs_n = tl.arange(0, BC_BLOCK_SIZE) + k_start
        k_offs_d = tl.arange(0, d)
        k_tile_transposed = k_offs_n[None, :] * d + k_offs_d[:, None]
        kj_T = tl.load(K_ptr + k_tile_transposed)
        _print("kj.shape: ", kj_T.shape)

        # NOTE: Do not do `tl.trans` on local blocks! Local block transposes are
        #       not equivalent to the whole K.T matrix transpose! Ie. don't do
        #       this:
        #           k = tl.trans(K_j)
        #           S_ij_ = tl.dot(Q_i, k)
        #       Each pid does a coalesced block load, but SIMD makes the transpose
        #       operate per PID on its respective block, not as a whole.

        # qk.T dot product.
        Sij = tl.dot(qi, kj_T)

        # Calculate current tile statistics.
        mij_tilde = tl.max(Sij, axis=1)
        Pij_tilde = tl.exp(Sij - mij_tilde[:, None])
        lij_tilde = tl.sum(Pij_tilde, axis=1)

        # Update tile statistics.
        mi_new = tl.maximum(mi, mij_tilde)
        diff_mi = tl.exp(mi - mi_new)
        diff_mij_tilde = tl.exp(mij_tilde - mi_new)
        li_new = diff_mi * li + diff_mij_tilde * lij_tilde

        # Load the Vj tile.
        v_start = offset * BC_BLOCK_SIZE
        v_offs_n = tl.arange(0, BC_BLOCK_SIZE) + v_start
        v_offs_d = tl.arange(0, d)
        v_tile = v_offs_n[:, None] * d + v_offs_d[None, :]
        vj = tl.load(V_ptr + v_tile)
        _print("vj.shape: ", vj.shape)

        # Calculat Oi for iteration.
        prefactor = (1. / li_new)[:, None]
        first_term = (
            li[:, None] *
            diff_mi[:, None] *
            Oi
        )
        second_term = (
            diff_mij_tilde[:, None] *
            tl.dot(Pij_tilde, vj)
        )

        # Update accumulator and estimates.
        Oi = prefactor * (first_term + second_term)
        li = li_new
        mi = mi_new

    # Store back Oi, li, and mi.
    tl.store(O_ptr + q_tile, Oi)
    tl.store(l_ptr + lm_tile, li)
    tl.store(m_ptr + lm_tile, mi)


def flash_attn_fwd(
    Q, K, V,
    N, d, M,
):
    # Allocate output tensor.
    O = torch.zeros_like(Q)

    # Get strides.
    assert Q.shape == K.shape == V.shape, "QKV matrices don't have matching shapes!"
    assert len(Q.shape) == 2, "QKV matrices don't support batch dim yet."
    N, d = Q.shape

    # Calculate block sizes.
    BC_BLOCK_SIZE = triton.cdiv(M, 4 * d)
    BR_BLOCK_SIZE = min(triton.cdiv(M, 4 * d), d)

    # Heuristic
    BC_BLOCK_SIZE = 64
    BR_BLOCK_SIZE = 64
    TC_NUM_BLOCKS = triton.cdiv(N, BC_BLOCK_SIZE)
    TR_NUM_BLOCKS = triton.cdiv(N, BR_BLOCK_SIZE)

    # Allocate statistics tensors.
    l = torch.zeros((N,), dtype=torch.float32, device="cuda")
    m = torch.zeros((N,), dtype=torch.float32, device="cuda") - torch.inf

    # Debug.
    print("*" * 80)
    print(f"Running Flash Attention with parameters:\n"
          f"{Q.shape}, {K.shape}, {V.shape}\n"
          f"{O.shape}, {l.shape}, {m.shape}\n"
          f"N:{N}, d:{d},\n"
          f"Br:{BR_BLOCK_SIZE}, Bc:{BC_BLOCK_SIZE}\n"
          f"Tr:{TR_NUM_BLOCKS}, Tc:{TC_NUM_BLOCKS}\n")
    print("*" * 80)
    debug_tensor_1 = torch.zeros((DEBUG_TENSOR_SHAPE_1), dtype=DEBUG_TENSOR_DTYPE_1, device="cuda")
    debug_tensor_2 = torch.zeros((DEBUG_TENSOR_SHAPE_2), dtype=DEBUG_TENSOR_DTYPE_2, device="cuda")
    debug_tensor_3 = torch.zeros((DEBUG_TENSOR_SHAPE_3), dtype=DEBUG_TENSOR_DTYPE_3, device="cuda")

    # Enqueue kernel.
    num_warps = 2
    flash_attn_fwd_kernel[(TC_NUM_BLOCKS,)](
        Q, K, V,
        O, l, m,
        N, d,
        BR_BLOCK_SIZE=BR_BLOCK_SIZE, BC_BLOCK_SIZE=BC_BLOCK_SIZE,
        TR_NUM_BLOCKS=TR_NUM_BLOCKS, TC_NUM_BLOCKS=TC_NUM_BLOCKS,
        num_warps=num_warps,
    )
    m_true = torch.max(Q@K.T, dim=-1).values
    l_true = torch.sum(torch.exp(Q@K.T - m_true[:, None]), dim=-1)
    assert torch.allclose(m, m_true, atol=1e-1)
    assert torch.allclose(l, l_true, atol=1e-1)
    return O, debug_tensor_1, debug_tensor_2, debug_tensor_3


def test_flash_attn_fwd(N, d):
    torch.manual_seed(0)
    Q = torch.randn((N, d), dtype=torch.float32, device="cuda")
    K = torch.randn((N, d), dtype=torch.float32, device="cuda")
    V = torch.randn((N, d), dtype=torch.float32, device="cuda")
    ret, debug_tensor_1, debug_tensor_2, debug_tensor_3 = flash_attn_fwd(Q, K, V, N, d, M=167936)
    ref = self_attn_fwd(Q, K, V)
    print(ret)
    print(ref)
    assert torch.allclose(ret, ref, atol=2e-2, rtol=0)
    return ret, debug_tensor_1, debug_tensor_2, debug_tensor_3


def main():
    ret, debug_tensor_1, debug_tensor_2, debug_tensor_3 = test_flash_attn_fwd(N=1024, d=64)


if __name__ == "__main__":
    main()
