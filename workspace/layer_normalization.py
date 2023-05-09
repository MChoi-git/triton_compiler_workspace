## Motivations ##
# Improves the perf of sequential models (ie. transformers) or NNs with small
# batch size. Takes an input vector x and produces an output vector y of the
# same shape. The normalization is done by subtracting the mean and dividing
# by the standard deviation of x. After the normalization, a learnable affine
# transformation with weights w and biases b is applied. The forward pass can
# be expressed as follows:
#
#   y = [(x - E[x]) / sqrt(Var(x) + eps)] * w + b
#
# where eps is a small constant added for numerical stability. Here's the
# implementation.
import torch

import triton
import triton.language as tl

try:
    import apex
    HAS_APEX = True

except ModuleNotFoundError:
    HAS_APEX = False


@triton.jit
def _layer_norm_fwd_fused(
    X,  # Pointer to input.
    Y,  # Pointer to output.
    W,  # Pointer to weights.
    B,  # Pointer to bias.
    Mean,   # Pointer to the mean
    Rstd,   # Pointer to the 1/std
    stride, # How much to increase the pointer when moving 1 row.
    N,  # Number of columns in X.
    eps,    # Epsilon to avoid division by zero.
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(axis=0)
    Y += row * stride
    X += row * stride
    # Compute the mean.
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute the variance.
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean and std.
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation.
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output.
        tl.store(Y + cols, y, mask=mask)


## Backward Pass ##
# The backward pass for the layer norm operator is quite a bit more involved
# than the forward pass, so refer to 
# `https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html`
# for the derivations. The main targets however are the VJPs for delx, delw and
# delb.
#
# Recall that the layer norm operator described here is a mapping LN: x -> y,
# where x and y are both vectors. For the grads wrt. x, we calculate them row-
# wise and stack them to form the full grad. But for the grads wrt. w and b, we
# must reduce them since the same w and b vectors are applied to every row. To
# do this, we can use a parallel reduction strategy: Each kernel instance
# accumulates partial delw and delb across certain rows into one of
# `GROUP_SIZE_M` independent buffers. These buffers stay in L2 cache, and then
# are further reduced by another function to compute the actual delw and delb.
@triton.jit
def _layer_norm_bwd_dx_fused(
    DX, # Pointer to the input gradient.
    DY, # Pointer to the output gradient.
    DW, # Pointer to the partial sum of weights gradient.
    DB, # Pointer to the partial sum of biases gradient.
    X,  # Pointer to the input.
    W,  # Pointer to the weights.
    B,  # Pointer to the biases.
    Mean,   # Pointer to the mean.
    Rstd,   # Pointer to the 1/std.
    Lock,   # Pointer to the lock.
    stride, # How much to increase the pointer when moving by 1 row.
    N,  # Number of columns in X.
    eps,    # Prevent divide by zero.
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases grad pointer for parallel reduction.
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM.
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx.
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx.
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw, db.
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    # Wait until lock is free (0) and grab it (0 -> 1).
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate.
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock.
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW, # Pointer to the partial sum of weights gradient.
    DB, # Pointer to the partial sum of biases gradient.
    FINAL_DW,   # Pointer to the weights gradient.
    FINAL_DB,   # Pointer to the biases gradient.
    M,  # GROUP_SIZE_M.
    N,  # Number of columns.
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(axis=0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


## Benchmark ##
# Let's compare the kernel to Pytorch. Here the inputs are less than 64 KB
# (65536 B / 32768 params @ bf16). Specifically, you can set
# `"mode": "backward"` to benchmark the backward pass.
class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        # Allocate output.
        y = torch.empty_like(x)
        # Reshape input data into 2D tensor.
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64 KB per feature, enqueue fused kernel.
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim "
                               ">= 64KB")
        # Heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # Enqueue kernel.
        _layer_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,
            x_arg.stride(0),
            N,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, v = ctx.saved_tensors
        # Heuristics for amount of parallel reduction stream for DW/DB.
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256
        # Allocate output.
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        _dw = torch.empty(
            (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device,
        )
        _db = torch.empty(
            (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device,
        )
        dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # Enqueue kernel using forward pass heuristics. Also compute partial
        # sum for DW and DB.
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M,)](
            dx,
            dy,
            _dw,
            _db,
            x,
            w,
            b,
            m,
            v,
            locks,
            x_arg.stride(0),
            N,
            ctx.eps,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=ctx.num_warps,
        )
        grid = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
        # Accumulate partial sums in separate kernel.
        _layer_norm_bwd_dwdb[grid](
            _dw,
            _db,
            dw,
            db,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=128,
        )
        return dx, None, dw, db, None


layer_norm = LayerNorm.apply


def test_layer_norm(M, N, dtype, eps=1e-5, device="cuda"):
    # Create data.
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(
        w_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.rand(
        w_shape, dtype=dtype, device="cuda", requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # Forward pass.
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # Backward pass (Triton).
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # Backward pass (Torch).
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # Compare
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['triton', 'torch'] + (['apex'] if HAS_APEX else []),
        line_names=['Triton', 'Torch'] + (['Apex'] if HAS_APEX else []),
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'}
    )
)
def bench_layer_norm(M, N, dtype, provider, mode="backward", eps=1e-5, device="cuda"):
    # Create data.
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(
        w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == 'triton':
        y_fwd = lambda: layer_norm(x, w_shape, weight, bias, eps)
    if provider == 'torch':
        y_fwd = lambda: torch.nn.functional.layer_norm(
            x, w_shape, weight, bias, eps)
    if provider == 'apex':
        apex_layer_norm = apex.normalization.FusedLayerNorm(w_shape).to(
            x.device).to(x.dtype)
        y_fwd = lambda: apex_layer_norm(x)
    # forward pass
    if mode == 'forward':
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(
            y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    if mode == 'backward':
        gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=quantiles, grad_to_none=[x], rep=500
        )
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def main():
    test_layer_norm(1151, 8192, torch.float16)
    bench_layer_norm.run(
        print_data=True,
        save_path="/scratch/ssd004/scratch/mchoi/triton_compiler/workspace/benchmarks",
    )


if __name__ == "__main__":
    main()
