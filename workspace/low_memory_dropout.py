## Baseline ##
# Dropout was introduced as a way to improve the performance of deep NNs as a
# form of regularization. It takes a vector as input and produces a vector of
# the same shape as output. Each element in the vector has a probability p of
# being set to zero, otherwise it passes through unhindered. This forces the
# net to do well even when 1-p scalars from the input are available.
#
# At eval time we want to use the full power of the network so we set p=0. But
# this artificially increases the norm of the activations (ie. increase in
# output softmax temperature), so we multiply the output by 1/(1-p), which
# keeps the norm consistent regardless of the dropout probability.
#
# Here's the baseline implementation.
import tabulate
import torch

import triton
import triton.language as tl


@triton.jit
def _dropout(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data.
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above.
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write back output.
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


## Seeded Dropout ##
# The above implementation works fine, but is a bit awkward. Firstly we need to
# store the dropout mask for backprop, and secondly dropout state management
# can get tricky when using recompute/checkpointing. However here we will show
# an alternative implementation which has a smaller memory footprint, requires
# less data movement, and simplifies the management of persisting randomness
# across multiple invocations of the kernel.
#
# Pseudo-random number generation in Triton is simple. Here we'll use
# `triton.language.rand` function which generates a block of uniformly
# distribued float32 values in [0, 1), given a seed and a block of int32
# offsets.
@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    # Compute memory offsets of elements handled by this instance.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Load data from `x`.
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Randomly prune it.
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # Write-back.
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


def main():
    # Input tensor.
    x = torch.randn(size=(10,)).cuda()
    # Dropout mask
    p = 0.5
    x_keep = (torch.rand(size=(10,)) > p).to(torch.int32).cuda()
    output = dropout(x, x_keep=x_keep, p=p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep_mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))

    x = torch.randn(size=(10,)).cuda()
    # Compare this to the baseline - dropout mask is never instantiated!
    output = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)

    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist()
    ]))


if __name__ == "__main__":
    main()
