"""Chunked diagonal scan used by the pure-PyTorch Mamba baseline.

The naive sequential scan (a Python loop over all T time steps) is correct but
launch-bound. The two-level chunked scan reduces the number of Python/autograd
steps from T to roughly (chunk + T/chunk):

  1. intra-chunk pass : run the recurrence inside every chunk in parallel
                        (loop length = chunk), assuming zero chunk-start state.
  2. inter-chunk pass : propagate the carry state across chunks
                        (loop length = T/chunk).
  3. combine          : add each chunk's carry contribution to every position.

Recurrence form: diagonal  x_t = a_t * x_{t-1} + b_t.
"""
import torch
import torch.nn.functional as F


def sequential_diagonal(a, b):
    """Reference. a,b: [B,T,di,ds] -> x: [B,T,di,ds]."""
    B, T, di, ds = a.shape
    x = a.new_zeros(B, di, ds)
    out = []
    for t in range(T):
        x = a[:, t] * x + b[:, t]
        out.append(x)
    return torch.stack(out, dim=1)


def chunked_diagonal(a, b, chunk=64):
    """Fast diagonal scan. a,b: [B,T,di,ds] -> x: [B,T,di,ds]."""
    B, T, di, ds = a.shape
    pad = (chunk - T % chunk) % chunk
    if pad:
        a = F.pad(a, (0, 0, 0, 0, 0, pad), value=1.0)  # a=1 keeps state
        b = F.pad(b, (0, 0, 0, 0, 0, pad), value=0.0)
    Tp = T + pad
    nc = Tp // chunk
    a = a.reshape(B, nc, chunk, di, ds)
    b = b.reshape(B, nc, chunk, di, ds)

    # intra-chunk: state assuming zero chunk-start, and cumulative product of a
    local = a.new_zeros(B, nc, di, ds)
    Pa = a.new_ones(B, nc, di, ds)
    outs, Pas = [], []
    for i in range(chunk):
        local = a[:, :, i] * local + b[:, :, i]
        Pa = Pa * a[:, :, i]
        outs.append(local)
        Pas.append(Pa)
    local_states = torch.stack(outs, dim=2)   # [B,nc,chunk,di,ds]
    Pa_cum = torch.stack(Pas, dim=2)          # [B,nc,chunk,di,ds]

    # inter-chunk carry recurrence
    chunk_last = local_states[:, :, -1]       # [B,nc,di,ds]
    chunk_Pa = Pa_cum[:, :, -1]
    carry = a.new_zeros(B, di, ds)
    carries = []
    for c in range(nc):
        carries.append(carry)                 # carry entering chunk c
        carry = chunk_Pa[:, c] * carry + chunk_last[:, c]
    carry_in = torch.stack(carries, dim=1)    # [B,nc,di,ds]

    x = local_states + Pa_cum * carry_in[:, :, None]
    x = x.reshape(B, Tp, di, ds)[:, :T]
    return x
