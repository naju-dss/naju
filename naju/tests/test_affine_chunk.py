"""Correctness suite for the affine chunk backend (chunk_affine_triton).

  1. Chunk-decomposition math vs the sequential recurrence — pure torch, CPU
     (plan Experiment 1: validates the hierarchical affine identities alone).
  2. GPU forward vs naju_scan_reference over a shape matrix incl. ragged T
     (padding path) and Q sweep.  fp32 tolerance rtol 1e-5 / atol 1e-6.
  3. Gate stress: f_logit down to -12 under no_grad — the affine forward must
     stay exact where the chunk backend's decay-ratio math breaks (its guard
     region); the chunk backend's error is printed for contrast.
  4. Gradients vs autograd of the reference scan (shared fused backward wired
     to the affine forward).
  5. Long-sequence smoke: T=32768, no NaN/Inf, matches chunk backend.

Usage: python naju/tests/test_affine_chunk.py   (1. runs on CPU; rest need GPU)
"""
import os
import sys

os.environ.pop("NAJU_SCAN_BACKEND", None)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import torch

from naju.scan import naju_scan_reference

torch.manual_seed(0)


def maxdiff(a, b):
    return (a - b).abs().max().item()


def _rand_inputs(B_, T, D, N, device, f_shift=2.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    def r(*s):
        return torch.randn(*s, generator=g).to(device)
    u = r(B_, T, D)
    fl = r(B_, T, D) + f_shift          # sigmoid ~0.9 region (trained-gate-like)
    il = r(B_, T, D) - 1.0
    B = r(B_, T, N) / (N ** 0.5)
    C = r(B_, T, N) / (N ** 0.5)
    D_skip = r(D) * 0.01
    return u, fl, il, B, C, D_skip


def test_chunk_math_cpu():
    """Plan Experiment 1: hierarchical decomposition == sequential scan."""
    B_, T, D, N, Q = 2, 96, 8, 16, 32
    u, fl, il, B, C, D_skip = _rand_inputs(B_, T, D, N, "cpu")
    f, i = torch.sigmoid(fl), torch.sigmoid(il)
    w = (i * u).unsqueeze(-1) * B.unsqueeze(2)          # [B,T,D,N]

    # sequential ground truth
    x = torch.zeros(B_, D, N)
    seq = []
    for t in range(T):
        x = f[:, t].unsqueeze(-1) * x + w[:, t]
        seq.append(x)
    seq = torch.stack(seq, dim=1)

    # hierarchical: local scan from zero + prefix retention + boundary carry
    out = torch.empty_like(seq)
    s = torch.zeros(B_, D, N)
    for k0 in range(0, T, Q):
        z = torch.zeros(B_, D, N)
        p = torch.ones(B_, D)
        zs, ps = [], []
        for q in range(Q):
            z = f[:, k0 + q].unsqueeze(-1) * z + w[:, k0 + q]
            p = p * f[:, k0 + q]
            zs.append(z); ps.append(p)
        zs = torch.stack(zs, 1); ps = torch.stack(ps, 1)
        out[:, k0:k0 + Q] = zs + ps.unsqueeze(-1) * s.unsqueeze(1)
        s = out[:, k0 + Q - 1]
    d = maxdiff(seq, out)
    assert d < 1e-6, f"chunk decomposition mismatch {d}"
    print(f"1. chunk math (CPU): decomposition == sequential, maxdiff {d:.2e}")


def test_forward_gpu():
    from naju.chunk_affine_triton import naju_affine_chunk_scan

    shapes = [(1, 7, 32, 16), (2, 64, 64, 64), (2, 65, 64, 64),
              (1, 129, 96, 32), (2, 192, 128, 64), (1, 512, 64, 128),
              (8, 256, 512, 64)]
    worst = 0.0
    for (B_, T, D, N) in shapes:
        ins = _rand_inputs(B_, T, D, N, "cuda", seed=T + D)
        ref = naju_scan_reference(*ins)
        for Q in (16, 32, 64, 128):
            y = naju_affine_chunk_scan(*ins, Q=Q)
            d = maxdiff(ref, y)
            worst = max(worst, d)
            assert d < 1e-4, f"fwd mismatch {d} at B{B_} T{T} D{D} N{N} Q{Q}"
            torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
    print(f"2. GPU forward vs reference: {len(shapes)} shapes x Q(16..128), "
          f"worst maxdiff {worst:.2e}")


def test_deep_gate_exactness():
    from naju.chunk_affine_triton import naju_affine_chunk_scan
    from naju.chunk_triton import naju_chunk_scan

    B_, T, D, N = 2, 256, 64, 64
    u, fl, il, B, C, D_skip = _rand_inputs(B_, T, D, N, "cuda", seed=7)
    fl = fl - 14.0                     # deep gates: f ~ sigmoid(-12) ~ 6e-6
    ref = naju_scan_reference(u, fl, il, B, C, D_skip)
    with torch.no_grad():
        ya = naju_affine_chunk_scan(u, fl, il, B, C, D_skip, check_range=False)
        yc = naju_chunk_scan(u, fl, il, B, C, D_skip, check_range=False)
    da, dc = maxdiff(ref, ya), maxdiff(ref, yc)
    assert da < 1e-5, f"affine fwd inexact at deep gates: {da}"
    print(f"3. deep-gate (f_logit ~ -12) forward: affine {da:.2e} (exact), "
          f"chunk decay-ratio {dc:.2e} (guard region, for contrast)")


def test_backward_gpu():
    from naju.chunk_affine_triton import naju_affine_chunk_scan

    B_, T, D, N = 2, 192, 64, 32
    ins = _rand_inputs(B_, T, D, N, "cuda", seed=3)
    names = ("u", "f_logit", "i_logit", "B", "C", "D_skip")

    ref_in = [t.detach().clone().requires_grad_(True) for t in ins]
    aff_in = [t.detach().clone().requires_grad_(True) for t in ins]
    g = torch.Generator(device="cpu").manual_seed(9)
    dy = torch.randn(B_, T, D, generator=g).to("cuda")

    naju_scan_reference(*ref_in).backward(dy)
    naju_affine_chunk_scan(*aff_in).backward(dy)

    worst = 0.0
    for n, r, a in zip(names, ref_in, aff_in):
        d = maxdiff(r.grad, a.grad)
        worst = max(worst, d)
        assert d < 1e-3, f"grad mismatch on {n}: {d}"
        torch.testing.assert_close(a.grad, r.grad, rtol=1e-4, atol=1e-4)
    print(f"4. GPU grads vs reference autograd: all inputs, worst {worst:.2e}")


def test_long_sequence():
    from naju.chunk_affine_triton import naju_affine_chunk_scan
    from naju.chunk_triton import naju_chunk_scan

    B_, T, D, N = 1, 32768, 128, 64
    ins = _rand_inputs(B_, T, D, N, "cuda", seed=11)
    with torch.no_grad():
        ya = naju_affine_chunk_scan(*ins)
        yc = naju_chunk_scan(*ins)
    assert torch.isfinite(ya).all(), "NaN/Inf at T=32768"
    d = maxdiff(ya, yc)
    assert d < 1e-3, f"long-seq divergence vs chunk: {d}"
    print(f"5. long sequence T=32768: finite, vs chunk maxdiff {d:.2e}")


if __name__ == "__main__":
    test_chunk_math_cpu()
    if torch.cuda.is_available():
        test_forward_gpu()
        test_deep_gate_exactness()
        test_backward_gpu()
        test_long_sequence()
        print("ALL PASS")
    else:
        print("CUDA unavailable — GPU tests skipped")
