"""GPU smoke test: numerical agreement of the GPU scan backends vs reference.

Checks, at small shapes (a few hundred MB, safe to share a busy GPU):

  1. chunk   vs reference — forward + all input grads
  2. chunk deep-logit guard — a batch with f_logit < -5 must route through the
     cuda fallback and still match reference
  3. cuda    vs reference — forward + all input grads (JIT-builds the extension
     on first use)
  4. cuda_bw vs reference — training path (fwd+bwd) and inference path
     (no_grad -> chunk-parallel forward)

Usage: python naju/tests/test_gpu_smoke.py
"""
import os
import sys

os.environ.pop("NAJU_SCAN_BACKEND", None)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import torch

from naju.scan import naju_scan_reference

assert torch.cuda.is_available(), "this smoke test needs a CUDA GPU"
DEV = "cuda"
SHAPES = [(2, 256, 64, 16), (2, 512, 96, 64)]   # (B, T, D, N)


def make_inputs(B, T, D, N, deep_logit=False, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    def rnd(*shape, shift=0.0):
        return (torch.randn(*shape, device=DEV, generator=g) + shift).requires_grad_(True)
    u = rnd(B, T, D)
    fl = rnd(B, T, D, shift=3.0)
    if deep_logit:
        with torch.no_grad():
            fl[:, T // 2] = -8.0          # below the chunk guard (-5)
    il = rnd(B, T, D, shift=-2.0)
    Bm = rnd(B, T, N)
    Cm = rnd(B, T, N)
    Dv = (0.01 * torch.randn(D, device=DEV, generator=g)).requires_grad_(True)
    return u, fl, il, Bm, Cm, Dv


def compare(name, fn, shape, deep_logit=False, tol_y=2e-4, tol_g=2e-3):
    B, T, D, N = shape
    ref_in = make_inputs(B, T, D, N, deep_logit)
    tst_in = [t.detach().clone().requires_grad_(True) for t in ref_in]

    y_ref = naju_scan_reference(*ref_in)
    y_tst = fn(*tst_in)
    dy = maxdiff = (y_ref - y_tst).abs().max().item()
    assert maxdiff < tol_y, f"{name} {shape}: fwd diff {maxdiff}"

    seed_grad = torch.randn_like(y_ref)
    y_ref.backward(seed_grad)
    y_tst.backward(seed_grad)
    gmax = 0.0
    for i, (a, b) in enumerate(zip(ref_in, tst_in)):
        gd = (a.grad - b.grad).abs().max().item()
        gmax = max(gmax, gd)
        assert gd < tol_g, f"{name} {shape}: grad[{i}] diff {gd}"
    print(f"  {name:8s} {str(shape):20s} fwd {dy:.2e}  grad(max) {gmax:.2e}")
    return dy, gmax


def main():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}, free {free/2**30:.1f} GiB")

    print("[1] chunk vs reference")
    from naju.chunk_triton import naju_chunk_scan
    for s in SHAPES:
        compare("chunk", naju_chunk_scan, s)

    print("[2] cuda vs reference (JIT build on first call)")
    from naju.cuda import naju_cuda_scan
    for s in SHAPES:
        compare("cuda", naju_cuda_scan, s, tol_y=1e-4, tol_g=1e-3)

    print("[3] chunk deep-logit guard -> cuda fallback")
    for s in SHAPES:
        compare("chunk", naju_chunk_scan, s, deep_logit=True)

    print("[4] cuda_bw vs reference (train path + inference path)")
    from naju.cuda.bw import naju_cuda_bw_scan
    for s in SHAPES:
        compare("cuda_bw", naju_cuda_bw_scan, s, tol_y=1e-4, tol_g=1e-3)
    for s in SHAPES:
        B, T, D, N = s
        ins = make_inputs(B, T, D, N)
        with torch.no_grad():
            y_ref = naju_scan_reference(*ins)
            y_inf = naju_cuda_bw_scan(*ins)   # no_grad -> fwd_chunked path
        d = (y_ref - y_inf).abs().max().item()
        assert d < 1e-4, f"cuda_bw inference {s}: diff {d}"
        print(f"  cuda_bw  {str(s):20s} inference(no_grad) {d:.2e}")

    print(f"peak GPU mem this process: "
          f"{torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    print("GPU SMOKE ALL PASS")


if __name__ == "__main__":
    main()
