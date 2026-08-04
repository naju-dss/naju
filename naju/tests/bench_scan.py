"""Latency benchmark: affine chunk backend vs existing scan backends.

Measures forward (no_grad) and forward+backward wall time per call with CUDA
events. Paper-linked shape (d_inner=512, N=64) at several T; batch shrinks at
long T to fit alongside whatever else occupies the GPU.

Usage: python naju/tests/bench_scan.py [--iters 30] [--warmup 10]
"""
import argparse
import os
import sys

os.environ.pop("NAJU_SCAN_BACKEND", None)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

import torch


DTYPE = torch.float32


def _rand_inputs(B_, T, D, N, device, requires_grad=False):
    g = torch.Generator(device="cpu").manual_seed(0)
    def r(*s):
        return torch.randn(*s, generator=g).to(device)
    u = r(B_, T, D)
    fl = r(B_, T, D) + 2.0
    il = r(B_, T, D) - 1.0
    B = r(B_, T, N) / (N ** 0.5)
    C = r(B_, T, N) / (N ** 0.5)
    D_skip = r(D) * 0.01
    ts = [t.to(DTYPE) for t in (u, fl, il, B, C, D_skip)]
    if requires_grad:
        ts = [t.requires_grad_(True) for t in ts]
    return ts


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    return times[n // 2], times[max(0, n // 10)], times[min(n - 1, 9 * n // 10)]


def bench(name, scan_fn, shape, warmup, iters, mode):
    B_, T, D, N = shape
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        if mode == "fwd":
            ins = _rand_inputs(B_, T, D, N, "cuda")
            with torch.no_grad():
                med, p10, p90 = _time(lambda: scan_fn(*ins), warmup, iters)
        else:
            ins = _rand_inputs(B_, T, D, N, "cuda", requires_grad=True)
            dy = torch.randn(B_, T, D, device="cuda", dtype=DTYPE)
            def step():
                y = scan_fn(*ins)
                y.backward(dy)
                for t in ins:
                    t.grad = None
            med, p10, p90 = _time(step, warmup, iters)
        peak = torch.cuda.max_memory_allocated() / 2**30
        tok_s = B_ * T / (med / 1e3)
        print(f"  {name:10s} {mode:7s} median {med:8.3f} ms  "
              f"[p10 {p10:7.3f} / p90 {p90:7.3f}]  peak {peak:5.2f} GB  "
              f"{tok_s/1e6:6.2f} Mtok/s")
        return med
    except torch.cuda.OutOfMemoryError:
        print(f"  {name:10s} {mode:7s} OOM")
    except Exception as e:
        print(f"  {name:10s} {mode:7s} FAILED: {type(e).__name__}: {e}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                    help="input tensor dtype (bf16 = autocast training regime)")
    args = ap.parse_args()
    global DTYPE
    DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    from naju.chunk_triton import naju_chunk_scan
    from naju.chunk_affine_triton import naju_affine_chunk_scan
    backends = {"chunk": naju_chunk_scan, "affine": naju_affine_chunk_scan}
    try:
        from naju.cuda import naju_cuda_scan
        naju_cuda_scan(*_rand_inputs(1, 64, 32, 16, "cuda"))   # trigger JIT
        backends["cuda"] = naju_cuda_scan
    except Exception as e:
        print(f"cuda backend unavailable ({type(e).__name__}), skipping")
    try:
        from naju.cuda.bw import naju_cuda_bw_scan
        with torch.no_grad():
            naju_cuda_bw_scan(*_rand_inputs(1, 64, 32, 16, "cuda"))
        backends["cuda_bw"] = naju_cuda_bw_scan
    except Exception as e:
        print(f"cuda_bw backend unavailable ({type(e).__name__}), skipping")

    dev = torch.cuda.get_device_name()
    free, total = torch.cuda.mem_get_info()
    print(f"{dev} | free {free/2**30:.1f} / {total/2**30:.1f} GB | "
          f"iters {args.iters} (median of)")

    shapes = [(8, 2048, 512, 64), (8, 8192, 512, 64), (2, 32768, 512, 64)]
    for shape in shapes:
        B_, T, D, N = shape
        print(f"\nB={B_} T={T} d_inner={D} N={N}")
        med = {}
        for mode in ("fwd", "fwd+bwd"):
            for name, fn in backends.items():
                if name == "cuda_bw" and mode == "fwd+bwd":
                    continue        # inference-only backend
                m = bench(name, fn, shape, args.warmup, args.iters, mode)
                if m:
                    med[(name, mode)] = m
        for mode in ("fwd", "fwd+bwd"):
            a, c = med.get(("affine", mode)), med.get(("chunk", mode))
            if a and c:
                print(f"  -> affine vs chunk {mode}: {c/a:.2f}x")


if __name__ == "__main__":
    main()
