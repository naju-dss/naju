"""CPU test suite for the naju package (no GPU required).

  1. Equivalence vs the original research implementation — forward and grads,
     weights transferred strict=True (proves original checkpoints load into
     NajuMixer). Skipped automatically unless the original research repo is
     available (path via NAJU_ORIG_REPO, for the authors' environment).
  2. gate_reparam init-forward identity (reparam changes dynamics, not init fwd).
  3. gate_reparam <-> build_optimizer auto-coupling (matched lr/wd/eps groups).
  4. gate_rank smoke (shapes / backward).

The GPU kernels (chunk/cuda/cuda_bw) are not exercised here — validate them
with a one-time chunk-vs-reference comparison on a GPU (see README).

Usage: python naju/tests/test_equivalence.py
"""
import os
import sys

os.environ["NAJU_D_INIT"] = "0.01"          # the original reads this at init
os.environ.pop("NAJU_SCAN_BACKEND", None)   # isolate from ambient env

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))      # package root
_ORIG_REPO = os.path.expanduser(os.environ.get("NAJU_ORIG_REPO", ""))
if _ORIG_REPO and os.path.isdir(_ORIG_REPO):
    sys.path.insert(0, _ORIG_REPO)

import torch

torch.manual_seed(0)
B, T, D_MODEL, D_STATE = 2, 48, 32, 64
X = torch.randn(B, T, D_MODEL, dtype=torch.float32)


def maxdiff(a, b):
    return (a - b).abs().max().item()


def test_vs_original():
    if not (_ORIG_REPO and os.path.isdir(_ORIG_REPO)):
        print("1. equivalence vs original research code: SKIPPED "
              "(set NAJU_ORIG_REPO to the original repo to enable)")
        return
    # module path inside the authors' original (non-public) research repo;
    # only reachable when NAJU_ORIG_REPO points at it
    import models.naju_dss_v2_rg_core as rg
    from naju.mixer import NajuMixer
    from naju.scan import naju_scan_reference

    rg.naju_dss_v2_scan = naju_scan_reference   # CPU-safe backend for the original

    orig = rg.NajuDSSv2RGMixer(
        D_MODEL, d_state=D_STATE, expand=2, gate_conv_kernel=4,
        forget_bias_init=5.0, input_bias_init=-2.0,
        readout_input="skip", use_z_gate=True, use_readout_gate=False,
        readout_sqrt_norm=True)
    clean = NajuMixer(D_MODEL, scan_backend="reference")   # defaults = final config
    clean.load_state_dict(orig.state_dict(), strict=True)  # v1 ckpt compat

    xo = X.clone().requires_grad_(True)
    xc = X.clone().requires_grad_(True)
    yo = orig(xo); yc = clean(xc)
    d = maxdiff(yo, yc)
    assert d < 1e-6, f"forward mismatch {d}"
    yo.sum().backward(); yc.sum().backward()
    for (n, po), pc in zip(orig.named_parameters(), clean.parameters()):
        if po.grad is not None:
            gd = maxdiff(po.grad, pc.grad)
            assert gd < 1e-5, f"grad mismatch on {n}: {gd}"
    gd_in = maxdiff(xo.grad, xc.grad)
    assert gd_in < 1e-5, f"input-grad mismatch {gd_in}"
    print(f"1. equivalence vs original research code: fwd {d:.2e}, "
          f"grads OK, strict state_dict load OK")


def test_gate_reparam_init_forward():
    from naju.mixer import NajuMixer

    torch.manual_seed(2)
    base = NajuMixer(D_MODEL, scan_backend="reference")
    rep = NajuMixer(D_MODEL, gate_reparam=True, scan_backend="reference")
    sd = base.state_dict()
    s = base.d_inner ** 0.5
    sd["f_proj.weight"] = sd["f_proj.weight"] * s
    sd["i_proj.weight"] = sd["i_proj.weight"] * s
    rep.load_state_dict(sd, strict=True)
    d = maxdiff(base(X), rep(X))
    assert d < 1e-5, f"gate_reparam init-forward mismatch {d}"
    print(f"2. gate_reparam init-forward identity: {d:.2e}")


def test_optimizer_coupling():
    from naju.lm import NajuLM
    from naju.optim import build_optimizer

    lm = NajuLM(vocab_size=64, d_model=D_MODEL, n_layers=2, d_state=8)
    opt = build_optimizer(lm, lr=4e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 1, "no reparam -> plain AdamW"

    lm = NajuLM(vocab_size=64, d_model=D_MODEL, n_layers=2, d_state=8,
                gate_reparam=True)
    opt = build_optimizer(lm, lr=4e-3, weight_decay=0.1)
    mult = (2 * D_MODEL) ** 0.5
    g = opt.param_groups[1]
    assert len(opt.param_groups) == 2 and len(g["params"]) == 4  # 2 layers x f/i
    assert abs(g["lr"] - 4e-3 * mult) < 1e-12
    assert abs(g["weight_decay"] - 0.1 / mult) < 1e-12
    assert abs(g["eps"] - 1e-8 / mult) < 1e-20
    try:
        from naju.mixer import NajuMixer
        NajuMixer(D_MODEL, gate_reparam=True, gate_rank=8)
        raise AssertionError("gate_reparam+gate_rank should raise")
    except ValueError:
        pass
    print("3. gate_reparam <-> optimizer coupling: auto-detected groups, "
          "lr*sqrt(d_inner), wd & eps matched; gate_rank combination rejected")


def test_gate_rank_smoke():
    from naju.mixer import NajuMixer

    m = NajuMixer(D_MODEL, gate_rank=8, scan_backend="reference")
    y = m(X)
    assert y.shape == (B, T, D_MODEL)
    y.sum().backward()
    assert m.f_proj[0].weight.grad is not None
    print("4. gate_rank low-rank gates: forward/backward OK")


if __name__ == "__main__":
    test_vs_original()
    test_gate_reparam_init_forward()
    test_optimizer_coupling()
    test_gate_rank_smoke()
    print("ALL PASS")
