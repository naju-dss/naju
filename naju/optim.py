"""Gate-aware AdamW builder — the optimizer half of the gate reparameterization.

The gate reparameterization of the paper's scale-up study is optimizer-matched:
logit 1/sqrt(d_inner) rescaling in the mixer (NajuMixer(gate_reparam=True))
together with a matched per-group learning rate, weight decay, and Adam eps.
The two halves are always used together (A/B no-op verified across widths), so
here the coupling is automatic and cannot be mismatched:

    build_optimizer(model, lr, weight_decay)

detects mixers with gate_reparam=True and gives their gate projections
(f_proj / i_proj weights) their own param group with

    lr           lr * sqrt(d_inner)
    weight_decay wd / sqrt(d_inner)   (AdamW applies decay as lr*wd per step;
                                       without this the gate weights are
                                       annihilated)
    eps          1e-8 / sqrt(d_inner)

If no mixer has gate_reparam=True (the default), this returns a plain AdamW.
"""
import torch

from naju.mixer import NajuMixer


def build_optimizer(model, lr, weight_decay=0.1):
    d_inners = {m.d_inner for m in model.modules()
                if isinstance(m, NajuMixer) and m.gate_reparam}
    if not d_inners:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if len(d_inners) > 1:
        raise ValueError(f"mixed d_inner across gate_reparam mixers: {d_inners}")
    mult = d_inners.pop() ** 0.5
    gate_ids = {id(p) for m in model.modules()
                if isinstance(m, NajuMixer) and m.gate_reparam
                for p in (m.f_proj.weight, m.i_proj.weight)}
    gate = [p for p in model.parameters() if id(p) in gate_ids]
    rest = [p for p in model.parameters() if id(p) not in gate_ids]
    return torch.optim.AdamW(
        [{"params": rest},
         {"params": gate, "lr": lr * mult,
          "weight_decay": weight_decay / mult, "eps": 1e-8 / mult}],
        lr=lr, weight_decay=weight_decay)
