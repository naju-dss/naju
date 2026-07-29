# Naju

Reference implementation of **Naju**, a natively discrete gated state-space
model: decoupled forget/input sigmoid gates drive a diagonal recurrence with no
ZOH discretization, read out through a normalized state readout plus a learnable
feedthrough, and gated by a GLU-style output gate. Constructor defaults are the
paper's final configuration — `NajuMixer(d_model)` is exactly the model used in
the paper's experiments.

## Model

Notation: $u^{(l)}$ is the residual stream (block input/output), $h$ the
convolved content branch, $n$ the token index, $N$ = `d_state`.

```
ũ = RMSNorm(u^{(l)})                          pre-norm (NajuBlock)
[x_raw, z_n] = W_in ũ_n                       in_proj: d_model -> 2 d_inner
h_n  = SiLU(DWConv(x_raw))                    content branch (causal dwconv, k=4)
[B_n, C_n] = W_x h_n                          x_proj: d_inner -> 2 d_state
a_n  = W_f h_n + DWConv_f(h_n) + b_f          forget logit  (b_f init +5)
c_n  = W_i h_n + DWConv_i(h_n) + b_i          input  logit  (b_i init -2)
x_n  = σ(a_n) ⊙ x_{n-1} + σ(c_n) ⊙ B_n h_n    scan (run with D=0)
cx_n = (1/√N) C_n · x_n                       readout normalization
y_n  = cx_n + D ⊙ h_n                         feedthrough (D learnable, init 0.01)

u^{(l+1)}_n = u^{(l)}_n + W_o(y_n ⊙ SiLU(z_n))    block output (residual)
```

The current input reaches the next layer by two routes: the gated feedthrough
$D h_n$ inside the SSM branch (passes through the z-gate and $W_o$), and the
clean identity path of the outer residual.

Two elements are fixed parts of the block, not options:

- **Readout normalization** $cx \leftarrow cx/\sqrt{N}$ (attention-style) —
  keeps the readout magnitude independent of `d_state`. Note this is a
  per-task readout *scale correction*, not a stability device: the recurrence
  is BIBO-stable regardless (the sigmoid pole is < 1).
- **z-gate** $y \odot \mathrm{SiLU}(z)$ — the GLU/SwiGLU-style output gate,
  the analogue of an LSTM output gate.

## Options (`NajuMixer`)

Defaults reproduce the paper configuration; change them only for ablations.

| Argument | Default | Meaning |
|---|---|---|
| `d_state` | 64 | state dimension $N$ (the LM scale-up also uses 128) |
| `expand` | 2 | `d_inner = expand * d_model` |
| `d_conv` | 4 | content-branch causal depthwise-conv kernel |
| `gate_conv_kernel` | 4 | gate local-conv kernel (gates = proj + conv + bias) |
| `forget_bias_init` | +5.0 | preserve-first init: retention can approach 1; sets a well-conditioned decay pole ($1-f \lesssim 1/L$ maps bias to memory horizon) |
| `input_bias_init` | −2.0 | write gate starts mostly closed (selective writing) |
| `d_init` | 0.01 | init scale of the feedthrough $D$ (1.0 = Mamba convention). Small init lets the feedthrough start negligible and grow only where the task wants it — large init creates a shortcut that suppresses recall learning |
| `gate_rank` | None | low-rank factorization of the gate projections. Ablation only: the paper's diagnostic shows the gates are not the latency bottleneck (the scan is). Incompatible with `gate_reparam` |
| `gate_reparam` | False | full optimizer-matched gate reparameterization (below) |
| `scan_backend` | None | scan kernel selection (below) |

### Gate reparameterization (`gate_reparam=True`)

From the paper's scale-up study: (1) model side — gate weights scaled up by
$\sqrt{d_{inner}}$ at init and the pre-activation divided back down (the
initial forward is identical; only the Adam update dynamics change), (2)
optimizer side — `build_optimizer` detects the flag and gives the gate
projections a matched group: lr × $\sqrt{d_{inner}}$, weight decay ÷
$\sqrt{d_{inner}}$ (AdamW applies decay as lr·wd — without this the gate
weights are annihilated), Adam ε ÷ $\sqrt{d_{inner}}$. The two halves are one
mechanism and cannot be enabled separately. Verified as a no-op A/B across
widths; its purpose is letting the inherited learning rate carry to larger
widths.

```python
opt = build_optimizer(model, lr=4e-3, weight_decay=0.1)   # auto-detects the flag
```

## Scan backends (`scan.py`) — one per role

| Backend | Role | Requirements | Notes |
|---|---|---|---|
| `reference` | correctness ground truth | any device, no deps | sequential loop; slow |
| `chunk` | **training standard** | GPU + Triton | SSD-style chunk-parallel; exact for `f_logit ≥ −5`, auto-falls back to `cuda` beyond |
| `cuda` | chunk's fallback; memory-light long-T training | GPU + nvcc (JIT) | exact for any logit; sequential training path with checkpoint-recompute backward, chunk-parallel inference path |
| `cuda_bw` | **inference standard** | GPU + nvcc (JIT) | warp-shuffle kernels, 2.5–3.2× faster inference; the paper's efficiency tables use this backend under `no_grad` |

Selection order: explicit argument > `NAJU_SCAN_BACKEND` env > auto (`chunk`
if CUDA is available, else `reference`).

## torch.compile (recommended for training)

Measured in the paper's training-efficiency study (BF16, chunk backend):
`torch.compile(mode="max-autotune")` raises Naju training throughput by
**+22–24%, length-invariant** (the eager path leaves the gate-logit sums,
readout gain, and output-gate products unfused; the compiler recovers them).
The scan is a custom autograd Function, so a graph break occurs at its
boundary — the gain above already includes that.

Skip compilation for numerics debugging and equivalence testing (fusion changes
floating-point op order), and pass `dynamic=True` (or fix the sequence length)
to avoid recompiles under variable `T`.

## Usage

```python
import torch
from naju import NajuLM, build_optimizer

model = NajuLM(vocab_size=50257, d_model=2048, n_layers=6, d_state=128)
model = torch.compile(model.cuda(), mode="max-autotune")
opt = build_optimizer(model, lr=4e-3, weight_decay=0.1)
```

`NajuBlock(use_ckpt=True)` enables block-level gradient checkpointing
(identical numerics, activations recomputed in backward) — prefer it over
shrinking the batch when memory-bound at long sequence lengths.

Checkpoints trained with the original research code load directly
(`strict=True`): parameter names are identical.

## Tests

```bash
python naju/tests/test_equivalence.py   # CPU: model equivalence, reparam, optimizer coupling
python naju/tests/test_gpu_smoke.py     # GPU: all scan backends vs reference (fwd + grads)
```

The CPU suite covers the model math (plus an equivalence check against the
original research implementation that auto-skips when that repo is not
present). The GPU suite verifies `chunk`, `cuda`, and `cuda_bw` against the
reference scan — forward, all input gradients, the deep-logit fallback route,
and the `no_grad` inference path — in under 100 MiB of GPU memory.
