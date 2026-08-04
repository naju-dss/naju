# Naju (Native Adaptive Junction Unit)

Official implementation of **Naju**, a natively discrete gated state-space
model with independent retention and writing
([arXiv:2607.21000](https://arxiv.org/abs/2607.21000)). Contents:

```
naju/            reference implementation of the Naju model (mixer, scan
                 kernels, optimizer coupling, CPU/GPU test suites)
                 — see naju/README.md
evals/synthetic/ T1-T4 synthetic suite (data generators + training harness,
                 Naju and all baselines) — see evals/synthetic/README.md
evals/lm/        WikiText-103 language-modeling harness (Naju / Mamba /
                 Mamba-2 / Transformer) — see evals/lm/README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Core requirements are `torch` (with a matching `triton`, bundled in CUDA
builds), `numpy`, `pyyaml`. Optional per-baseline extras: `mamba-ssm`
(Mamba / Mamba-2), `flash-linear-attention` (GLA / HGRN / RWKV-6 / RetNet),
`xlstm` (xLSTM), `transformers` + `pyarrow` (WikiText-103 tokenization). All
baseline imports are lazy — Naju-only runs need none of the extras.

Optional sanity checks (naju kernel test suites):

```bash
python naju/tests/test_equivalence.py   # CPU
python naju/tests/test_gpu_smoke.py     # GPU: all scan backends vs reference
python naju/tests/test_affine_chunk.py  # GPU: experimental affine backend
```

## Experiments

Two experiment sets reproduce the paper's results; each harness has its own
README with the data-preparation steps and exact training commands:

- **Synthetic suite (T1–T4)** — the four synthetic long-sequence memory
  tasks (key-value retrieval, scattered retrieval, state tracking,
  recency-only tracking), Naju and all baselines, with length
  extrapolation. See `evals/synthetic/README.md`.
- **WikiText-103** — causal language modeling under the paper's
  1.2B-token protocol (Naju / Mamba / Mamba-2 / Transformer, token-level
  PPL). See `evals/lm/README.md`.

## Not included

The MQAR and Long Range Arena experiments build on third-party harnesses
(zoology; the LRA data pipeline); to keep this repository self-contained they
are not bundled.

## Citation

```bibtex
@article{lim2026naju,
  title={Naju: A Native Discrete State-Space Model with Independent Retention
         and Writing for Long-Sequence Memory},
  author={Lim, Hyuk and Yoon, Seunghyun},
  journal={arXiv preprint arXiv:2607.21000},
  year={2026}
}
```
