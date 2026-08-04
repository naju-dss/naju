# Naju (Native Adaptive Junction Unit)

Official implementation of **Naju**, a natively discrete gated state-space
model with independent retention and writing
([arXiv:2607.21000](https://arxiv.org/abs/2607.21000)). Contents:

```
naju/            reference implementation of the Naju model (mixer, scan
                 kernels, optimizer coupling, CPU/GPU test suites)
                 — see naju/README.md
evals/synthetic/ T1-T4 synthetic suite (data generators + training harness,
                 Naju and all baselines)
evals/lm/        WikiText-103 language-modeling harness (Naju / Mamba /
                 Mamba-2 / Transformer)
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

## Synthetic suite (T1–T4)

Run from `evals/synthetic/`. 1) Generate data at the training length (512) and the
extrapolation lengths (1024, 2048 — test splits are all that is used at those
lengths):

```bash
for L in 512 1024 2048; do
  # T1: key-value retrieval
  python data/generate_kv_retrieval.py --seq_len $L
  # T2: scattered key-value retrieval
  python data/generate_kv_hard.py --task kv_spread --num_entities 32 \
      --n_facts 32 --num_values 16 --spread --seq_len $L
  # T3: current-state tracking
  python data/generate_state_tracking.py --seq_len $L
  # T4: recency-only current-state tracking
  python data/generate_st_hard.py --spread --num_values 16 --num_entities 32 \
      --max_distractor_entities 8 --seq_len $L
done
```

2) Train (per model / task / seed; the paper reports mean±std over seeds 1–5):

```bash
python train.py --model naju  --task kv_spread --seq_len 512 --seed 1 \
    --eval_seq_lens 1024 2048
python train.py --model mamba --task kv_spread --seq_len 512 --seed 1 \
    --eval_seq_lens 1024 2048
```

Models: `naju mamba transformer mamba2 xlstm gla hgrn rwkv retnet`
(configs in `configs/*.yaml`; defaults follow the paper's stated
configurations — Naju d128/N64, Mamba reference d256/N16, 50-epoch budget).
The Mamba width/state factorial uses `--d_model`/`--d_state` overrides; the
extended-budget Transformer rows use `--epochs 150`; data-efficiency runs use
`--max_train`. Results are written as JSON under `results/` (test metrics plus
per-length extrapolation, position/distance-bucketed accuracies, and the
stale-answer rate for T3/T4).

## WikiText-103

Run from `evals/lm/`. Download the WikiText-103-raw-v1 parquet files
(HuggingFace dataset `wikitext`, config `wikitext-103-raw-v1`) into
`evals/lm/data/wikitext103_raw/` (file names in `evals/lm/data.py`). Tokenization (GPT-2
BPE, one EOS per document) runs once and is cached.

Paper protocol — 1.2B-token budget, 32,768 tokens per optimizer update,
selection on best validation PPL, per-model learning rate:

```bash
python train.py --backbone naju        --seed 1 --target_tokens 1200000000 \
    --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 4e-3
python train.py --backbone mamba       --seed 1 --target_tokens 1200000000 \
    --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 4e-3
python train.py --backbone mamba2      --seed 1 --target_tokens 1200000000 \
    --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 4e-3
python train.py --backbone transformer --seed 1 --target_tokens 1200000000 \
    --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 2e-3
```

Naju uses the paper-final configuration (the `naju` defaults) with the
chunk-parallel scan backend selected automatically on GPU. Width/state
scale-up rows use `--d_model` / `--d_state` / `--n_layers` overrides.

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
