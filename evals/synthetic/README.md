# Synthetic suite (T1–T4)

Data generators and training harness for the paper's four synthetic
long-sequence memory tasks, with Naju and all baselines. Run everything
from this directory.

1) Generate data at the training length (512) and the extrapolation lengths
(1024, 2048 — test splits are all that is used at those lengths):

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
