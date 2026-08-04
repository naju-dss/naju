# WikiText-103 language modeling

Causal LM harness (GPT-2 BPE, token-level perplexity) for Naju / Mamba /
Mamba-2 / Transformer. Run everything from this directory.

Download the WikiText-103-raw-v1 parquet files (HuggingFace dataset
`wikitext`, config `wikitext-103-raw-v1`) into `data/wikitext103_raw/`
(file names in `data.py`). Tokenization (GPT-2 BPE, one EOS per document)
runs once and is cached.

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
