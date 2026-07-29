"""WikiText-103 causal LM training (GPT-2 BPE, token-level perplexity).

Paper protocol: token budget --target_tokens 1.2e9, tokens per optimizer update
matched across models (batch_size * seq_len * grad_accum = 32768), selection on
best validation perplexity, test evaluated once at the end.

    python train.py --backbone naju        --seed 1 --target_tokens 1200000000 \
        --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 4e-3
    python train.py --backbone mamba2      --seed 1 --target_tokens 1200000000 \
        --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 4e-3
    python train.py --backbone transformer --seed 1 --target_tokens 1200000000 \
        --seq_len 1024 --batch_size 8 --grad_accum 4 --lr 2e-3
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                    # evals/ (for lm.*)
from lm.data import load_wikitext103_raw, WT103RawBlocks, WT103_VOCAB_SIZE
from lm.model import build_lm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


def cosine_warmup(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def run_epoch(model, loader, dev, vocab_size, opt=None, sched=None, amp=torch.bfloat16,
              grad_clip=1.0, accum=1, max_batches=None, max_tokens=None):
    """One pass; returns (mean CE nats/token, target tokens processed, optimizer steps)."""
    train = opt is not None
    model.train(train)
    ce_sum, tok, steps, micro = 0.0, 0, 0, 0
    if train:
        opt.zero_grad()
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        if train and max_tokens is not None and tok >= max_tokens:
            break
        x = batch["input"].to(dev, non_blocking=True)
        y = batch["target"].to(dev, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if train:
            (loss / accum).backward()
            micro += 1
            if micro == accum:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step(); sched.step(); opt.zero_grad()
                steps += 1; micro = 0
        n = y.numel()
        ce_sum += loss.item() * n; tok += n
    if train and micro:                                # flush trailing partial group
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); sched.step(); opt.zero_grad()
        steps += 1
    return ce_sum / max(tok, 1), tok, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="naju",
                    choices=["naju", "mamba", "mamba2", "transformer"])
    ap.add_argument("--d_init", type=float, default=0.01, help="Naju feedthrough init")
    ap.add_argument("--n_heads", type=int, default=4, help="transformer attention heads")
    ap.add_argument("--epochs", type=int, default=10,
                    help="max epochs (and budget if --target_tokens 0)")
    ap.add_argument("--target_tokens", type=int, default=0,
                    help="training budget in target tokens; 0 = epochs mode")
    ap.add_argument("--grad_accum", type=int, default=1,
                    help="micro-batches per optimizer update (equalize tokens/update)")
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_state", type=int, default=None, help="override mixer d_state")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val_batches", type=int, default=None,
                    help="cap val batches during training")
    ap.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile(mode='max-autotune'); ~20-25%% faster "
                         "training, identical protocol otherwise (paper numbers "
                         "were produced without it)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = WT103_VOCAB_SIZE

    cfg = dict(backbone=args.backbone, d_model=args.d_model, n_layers=args.n_layers,
               dropout=args.dropout, vocab_size=vocab_size, n_heads=args.n_heads,
               d_init=args.d_init)
    if args.d_state is not None:
        cfg["d_state"] = args.d_state
    model = build_lm(cfg).to(dev)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")
    nparam = sum(p.numel() for p in model.parameters())
    n_embed = model.embed.weight.numel()               # LM head is tied to this tensor
    print(f"wt103_raw backbone={args.backbone} vocab={vocab_size} "
          f"params={nparam/1e6:.2f}M (embed {n_embed/1e6:.2f}M / backbone "
          f"{(nparam-n_embed)/1e6:.2f}M) L={args.seq_len} d_model={args.d_model} "
          f"layers={args.n_layers} seed={args.seed} dev={dev}", flush=True)

    tr_arr = load_wikitext103_raw("train")
    va_arr = load_wikitext103_raw("validation")
    te_arr = load_wikitext103_raw("test")
    print(f"wt103_raw tokens: train={tr_arr.size/1e6:.1f}M val={va_arr.size/1e3:.0f}K "
          f"test={te_arr.size/1e3:.0f}K", flush=True)
    tr_ds = WT103RawBlocks(tr_arr, args.seq_len, "train", seed=args.seed)
    tr = make_loader(tr_ds, args.batch_size, False, 4)   # order shuffled inside set_epoch
    va = make_loader(WT103RawBlocks(va_arr, args.seq_len, "val"), args.batch_size, False, 2)
    te = make_loader(WT103RawBlocks(te_arr, args.seq_len, "test"), args.batch_size, False, 2)

    tokens_per_update = args.batch_size * args.seq_len * args.grad_accum
    epoch_tokens = len(tr.dataset) * args.seq_len
    if args.target_tokens:
        total_steps = math.ceil(args.target_tokens / tokens_per_update)
        max_epochs = math.ceil(args.target_tokens / epoch_tokens)
    else:
        total_steps = args.epochs * math.ceil(len(tr) / args.grad_accum)
        max_epochs = args.epochs
    print(f"budget: target_tokens={args.target_tokens or 'epochs-mode'} "
          f"tokens/update={tokens_per_update} steps={total_steps} max_epochs={max_epochs}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = cosine_warmup(opt, total_steps, args.warmup_ratio)

    best, best_state = float("inf"), None              # tracked in nats (lower=better)
    done_tokens, done_steps, t_start = 0, 0, time.perf_counter()
    for ep in range(max_epochs):
        t0 = time.perf_counter()
        if tr.dataset.split == "train":
            tr.dataset.set_epoch(ep)
        budget_left = (args.target_tokens - done_tokens) if args.target_tokens else None
        tr_nats, ep_tok, ep_steps = run_epoch(model, tr, dev, vocab_size, opt, sched,
                                              grad_clip=args.grad_clip, accum=args.grad_accum,
                                              max_tokens=budget_left)
        done_tokens += ep_tok; done_steps += ep_steps
        va_nats, _, _ = run_epoch(model, va, dev, vocab_size, max_batches=args.val_batches)
        if va_nats < best:
            best = va_nats
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{ep+1:>3}/{max_epochs}] train_ppl={math.exp(tr_nats):.4f} "
              f"val_ppl={math.exp(va_nats):.4f} best_ppl={math.exp(best):.4f} "
              f"tokens={done_tokens/1e6:.0f}M ({time.perf_counter()-t0:.0f}s)", flush=True)
        if args.target_tokens and done_tokens >= args.target_tokens:
            break
    train_time = time.perf_counter() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    test_nats, test_tok, _ = run_epoch(model, te, dev, vocab_size)
    test_ppl, best_ppl = math.exp(test_nats), math.exp(best)
    print(f"TEST ppl={test_ppl:.4f}  (best val ppl {best_ppl:.4f})", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tsuf = f"-{args.tag}" if args.tag else ""
    out = os.path.join(args.out_dir, f"wt103_{args.backbone}{tsuf}_seed{args.seed}.json")
    meta = {
        "dataset_config": "wikitext-103-raw-v1", "backbone": args.backbone,
        "d_init": args.d_init if args.backbone == "naju" else None,
        "n_heads": args.n_heads if args.backbone == "transformer" else None,
        "seed": args.seed, "tag": args.tag,
        "test_ppl": test_ppl, "best_val_ppl": best_ppl,
        "test_nats": test_nats, "best_val_nats": best, "test_target_tokens": test_tok,
        "params": nparam, "embed_params": n_embed, "backbone_params": nparam - n_embed,
        "vocab_size": vocab_size, "seq_len": args.seq_len,
        "d_model": args.d_model, "n_layers": args.n_layers,
        "epochs_run": ep + 1, "target_tokens": args.target_tokens,
        "tokens_processed": done_tokens, "optimizer_steps": done_steps,
        "effective_tokens_per_update": tokens_per_update,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio, "dropout": args.dropout,
        "train_wall_s": round(train_time, 1),
        "torch": torch.__version__,
    }
    json.dump(meta, open(out, "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
