"""Train one model on one synthetic task (T1-T4) at one training length.

The paper protocol trains at seq_len 512 and evaluates length extrapolation on
test sets generated at 1024/2048 (--eval_seq_lens). Tasks:

    T1  kv_retrieval    key-value retrieval (retention)
    T2  kv_spread       scattered key-value retrieval (retention + distance)
    T3  state_tracking  current-state tracking (overwriting, marked final update)
    T4  st_hard         recency-only current-state tracking (overwriting)

Example (T2, Naju, seed 1):
    python data/generate_kv_hard.py --task kv_spread --num_entities 32 \
        --n_facts 32 --num_values 16 --spread --seq_len 512
    python train.py --model naju --task kv_spread --seq_len 512 --seed 1 \
        --eval_seq_lens 1024 2048
"""
import argparse
import json
import os
import time

import torch

from common import set_seed, get_device, load_config, count_params
from data.dataset import SeqClsDataset, dataset_meta
from models import build_model, MODEL_NAMES
from engine import make_loader, cosine_warmup, trapezoid_warmup, train_one_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--task", required=True,
                   choices=["kv_retrieval", "kv_spread", "state_tracking", "st_hard"])
    p.add_argument("--eval_seq_lens", type=int, nargs="+", default=None,
                   help="extra test seq_lens to evaluate (length extrapolation)")
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="data/cache")
    p.add_argument("--out_dir", type=str, default="results")
    # overrides
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--d_state", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--tag", type=str, default=None,
                   help="suffix appended to the result filename")
    p.add_argument("--max_train", type=int, default=None,
                   help="subsample training set (data-efficiency runs / smoke)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--save_ckpt", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config or f"configs/{args.model}.yaml"
    cfg = load_config(cfg_path)
    for k in ("epochs", "batch_size", "lr", "d_model", "d_state", "warmup_ratio"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    set_seed(args.seed)
    device = get_device()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "off": None}[args.amp]

    meta = dataset_meta(args.cache_dir, args.task, args.seq_len)
    if not meta:
        raise SystemExit(
            f"No dataset for {args.task} seq_len={args.seq_len}. "
            f"Generate it first (see data/generate_*.py and the README)."
        )
    vocab_size, num_classes = meta["vocab_size"], meta["num_classes"]

    train_ds = SeqClsDataset(args.cache_dir, args.task, args.seq_len, "train")
    if args.max_train:
        train_ds.inputs = train_ds.inputs[: args.max_train]
        train_ds.labels = train_ds.labels[: args.max_train]
        train_ds.meta = {k: v[: args.max_train] for k, v in train_ds.meta.items()}
    val_ds = SeqClsDataset(args.cache_dir, args.task, args.seq_len, "val")
    test_ds = SeqClsDataset(args.cache_dir, args.task, args.seq_len, "test")

    bs = cfg["batch_size"]
    train_loader = make_loader(train_ds, bs, True, args.num_workers)
    val_loader = make_loader(val_ds, bs, False, args.num_workers)
    test_loader = make_loader(test_ds, bs, False, args.num_workers)

    model = build_model(args.model, cfg, vocab_size, num_classes).to(device)
    n_params = count_params(model)
    print(f"model={args.model} params={n_params/1e6:.2f}M device={device} "
          f"amp={args.amp} d_model={cfg.get('d_model')} compile={args.compile}")
    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    total_steps = cfg["epochs"] * len(train_loader)
    _wu = cfg.get("warmup_ratio", 0.05)
    if cfg.get("plateau_ratio") is not None:
        sched = trapezoid_warmup(opt, total_steps, _wu, cfg["plateau_ratio"])
    else:
        sched = cosine_warmup(opt, total_steps, _wu)
    grad_clip = cfg.get("grad_clip", 1.0)

    best_val, best_state = -1.0, None
    history = []
    for epoch in range(cfg["epochs"]):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, opt, sched, device,
                             amp_dtype, grad_clip)
        ep_time = time.perf_counter() - t0
        va = evaluate(model, val_loader, device, amp_dtype, args.task)
        history.append({"epoch": epoch, "train": tr, "val": va,
                        "epoch_time_s": ep_time})
        print(f"[{epoch+1:>2}/{cfg['epochs']}] "
              f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.3f} "
              f"val_acc={va['acc']:.3f} time={ep_time:.1f}s", flush=True)
        if va["acc"] > best_val:
            best_val = va["acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, amp_dtype, args.task)
    print("TEST:", {k: round(v, 4) for k, v in test_metrics.items()})

    # length extrapolation: evaluate the trained model on longer test sets
    extrapolation = {}
    for L in (args.eval_seq_lens or []):
        if L == args.seq_len:
            continue
        m = dataset_meta(args.cache_dir, args.task, L)
        if not m:
            print(f"[extrap] no test data for seq_len={L}, skipping")
            continue
        ext_ds = SeqClsDataset(args.cache_dir, args.task, L, "test")
        ext_loader = make_loader(ext_ds, bs, False, args.num_workers)
        em = evaluate(model, ext_loader, device, amp_dtype, args.task)
        extrapolation[str(L)] = em
        print(f"EXTRAP seq{L}:", {k: round(v, 4) for k, v in em.items()})

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.model}_{args.task}_seq{args.seq_len}_seed{args.seed}"
    if args.tag:
        tag = f"{tag}_{args.tag}"
    result = {
        "tag": tag, "model": args.model, "task": args.task,
        "seq_len": args.seq_len, "seed": args.seed,
        "params": n_params, "best_val_acc": best_val,
        "test": test_metrics, "extrapolation": extrapolation,
        "config": cfg, "history": history,
    }
    with open(os.path.join(args.out_dir, tag + ".json"), "w") as f:
        json.dump(result, f, indent=2)
    if args.save_ckpt:
        torch.save(model.state_dict(), os.path.join(args.out_dir, tag + ".pt"))
    print("saved", os.path.join(args.out_dir, tag + ".json"))


if __name__ == "__main__":
    main()
