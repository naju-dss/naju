"""Harder current-state tracking ("st_hard").

The original state_tracking marks the target's last update with a special FINAL
connective, so a model can answer by pattern-matching "FINAL + target entity"
without tracking temporal order -> trivially solved. Here we remove that
giveaway: ALL target updates use the same connective, so the model must identify
the *most recent* target update by sequence order (recency-based overwrite).

Difficulty levers (self-contained vocab; larger pools):
  * no marker  : same connective for every update (no INIT/FINAL hint)
  * spread     : scatter events across the sequence (distance to latest update)
  * num_values : larger output space
  * length extrapolation: train short, test long (latest update farther away)

Event:  [CONN entity PRED value SEP]   Query: [QUES entity NOW EOS]
Label = value of the target's most recent update.
"""
import argparse
import os
import numpy as np
import torch

# special token ids (match data/vocab.py)
PAD, BOS, SEP, QUES, NOW, LATER, LIKES, MOVED, EOS = 0, 1, 2, 5, 6, 8, 10, 11, 12
NUM_SPECIAL = 16
NUM_ENTITIES = 512
NUM_VALUES = 64
ENTITY_BASE = NUM_SPECIAL
VALUE_BASE = ENTITY_BASE + NUM_ENTITIES

EVENT_LEN = 5
QUERY_LEN = 4


def make_sample(rng, seq_len, max_updates, max_distractor_entities,
                ent_pool, n_values, spread):
    budget = seq_len - 1 - QUERY_LEN
    total_events = budget // EVENT_LEN
    if total_events < 3:
        raise ValueError(f"seq_len={seq_len} too small")

    k = int(rng.integers(2, min(max_updates, total_events) + 1))   # >=2 updates
    n_distractor = int(rng.integers(1, max_distractor_entities + 1))
    pool = rng.choice(ent_pool, size=1 + n_distractor, replace=False)
    target, distractors = pool[0], pool[1:]

    target_slots = np.sort(rng.choice(total_events, size=k, replace=False))
    target_values = rng.integers(0, n_values, size=k)
    label = int(target_values[-1])                    # latest = last in order
    prev_value = int(target_values[-2])
    last_slot = int(target_slots[-1])

    ev = np.empty((total_events, EVENT_LEN), dtype=np.int64)
    ti = 0
    for s in range(total_events):
        if ti < k and s == target_slots[ti]:
            # NO marker: same connective (LATER) for all target updates
            ev[s] = [LATER, ENTITY_BASE + target, LIKES,
                     VALUE_BASE + int(target_values[ti]), SEP]
            ti += 1
        else:
            ent = int(rng.choice(distractors))
            pred = LIKES if rng.random() < 0.5 else MOVED
            val = int(rng.integers(0, n_values))
            ev[s] = [LATER, ENTITY_BASE + ent, pred, VALUE_BASE + val, SEP]

    total_pad = budget - total_events * EVENT_LEN
    query = np.array([QUES, ENTITY_BASE + target, NOW, EOS], dtype=np.int64)
    if spread and total_pad > 0:
        cuts = np.sort(rng.integers(0, total_pad + 1, size=total_events))
        gaps = np.diff(np.concatenate([[0], cuts, [total_pad]]))
        parts = [np.array([BOS], dtype=np.int64)]
        for s in range(total_events):
            parts.append(np.full(gaps[s], PAD, dtype=np.int64)); parts.append(ev[s])
        parts.append(np.full(gaps[total_events], PAD, dtype=np.int64)); parts.append(query)
        seq = np.concatenate(parts)
    else:
        seq = np.concatenate([np.array([BOS], dtype=np.int64),
                              np.full(total_pad, PAD, dtype=np.int64),
                              ev.reshape(-1), query])
    distance = (total_events - 1 - last_slot) * EVENT_LEN + QUERY_LEN
    assert seq.shape[0] == seq_len
    return seq, label, k, distance, prev_value


def dist_bucket(distance, seq_len):
    f = distance / seq_len
    return 0 if f < 1/3 else (1 if f < 2/3 else 2)


def build_split(rng, n, seq_len, mu, mde, ent_pool, n_values, spread):
    inp = np.empty((n, seq_len), dtype=np.int16); lab = np.empty(n, dtype=np.int64)
    nu = np.empty(n, dtype=np.int64); db = np.empty(n, dtype=np.int64); pv = np.empty(n, dtype=np.int64)
    for i in range(n):
        seq, label, k, dist, prev = make_sample(rng, seq_len, mu, mde, ent_pool, n_values, spread)
        inp[i] = seq.astype(np.int16); lab[i] = label; nu[i] = k
        db[i] = dist_bucket(dist, seq_len); pv[i] = prev
    return {"inputs": torch.from_numpy(inp), "labels": torch.from_numpy(lab),
            "meta": {"num_updates": torch.from_numpy(nu),
                     "dist_bucket": torch.from_numpy(db),
                     "prev_value": torch.from_numpy(pv)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--max_updates", type=int, default=8)
    p.add_argument("--max_distractor_entities", type=int, default=20)
    p.add_argument("--num_values", type=int, default=NUM_VALUES)
    p.add_argument("--num_entities", type=int, default=NUM_ENTITIES)
    p.add_argument("--spread", action="store_true")
    p.add_argument("--task", type=str, default="st_hard")
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="data/cache")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split, n in [("train", args.num_train), ("val", args.num_val), ("test", args.num_test)]:
        rng = np.random.default_rng(args.seed + {"train": 0, "val": 1, "test": 2}[split])
        blob = build_split(rng, n, args.seq_len, args.max_updates,
                           args.max_distractor_entities, args.num_entities,
                           args.num_values, args.spread)
        path = os.path.join(args.out_dir, f"{args.task}_{args.seq_len}_{split}.pt")
        torch.save(blob, path)
        print(f"saved {path} ({n}x{args.seq_len}, updates~{int(blob['meta']['num_updates'].float().mean())})")
    meta = {"vocab_size": VALUE_BASE + args.num_values, "num_classes": args.num_values,
            "seq_len": args.seq_len, "task": args.task}
    torch.save(meta, os.path.join(args.out_dir, f"{args.task}_{args.seq_len}_meta.pt"))
    print("meta:", meta)


if __name__ == "__main__":
    main()
