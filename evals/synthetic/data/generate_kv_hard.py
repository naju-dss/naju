"""Harder key-value retrieval ("kv_hard").

Differences from kv_retrieval (which caps at NUM_ENTITIES=32 facts and pads the
rest, making length non-discriminative):
  * large pools: 512 entities, 64 values  -> bigger output space, more bindings
  * DENSE packing: facts fill the whole sequence (n_facts ~ seq_len/5), so
    longer sequences carry MORE key-value pairs and a longer query distance.
This makes the task (a) harder at fixed length and (b) a real length-
extrapolation test (train short -> test long sees more facts, farther query).

Self-contained vocab (reuses the special-token ids, own entity/value pools):
  specials 0..15  | entities 16..527 (512) | values 528..591 (64)
  vocab_size = 592, num_classes = 64
Sequence: [BOS] (KEY entity IS value SEP)xN (QUES KEY query_entity EOS), pad<5.
"""
import argparse
import os
import numpy as np
import torch

# special token ids (must match data/vocab.py)
PAD, BOS, SEP, KEY, IS, QUES, EOS = 0, 1, 2, 3, 4, 5, 12
NUM_SPECIAL = 16
NUM_ENTITIES = 512
NUM_VALUES = 64
ENTITY_BASE = NUM_SPECIAL
VALUE_BASE = ENTITY_BASE + NUM_ENTITIES
VOCAB_SIZE = VALUE_BASE + NUM_VALUES      # 592
NUM_CLASSES = NUM_VALUES                  # 64

FACT_LEN = 5    # KEY entity IS value SEP
QUERY_LEN = 4   # QUES KEY entity EOS


def make_sample(rng, seq_len, n_facts_cap=None, spread=False, ent_pool=NUM_ENTITIES):
    budget = seq_len - 1 - QUERY_LEN
    max_facts = budget // FACT_LEN
    n_facts = min(max_facts, ent_pool)                  # DENSE default
    if n_facts_cap is not None:
        n_facts = min(n_facts, n_facts_cap)             # fixed moderate count
    if n_facts < 2:
        raise ValueError(f"seq_len={seq_len} too small")

    ent_ids = rng.choice(ent_pool, size=n_facts, replace=False)
    val_ids = rng.integers(0, NUM_VALUES, size=n_facts)

    facts = np.empty((n_facts, FACT_LEN), dtype=np.int64)
    facts[:, 0] = KEY
    facts[:, 1] = ENTITY_BASE + ent_ids
    facts[:, 2] = IS
    facts[:, 3] = VALUE_BASE + val_ids
    facts[:, 4] = SEP

    q = int(rng.integers(0, n_facts))
    frac = q / max(n_facts - 1, 1)
    bucket = 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)
    label = int(val_ids[q])
    query = np.array([QUES, KEY, ENTITY_BASE + ent_ids[q], EOS], dtype=np.int64)

    total_pad = budget - n_facts * FACT_LEN
    if spread and total_pad > 0:
        # scatter facts across the sequence: random PAD gaps before each fact
        cuts = np.sort(rng.integers(0, total_pad + 1, size=n_facts))
        gaps = np.diff(np.concatenate([[0], cuts, [total_pad]]))  # n_facts+1 gaps
        parts = [np.array([BOS], dtype=np.int64)]
        for k in range(n_facts):
            parts.append(np.full(gaps[k], PAD, dtype=np.int64))
            parts.append(facts[k])
        parts.append(np.full(gaps[n_facts], PAD, dtype=np.int64))
        parts.append(query)
        seq = np.concatenate(parts)
    else:
        seq = np.concatenate([
            np.array([BOS], dtype=np.int64),
            np.full(total_pad, PAD, dtype=np.int64),
            facts.reshape(-1), query,
        ])
    assert seq.shape[0] == seq_len, (seq.shape[0], seq_len)
    return seq, label, bucket, n_facts


def build_split(rng, n, seq_len, n_facts_cap=None, spread=False, ent_pool=NUM_ENTITIES):
    inputs = np.empty((n, seq_len), dtype=np.int16)
    labels = np.empty(n, dtype=np.int64)
    buckets = np.empty(n, dtype=np.int64)
    nfacts = np.empty(n, dtype=np.int64)
    for i in range(n):
        seq, label, bucket, nf = make_sample(rng, seq_len, n_facts_cap, spread, ent_pool)
        inputs[i] = seq.astype(np.int16)
        labels[i] = label
        buckets[i] = bucket
        nfacts[i] = nf
    return {
        "inputs": torch.from_numpy(inputs),
        "labels": torch.from_numpy(labels),
        "meta": {
            "fact_pos_bucket": torch.from_numpy(buckets),
            "n_facts": torch.from_numpy(nfacts),
        },
    }


def main():
    global NUM_VALUES, NUM_CLASSES
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--num_values", type=int, default=NUM_VALUES)
    p.add_argument("--n_facts", type=int, default=None, help="cap facts (moderate difficulty)")
    p.add_argument("--spread", action="store_true", help="scatter facts across the sequence")
    p.add_argument("--num_entities", type=int, default=NUM_ENTITIES, help="entity pool size (small=more repetition=learnable)")
    p.add_argument("--task", type=str, default="kv_hard")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="data/cache")
    args = p.parse_args()

    NUM_VALUES = args.num_values
    NUM_CLASSES = NUM_VALUES

    os.makedirs(args.out_dir, exist_ok=True)
    task = args.task
    for split, n in [("train", args.num_train), ("val", args.num_val), ("test", args.num_test)]:
        rng = np.random.default_rng(args.seed + {"train": 0, "val": 1, "test": 2}[split])
        blob = build_split(rng, n, args.seq_len, args.n_facts, args.spread, args.num_entities)
        path = os.path.join(args.out_dir, f"{task}_{args.seq_len}_{split}.pt")
        torch.save(blob, path)
        print(f"saved {path}  ({n} x {args.seq_len}, n_facts~{int(blob['meta']['n_facts'].float().mean())})")

    meta = {"vocab_size": VALUE_BASE + NUM_VALUES, "num_classes": NUM_VALUES,
            "seq_len": args.seq_len, "task": task}
    torch.save(meta, os.path.join(args.out_dir, f"{task}_{args.seq_len}_meta.pt"))
    print("meta:", meta)


if __name__ == "__main__":
    main()
