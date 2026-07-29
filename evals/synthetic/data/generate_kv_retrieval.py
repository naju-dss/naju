"""Generate the Long Key-Value Retrieval synthetic task.

Each sample is a long sequence of "<KEY> entity <IS> value <SEP>" facts followed
by "<QUES> <KEY> query_entity <EOS>". The label is the value of the queried
entity. We record the position bucket (early/middle/late) of the relevant fact.

Run:
    python data/generate_kv_retrieval.py --seq_len 512 --num_train 20000 \
        --num_val 2000 --num_test 2000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import vocab as V  # noqa: E402

FACT_LEN = 5   # KEY entity IS value SEP
QUERY_LEN = 4  # QUES KEY entity EOS


def make_sample(rng, seq_len, num_entities):
    budget = seq_len - 1 - QUERY_LEN  # minus BOS and query
    max_facts = budget // FACT_LEN
    n_facts = min(num_entities, max_facts, V.NUM_ENTITIES)
    if n_facts < 1:
        raise ValueError(f"seq_len={seq_len} too small for the KV task")

    # distinct entities and their random values
    ent_ids = rng.choice(V.NUM_ENTITIES, size=n_facts, replace=False)
    val_ids = rng.integers(0, V.NUM_VALUES, size=n_facts)

    facts = np.empty((n_facts, FACT_LEN), dtype=np.int64)
    facts[:, 0] = V.KEY
    facts[:, 1] = V.ENTITY_BASE + ent_ids
    facts[:, 2] = V.IS
    facts[:, 3] = V.VALUE_BASE + val_ids
    facts[:, 4] = V.SEP
    facts = facts.reshape(-1)

    # pick the queried fact uniformly -> position bucket
    q = int(rng.integers(0, n_facts))
    frac = q / max(n_facts - 1, 1)
    bucket = 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)
    label = int(val_ids[q])

    pad_count = budget - n_facts * FACT_LEN
    query = np.array(
        [V.QUES, V.KEY, V.ENTITY_BASE + ent_ids[q], V.EOS], dtype=np.int64
    )
    seq = np.concatenate(
        [
            np.array([V.BOS], dtype=np.int64),
            np.full(pad_count, V.PAD, dtype=np.int64),
            facts,
            query,
        ]
    )
    assert seq.shape[0] == seq_len, (seq.shape[0], seq_len)
    return seq, label, bucket, n_facts


def build_split(rng, n, seq_len, num_entities):
    inputs = np.empty((n, seq_len), dtype=np.int16)
    labels = np.empty(n, dtype=np.int64)
    buckets = np.empty(n, dtype=np.int64)
    nfacts = np.empty(n, dtype=np.int64)
    for i in range(n):
        seq, label, bucket, nf = make_sample(rng, seq_len, num_entities)
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
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--num_entities", type=int, default=400)
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="data/cache")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    task = "kv_retrieval"
    for split, n in [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test),
    ]:
        rng = np.random.default_rng(args.seed + hash(split) % 1000)
        blob = build_split(rng, n, args.seq_len, args.num_entities)
        path = os.path.join(args.out_dir, f"{task}_{args.seq_len}_{split}.pt")
        torch.save(blob, path)
        print(f"wrote {path}  inputs={tuple(blob['inputs'].shape)}")

    meta = {
        "vocab_size": V.VOCAB_SIZE,
        "num_classes": V.NUM_CLASSES,
        "seq_len": args.seq_len,
        "task": task,
    }
    torch.save(meta, os.path.join(args.out_dir, f"{task}_{args.seq_len}_meta.pt"))
    print("meta:", meta)


if __name__ == "__main__":
    main()
