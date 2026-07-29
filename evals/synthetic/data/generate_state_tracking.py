"""Generate the Long Current-State Tracking synthetic task.

A target entity receives several state updates interleaved with distractor
events. The model must answer the *latest* state of the target. We record the
number of updates and the token distance from the final update to the query so
that accuracy and stale-answer rate can be sliced by those factors.

Run:
    python data/generate_state_tracking.py --seq_len 512 --num_train 20000 \
        --num_val 2000 --num_test 2000
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import vocab as V  # noqa: E402

EVENT_LEN = 5   # CONN entity PRED value SEP
QUERY_LEN = 4   # QUES entity NOW EOS


def make_sample(rng, seq_len, max_updates, max_distractor_entities):
    budget = seq_len - 1 - QUERY_LEN
    total_events = budget // EVENT_LEN
    if total_events < 2:
        raise ValueError(f"seq_len={seq_len} too small for the state-tracking task")

    k = int(rng.integers(1, min(max_updates, total_events) + 1))  # target updates

    # pick distinct entity ids: 1 target + distractors
    n_distractor_ent = int(rng.integers(1, max_distractor_entities + 1))
    ent_pool = rng.choice(
        V.NUM_ENTITIES, size=1 + n_distractor_ent, replace=False
    )
    target = ent_pool[0]
    distractors = ent_pool[1:]

    # which event slots belong to the target (in temporal order)
    target_slots = np.sort(rng.choice(total_events, size=k, replace=False))
    target_values = rng.integers(0, V.NUM_VALUES, size=k)
    label = int(target_values[-1])  # latest state
    prev_value = int(target_values[-2]) if k >= 2 else -1  # for stale-error rate
    last_slot = int(target_slots[-1])
    distance = (total_events - 1 - last_slot) * EVENT_LEN + QUERY_LEN

    events = np.empty((total_events, EVENT_LEN), dtype=np.int64)
    ti = 0  # index into target updates
    for s in range(total_events):
        if ti < k and s == target_slots[ti]:
            conn = V.INIT if ti == 0 else (V.FINAL if ti == k - 1 else V.LATER)
            events[s] = [conn, V.ENTITY_BASE + target, V.LIKES,
                         V.VALUE_BASE + int(target_values[ti]), V.SEP]
            ti += 1
        else:
            ent = int(rng.choice(distractors))
            pred = V.LIKES if rng.random() < 0.5 else V.MOVED
            val = int(rng.integers(0, V.NUM_VALUES))
            conn = V.LATER if rng.random() < 0.5 else V.MOVED
            events[s] = [conn, V.ENTITY_BASE + ent, pred, V.VALUE_BASE + val, V.SEP]
    events = events.reshape(-1)

    pad_count = budget - total_events * EVENT_LEN
    query = np.array(
        [V.QUES, V.ENTITY_BASE + target, V.NOW, V.EOS], dtype=np.int64
    )
    seq = np.concatenate(
        [
            np.array([V.BOS], dtype=np.int64),
            np.full(pad_count, V.PAD, dtype=np.int64),
            events,
            query,
        ]
    )
    assert seq.shape[0] == seq_len, (seq.shape[0], seq_len)
    return seq, label, k, distance, prev_value


def dist_bucket(distance, seq_len):
    frac = distance / seq_len
    return 0 if frac < 1 / 3 else (1 if frac < 2 / 3 else 2)


def build_split(rng, n, seq_len, max_updates, max_distractor_entities):
    inputs = np.empty((n, seq_len), dtype=np.int16)
    labels = np.empty(n, dtype=np.int64)
    nupd = np.empty(n, dtype=np.int64)
    dbucket = np.empty(n, dtype=np.int64)
    prevval = np.empty(n, dtype=np.int64)
    for i in range(n):
        seq, label, k, distance, prev_value = make_sample(
            rng, seq_len, max_updates, max_distractor_entities
        )
        inputs[i] = seq.astype(np.int16)
        labels[i] = label
        nupd[i] = k
        dbucket[i] = dist_bucket(distance, seq_len)
        prevval[i] = prev_value
    return {
        "inputs": torch.from_numpy(inputs),
        "labels": torch.from_numpy(labels),
        "meta": {
            "num_updates": torch.from_numpy(nupd),
            "dist_bucket": torch.from_numpy(dbucket),
            "prev_value": torch.from_numpy(prevval),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_len", type=int, required=True)
    p.add_argument("--max_updates", type=int, default=8)
    p.add_argument("--max_distractor_entities", type=int, default=20)
    p.add_argument("--num_train", type=int, default=20000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="data/cache")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    task = "state_tracking"
    for split, n in [
        ("train", args.num_train),
        ("val", args.num_val),
        ("test", args.num_test),
    ]:
        rng = np.random.default_rng(args.seed + hash(split) % 1000)
        blob = build_split(
            rng, n, args.seq_len, args.max_updates, args.max_distractor_entities
        )
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
