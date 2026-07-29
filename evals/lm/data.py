"""WikiText-103-raw-v1 data for the LM experiments (GPT-2 BPE, token-level PPL).

Raw (untokenized) articles, GPT-2 BPE per document, GPT-2 EOS (50256) appended
after each document: D1,EOS,D2,EOS,...  Each split is an independent stream.
Tokenization is done once and cached (uint16 npy + JSON manifest).

Expected layout (see README for the download instructions):
    lm/data/wikitext103_raw/
        train-00000-of-00002.parquet  train-00001-of-00002.parquet
        validation-00000-of-00001.parquet  test-00000-of-00001.parquet
"""
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
WT103_RAW_DIR = os.path.join(_HERE, "data", "wikitext103_raw")
WT103_VOCAB_SIZE = 50257  # GPT-2 BPE
GPT2_EOS = 50256
_WT103_RAW_PARQUET = {
    "train": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
    "validation": ["validation-00000-of-00001.parquet"],
    "test": ["test-00000-of-00001.parquet"],
}


def _gpt2_tokenizer():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def _is_doc_heading(line):
    # A new article starts with a level-1 heading " = Title = " (level-2+ is " = = ... = = ").
    s = line.strip()
    return s.startswith("= ") and s.endswith(" =") and not s.startswith("= =")


def load_wikitext103_raw(split, cache=True):
    """GPT-2 BPE ids for a raw-v1 split, one EOS after each document (uint16)."""
    split = {"val": "validation"}.get(split, split)
    assert split in _WT103_RAW_PARQUET, split
    cdir = os.path.join(WT103_RAW_DIR, "cache")
    cpath = os.path.join(cdir, f"wt103_raw_gpt2_{split}.npy")
    if cache and os.path.exists(cpath):
        return np.load(cpath)                        # cache hit: no pyarrow/tokenizer needed
    import pyarrow.parquet as pq                     # only for the one-time cache build
    os.makedirs(cdir, exist_ok=True)
    tok = _gpt2_tokenizer()

    docs, cur = [], []
    for fn in _WT103_RAW_PARQUET[split]:
        for line in pq.read_table(os.path.join(WT103_RAW_DIR, fn)).column("text").to_pylist():
            if line and _is_doc_heading(line) and cur:
                docs.append("".join(cur)); cur = []
            cur.append(line)
    if cur:
        docs.append("".join(cur))

    ids = []
    B = 256                                            # tokenizer batches for speed
    for i in range(0, len(docs), B):
        for enc in tok(docs[i:i + B], add_special_tokens=False)["input_ids"]:
            ids.extend(enc); ids.append(GPT2_EOS)
    arr = np.asarray(ids, dtype=np.uint16)
    if cache:
        np.save(cpath, arr)
        _write_wt103_manifest(cdir, split, arr, len(docs))
    return arr


def _write_wt103_manifest(cdir, split, arr, n_docs):
    mpath = os.path.join(cdir, "manifest.json")
    m = json.load(open(mpath)) if os.path.exists(mpath) else {
        "dataset": "WikiText-103", "dataset_config": "wikitext-103-raw-v1",
        "tokenizer": "gpt2", "vocab_size": WT103_VOCAB_SIZE,
        "document_separator": "gpt2_eos", "cache_dtype": "uint16"}
    m[f"{split}_tokens"] = int(arr.size)
    m[f"{split}_documents"] = int(n_docs)
    json.dump(m, open(mpath, "w"), indent=2)


class WT103RawBlocks(Dataset):
    """Block sampling over a token stream.

    train: per-epoch global offset o_e ~ Uniform{0..L} drawn from (seed, epoch), stream
    cut into non-overlapping (L+1)-blocks after the offset, block ORDER shuffled by the
    same rng — every token seen ~once/epoch and identical (offset, order) across models
    at the same seed. val/test: offset 0, deterministic order, incomplete tail dropped.
    Call set_epoch(e) each training epoch.
    """

    def __init__(self, arr, seq_len, split, seed=0):
        self.arr = arr
        self.seq_len = seq_len
        self.split = split
        self.seed = seed
        self.n = int(arr.shape[0])
        # fixed length = worst-case offset L so DataLoader length is epoch-stable
        L = seq_len
        self.num = (self.n - L - 1) // L if split == "train" else (self.n - 1) // L
        self.starts = np.arange(self.num, dtype=np.int64) * L
        if split == "train":
            self.set_epoch(0)

    def set_epoch(self, epoch):
        assert self.split == "train"
        rng = np.random.default_rng((self.seed, epoch))
        off = int(rng.integers(0, self.seq_len + 1))
        self.starts = off + np.arange(self.num, dtype=np.int64) * self.seq_len
        rng.shuffle(self.starts)

    def __len__(self):
        return self.num

    def __getitem__(self, i):
        s = int(self.starts[i])
        chunk = self.arr[s: s + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return {"input": x, "target": y}
