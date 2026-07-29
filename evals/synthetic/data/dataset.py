"""Dataset loading helpers shared by train/evaluate scripts."""
import os
import torch
from torch.utils.data import Dataset


class SeqClsDataset(Dataset):
    """Loads a cached {task}_{seqlen}_{split}.pt tensor dict."""

    def __init__(self, cache_dir, task, seq_len, split):
        path = os.path.join(cache_dir, f"{task}_{seq_len}_{split}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset {path}. Generate it first with the data/ scripts."
            )
        blob = torch.load(path, weights_only=False)
        self.inputs = blob["inputs"]  # int16 [N, T]
        self.labels = blob["labels"]  # int64 [N]
        self.meta = blob.get("meta", {})  # dict of arrays, optional
        self.seq_len = seq_len

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = self.inputs[idx].to(torch.long)
        y = self.labels[idx].to(torch.long)
        item = {"input": x, "label": y}
        for k, v in self.meta.items():
            item[k] = int(v[idx])
        return item


def dataset_meta(cache_dir, task, seq_len):
    """Return the meta.json-like info saved alongside, if present."""
    path = os.path.join(cache_dir, f"{task}_{seq_len}_meta.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    return {}
