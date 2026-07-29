"""Training / evaluation loop shared by train.py and evaluate.py."""
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import AverageMeter


def make_loader(dataset, batch_size, shuffle, num_workers=4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


def cosine_warmup(optimizer, total_steps, warmup_ratio):
    warmup = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def trapezoid_warmup(optimizer, total_steps, warmup_ratio, plateau_ratio):
    """Warmup -> constant-LR plateau -> cosine decay.

    Keeps the LR at its peak through the plateau so the grok window (which finishes
    early) sees sustained high LR without needing a long total horizon. warmup_ratio
    and plateau_ratio are fractions of total_steps; decay fills the remainder.
    """
    warmup = max(1, int(total_steps * warmup_ratio))
    plateau_end = min(total_steps, int(total_steps * (warmup_ratio + plateau_ratio)))

    def fn(step):
        if step < warmup:
            return step / warmup
        if step < plateau_end:
            return 1.0
        prog = (step - plateau_end) / max(1, total_steps - plateau_end)
        return 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train_one_epoch(model, loader, optimizer, scheduler, device, amp_dtype, grad_clip):
    model.train()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        loss_m.update(loss.item(), x.size(0))
        acc_m.update((logits.argmax(-1) == y).float().mean().item(), x.size(0))
    return {"loss": loss_m.avg, "acc": acc_m.avg}


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, task):
    model.eval()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    # sliced accuracy accumulators
    bucket_correct = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # key -> [correct, total]
    stale_err, stale_tot = 0, 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        pred = logits.argmax(-1)
        correct = (pred == y)
        loss_m.update(loss.item(), x.size(0))
        acc_m.update(correct.float().mean().item(), x.size(0))

        if task.startswith("kv") and "fact_pos_bucket" in batch:
            buck = batch["fact_pos_bucket"]
        elif task.startswith("st") and "dist_bucket" in batch:
            buck = batch["dist_bucket"]
        else:
            buck = None
        if buck is not None:
            for b in (0, 1, 2):
                m = buck == b
                bucket_correct[b][0] += int(correct.cpu()[m].sum())
                bucket_correct[b][1] += int(m.sum())

        if task.startswith("st") and "prev_value" in batch:
            prev = batch["prev_value"]
            valid = prev >= 0
            stale = (pred.cpu() == prev) & valid
            stale_err += int(stale.sum())
            stale_tot += int(valid.sum())

    out = {"loss": loss_m.avg, "acc": acc_m.avg}
    names = (["early", "middle", "late"] if task.startswith("kv")
             else ["near", "mid", "far"])
    for b, nm in zip((0, 1, 2), names):
        c, t = bucket_correct[b]
        out[f"acc_{nm}"] = c / t if t else float("nan")
    if task.startswith("st"):
        out["stale_error_rate"] = stale_err / stale_tot if stale_tot else float("nan")
    return out
