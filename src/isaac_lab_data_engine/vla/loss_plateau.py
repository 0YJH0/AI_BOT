"""Opt-in loss plateau monitor; stop only AFTER a complete checkpoint is saved."""
import json
import math
import os
from pathlib import Path


class PlateauStop(Exception):
    """Successful early termination with a recoverable optimizer checkpoint."""


class LossPlateau:
    def __init__(self, window=1000, min_steps=6000, patience=3, relative=0.01):
        if window < 1 or min_steps < window or patience < 1 or not 0 < relative < 1:
            raise ValueError("Invalid plateau settings")
        self.window, self.min_steps, self.patience, self.relative = window, min_steps, patience, relative
        self.step, self.total, self.count, self.stale = 0, 0.0, 0, 0
        self.reference = None
        self.history = []
        self.stop = False

    def update(self, loss):
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; this is a failure, not convergence")
        self.step += 1
        self.total += loss
        self.count += 1
        if self.count != self.window:
            return None
        mean = self.total / self.count
        improved = self.reference is None or mean < self.reference * (1 - self.relative)
        if improved:
            self.reference, self.stale = mean, 0
        else:
            self.stale += 1
        self.stop = self.step >= self.min_steps and self.stale >= self.patience
        record = dict(step=self.step, mean_loss=mean, significant_improvement=improved,
                      reference_loss=self.reference, stale_windows=self.stale, stop_requested=self.stop)
        self.history.append(record)
        self.total, self.count = 0.0, 0
        return record


def install(train_module):
    value = os.environ.get("LEROBOT_LOSS_PLATEAU_JSON")
    if not value:
        return
    import torch
    payload = json.loads(value)
    directory = Path(payload.pop("output"))
    directory.mkdir(parents=True, exist_ok=True)
    monitor = LossPlateau(**payload)
    original_update = train_module.update_policy
    original_checkpoint = train_module.update_last_checkpoint

    def write_status(status, **extra):
        record = dict(status=status, step=monitor.step, settings=payload,
                      history=monitor.history, **extra)
        tmp = directory / "training_progress.tmp"
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        tmp.replace(directory / "training_progress.json")

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        loss = float(result[0].loss.val)
        record = monitor.update(loss)
        if monitor.step % 100 == 0 or monitor.step == 10:
            memory = {"allocated_mib": torch.cuda.memory_allocated()/2**20,
                      "reserved_mib": torch.cuda.memory_reserved()/2**20,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20}
            with (directory / "loss_intervals.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(step=monitor.step, latest_loss=loss,
                    lr=result[0].lr.val, memory=memory, window=record)) + "\n")
            write_status("TRAINING", latest_loss=loss, memory=memory)
        if record:
            print("LOSS_PLATEAU_WINDOW=" + json.dumps(record), flush=True)
        return result

    def checkpoint(path):
        original_checkpoint(path)
        if int(Path(path).name) != monitor.step:
            raise RuntimeError("Checkpoint and plateau step disagree")
        status = "EARLY_STOPPED" if monitor.stop else "CHECKPOINT_SAVED"
        write_status(status, checkpoint=str(Path(path).resolve()))
        if monitor.stop:
            print("LOSS_PLATEAU_STOP_AFTER_CHECKPOINT=" + str(path), flush=True)
            raise PlateauStop(str(path))

    train_module.update_policy = update
    train_module.update_last_checkpoint = checkpoint
