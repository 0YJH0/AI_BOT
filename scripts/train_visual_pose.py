"""Train and evaluate the Phase 12 RGB-D object-position estimator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from isaac_lab_data_engine.vision import (
    RGBDFrameDataset,
    RGBDObjectPositionNet,
    RGBDPoseModelCfg,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: RGBDFrameDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.inference_mode()
def _evaluate(
    model: RGBDObjectPositionNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    errors: list[torch.Tensor] = []
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target_position_camera"].to(device, non_blocking=True)
        errors.append((model.predict_position(inputs) - targets).detach().cpu())
    if not errors:
        raise ValueError("Evaluation split contains no RGB-D frames")
    error = torch.cat(errors)
    absolute = error.abs()
    distance = torch.linalg.vector_norm(error, dim=-1)
    return {
        "frame_count": float(distance.numel()),
        "mae_x_m": float(absolute[:, 0].mean()),
        "mae_y_m": float(absolute[:, 1].mean()),
        "mae_z_m": float(absolute[:, 2].mean()),
        "mean_error_m": float(distance.mean()),
        "rmse_m": float(torch.sqrt(torch.mean(distance.square()))),
        "median_error_m": float(torch.quantile(distance, 0.50)),
        "p95_error_m": float(torch.quantile(distance, 0.95)),
        "within_1cm": float((distance <= 0.01).float().mean()),
        "within_2cm": float((distance <= 0.02).float().mean()),
        "within_5cm": float((distance <= 0.05).float().mean()),
    }


def _atomic_checkpoint(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")
    device = torch.device(args.device)
    config = RGBDPoseModelCfg(
        image_height=args.image_height,
        image_width=args.image_width,
        base_channels=args.base_channels,
    )
    datasets = {
        split: RGBDFrameDataset(
            args.dataset_dir,
            split,
            config=config,
            augment=split == "train",
        )
        for split in ("train", "validation", "test")
    }
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"The {split} split contains no RGB-D frames")
    target_mean, target_std = datasets["train"].target_statistics()
    model = RGBDObjectPositionNet(config, target_mean, target_std).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    train_loader = _loader(
        datasets["train"], args.batch_size, args.workers, shuffle=True, device=device
    )
    validation_loader = _loader(
        datasets["validation"], args.batch_size, args.workers, shuffle=False, device=device
    )
    test_loader = _loader(
        datasets["test"], args.batch_size, args.workers, shuffle=False, device=device
    )

    history: list[dict[str, float]] = []
    best_error = math.inf
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_frames = 0
        for batch in train_loader:
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target_position_camera"].to(device, non_blocking=True)
            normalized_targets = (targets - model.target_mean) / model.target_std
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = functional.smooth_l1_loss(predictions, normalized_targets, beta=0.25)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * inputs.shape[0]
            total_frames += inputs.shape[0]
        scheduler.step()

        validation = _evaluate(model, validation_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / total_frames,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"val_mean_cm={validation['mean_error_m'] * 100.0:.3f} "
            f"val_p95_cm={validation['p95_error_m'] * 100.0:.3f}",
            flush=True,
        )
        if validation["mean_error_m"] < best_error:
            best_error = validation["mean_error_m"]
            payload = model.checkpoint_payload(validation)
            payload.update({"epoch": epoch, "dataset_dir": str(args.dataset_dir.resolve())})
            _atomic_checkpoint(args.output_dir / "best_model.pt", payload)

    best_checkpoint = torch.load(
        args.output_dir / "best_model.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - start_time
    summary = {
        "status": "PHASE12_VISUAL_TRAINING_SUCCESS",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "checkpoint": str((args.output_dir / "best_model.pt").resolve()),
        "best_epoch": int(best_checkpoint["epoch"]),
        "elapsed_seconds": elapsed,
        "split_frames": {key: len(value) for key, value in datasets.items()},
        "target_mean_camera_m": target_mean.tolist(),
        "target_std_camera_m": target_std.tolist(),
        "validation_metrics": best_checkpoint["metrics"],
        "test_metrics": test_metrics,
        "config": vars(args) | {"dataset_dir": str(args.dataset_dir), "output_dir": str(args.output_dir)},
    }
    with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, default=str)
    print("PHASE12_TRAINING_SUMMARY=" + json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    for dataset in datasets.values():
        dataset.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
