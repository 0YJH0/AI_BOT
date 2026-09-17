"""Train RGB-D keypoint localization plus geometric 3-D reconstruction."""

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
from torch.utils.data import DataLoader, Dataset

from isaac_lab_data_engine.vision import (
    RGBDFrameDataset,
    RGBDKeypointPositionNet,
    RGBDPoseModelCfg,
)


class _CachedFrames(Dataset):
    """Materialize resized frames once so compressed HDF5 is not decoded every epoch."""

    KEYS = (
        "input",
        "target_position_camera",
        "resized_intrinsics",
        "target_pixel",
        "surface_depth_at_target",
    )

    def __init__(self, source: RGBDFrameDataset) -> None:
        values = [source[index] for index in range(len(source))]
        self.tensors = {key: torch.stack([value[key] for value in values]) for key in self.KEYS}

    def __len__(self) -> int:
        return self.tensors["input"].shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--image-height", type=int, default=60)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _augment(inputs: torch.Tensor) -> torch.Tensor:
    inputs = inputs.clone()
    scale = torch.empty((inputs.shape[0], 1, 1, 1), device=inputs.device).uniform_(0.85, 1.15)
    inputs[:, :3] = (inputs[:, :3] * scale + torch.randn_like(inputs[:, :3]) * 0.012).clamp(
        -1.0, 1.0
    )
    inputs[:, 3:] = (inputs[:, 3:] + torch.randn_like(inputs[:, 3:]) * 0.003).clamp(
        -1.0, 1.0
    )
    return inputs


def _loss(
    model: RGBDKeypointPositionNet,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    augment: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    inputs = batch["input"].to(device, non_blocking=True)
    if augment:
        inputs = _augment(inputs)
    target_position = batch["target_position_camera"].to(device, non_blocking=True)
    target_pixel = batch["target_pixel"].to(device, non_blocking=True)
    logits, normalized_depth = model(inputs)
    predicted_pixel = model.keypoint_pixels(logits)

    image_scale = torch.tensor(
        (model.config.image_width, model.config.image_height), device=device
    )
    coordinate_loss = functional.smooth_l1_loss(
        predicted_pixel / image_scale,
        target_pixel / image_scale,
        beta=0.02,
    )
    surface_depth = batch["surface_depth_at_target"].to(device, non_blocking=True)
    depth_target = (target_position[:, 2] - surface_depth - model.depth_mean) / model.depth_std
    depth_loss = functional.smooth_l1_loss(normalized_depth, depth_target, beta=0.25)
    heat_height, heat_width = logits.shape[-2:]
    target_x = torch.floor(target_pixel[:, 0] * heat_width / model.config.image_width).long()
    target_y = torch.floor(target_pixel[:, 1] * heat_height / model.config.image_height).long()
    target_x.clamp_(0, heat_width - 1)
    target_y.clamp_(0, heat_height - 1)
    classification_loss = functional.cross_entropy(
        logits.flatten(1), target_y * heat_width + target_x
    )
    total = 3.0 * coordinate_loss + depth_loss + 0.05 * classification_loss
    return total, {
        "coordinate_loss": float(coordinate_loss.detach()),
        "depth_loss": float(depth_loss.detach()),
        "classification_loss": float(classification_loss.detach()),
    }


@torch.inference_mode()
def _evaluate(
    model: RGBDKeypointPositionNet, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    errors: list[torch.Tensor] = []
    pixel_errors: list[torch.Tensor] = []
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        intrinsics = batch["resized_intrinsics"].to(device, non_blocking=True)
        target = batch["target_position_camera"].to(device, non_blocking=True)
        logits, normalized_depth = model(inputs)
        pixels = model.keypoint_pixels(logits)
        prediction = model.position_from_outputs(
            inputs, logits, normalized_depth, intrinsics
        )
        errors.append((prediction - target).cpu())
        pixel_errors.append(
            torch.linalg.vector_norm(
                pixels - batch["target_pixel"].to(device, non_blocking=True), dim=-1
            ).cpu()
        )
    error = torch.cat(errors)
    pixel_error = torch.cat(pixel_errors)
    absolute = error.abs()
    distance = torch.linalg.vector_norm(error, dim=-1)
    return {
        "frame_count": float(distance.numel()),
        "pixel_mean": float(pixel_error.mean()),
        "pixel_p95": float(torch.quantile(pixel_error, 0.95)),
        "mae_x_m": float(absolute[:, 0].mean()),
        "mae_y_m": float(absolute[:, 1].mean()),
        "mae_z_m": float(absolute[:, 2].mean()),
        "mean_error_m": float(distance.mean()),
        "rmse_m": float(torch.sqrt(distance.square().mean())),
        "median_error_m": float(torch.quantile(distance, 0.50)),
        "p95_error_m": float(torch.quantile(distance, 0.95)),
        "within_1cm": float((distance <= 0.01).float().mean()),
        "within_2cm": float((distance <= 0.02).float().mean()),
        "within_5cm": float((distance <= 0.05).float().mean()),
    }


def _save(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = RGBDPoseModelCfg(
        image_height=args.image_height,
        image_width=args.image_width,
        base_channels=args.base_channels,
    )

    source = {
        split: RGBDFrameDataset(args.dataset_dir, split, config=config, augment=False)
        for split in ("train", "validation", "test")
    }
    print("Caching preprocessed RGB-D frames...", flush=True)
    cached = {split: _CachedFrames(dataset) for split, dataset in source.items()}
    for dataset in source.values():
        dataset.close()
    train_depth = (
        cached["train"].tensors["target_position_camera"][:, 2]
        - cached["train"].tensors["surface_depth_at_target"]
    )
    model = RGBDKeypointPositionNet(
        config, depth_mean=train_depth.mean(), depth_std=train_depth.std().clamp_min(1.0e-4)
    ).to(device)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            pin_memory=device.type == "cuda",
        )
        for split, dataset in cached.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    best_error = math.inf
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        samples = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _loss(model, batch, device, augment=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = batch["input"].shape[0]
            running += float(loss.detach()) * count
            samples += count
        scheduler.step()
        metrics = _evaluate(model, loaders["validation"], device)
        row = {
            "epoch": float(epoch),
            "train_loss": running / samples,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_pixel={metrics['pixel_mean']:.2f} "
            f"val_mean_cm={metrics['mean_error_m'] * 100.0:.2f} "
            f"val_2cm={metrics['within_2cm'] * 100.0:.1f}%",
            flush=True,
        )
        if metrics["mean_error_m"] < best_error:
            best_error = metrics["mean_error_m"]
            payload = model.checkpoint_payload(metrics)
            payload.update({"epoch": epoch, "dataset_dir": str(args.dataset_dir.resolve())})
            _save(args.output_dir / "best_model.pt", payload)

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, loaders["test"], device)
    summary = {
        "status": "PHASE12_KEYPOINT_TRAINING_SUCCESS",
        "checkpoint": str((args.output_dir / "best_model.pt").resolve()),
        "best_epoch": int(checkpoint["epoch"]),
        "elapsed_seconds": time.perf_counter() - start,
        "split_frames": {key: len(value) for key, value in cached.items()},
        "validation_metrics": checkpoint["metrics"],
        "test_metrics": test_metrics,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("PHASE12_KEYPOINT_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
