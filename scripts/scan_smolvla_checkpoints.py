"""Scan SmolVLA PEFT checkpoints on one deterministic frame set."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.checkpoint import (
    checkpoint_chunk_size,
    load_lerobot_policy_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "exports/lerobot/pick_lift_vla_images_baseline_v1",
    )
    parser.add_argument(
        "--repo-id", default="local/isaac-pick-lift-vla-images-baseline-v1"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40,
        help="Fixed sample count; use 0 to evaluate every frame in the split.",
    )
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(1, 2),
        default=1,
        help="Inference batch size. Batch 2 automatically retries the checkpoint at batch 1 on CUDA OOM.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N processed frames; use 0 to disable intermediate progress.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output after validating that the saved scan configuration matches.",
    )
    parser.add_argument("--checkpoint-steps", type=int, nargs="*")
    parser.add_argument(
        "--position-target-mode",
        choices=("absolute", "delta"),
        default="absolute",
        help=(
            "Interpret action[:3] as an absolute robot-frame EEF target or as "
            "a current-EEF-relative delta. Delta scans use the zero action as "
            "their no-motion baseline."
        ),
    )
    parser.add_argument("--threshold-position-mae-cm", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def image_tensor(torch, value):
    image = to_numpy(value)
    if image.shape[-1] == 3:
        image = np.moveaxis(image, -1, 0)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
    if tensor.max() > 1.0:
        tensor.div_(255.0)
    return tensor.unsqueeze(0)


def quaternion_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted_norm = np.linalg.norm(predicted)
    target_norm = np.linalg.norm(target)
    if predicted_norm < 1e-8 or target_norm < 1e-8:
        return 180.0
    dot = abs(float(np.dot(predicted / predicted_norm, target / target_norm)))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0))))


def iter_index_batches(indices: Sequence[int], batch_size: int) -> Iterator[list[int]]:
    """Yield stable, ordered batches without changing the selected frame set."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(indices), batch_size):
        yield [int(index) for index in indices[start : start + batch_size]]


def deterministic_action_noise(
    torch,
    sample_indices: Sequence[int],
    *,
    seed: int,
    chunk_size: int,
    max_action_dim: int,
    device,
):
    """Build per-frame diffusion noise independent of inference batch boundaries.

    The previous scanner reset the RNG to seed + sample_index immediately
    before every policy call. Generating every row separately preserves that
    checkpoint-to-checkpoint pairing when multiple independent frames are
    concatenated into a batch.
    """

    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    cuda_devices = []
    if use_cuda:
        device_index = torch.device(device).index
        cuda_devices = [
            torch.cuda.current_device() if device_index is None else device_index
        ]
    rows = []
    # Avoid leaking per-frame reseeding into callers while retaining the exact
    # legacy torch.normal stream on the selected device.
    with torch.random.fork_rng(devices=cuda_devices):
        for sample_index in sample_indices:
            sample_seed = int(seed) + int(sample_index)
            torch.manual_seed(sample_seed)
            if use_cuda:
                torch.cuda.manual_seed_all(sample_seed)
            rows.append(
                torch.normal(
                    mean=0.0,
                    std=1.0,
                    size=(1, int(chunk_size), int(max_action_dim)),
                    dtype=torch.float32,
                    device=device,
                )
            )
    if not rows:
        raise ValueError("sample_indices must not be empty")
    return torch.cat(rows, dim=0)


def build_inference_batch(torch, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")
    return {
        "observation.images.table": torch.cat(
            [image_tensor(torch, sample["observation.images.table"]) for sample in samples],
            dim=0,
        ),
        "observation.state": torch.stack(
            [
                torch.from_numpy(
                    to_numpy(sample["observation.state"]).astype(np.float32)
                )
                for sample in samples
            ],
            dim=0,
        ),
        "task": [
            str(
                sample.get(
                    "task", "Pick up the red cube and lift it 10 centimeters."
                )
            )
            for sample in samples
        ],
    }


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and (
        "cuda" in message or "cublas" in message or "cudnn" in message
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace a JSON report only after a complete temporary file is written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def make_scan_config(
    args: argparse.Namespace,
    checkpoint_steps: Sequence[int],
    indices: Sequence[int],
    dataset_length: int,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "checkpoint_root": str(args.checkpoint_root.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "repo_id": str(args.repo_id),
        "dataset_length": int(dataset_length),
        "samples_requested": int(args.samples),
        "sample_count": len(indices),
        "indices": [int(index) for index in indices],
        "seed": int(args.seed),
        "device": str(args.device),
        "batch_size_requested": int(args.batch_size),
        "checkpoint_steps": [int(step) for step in checkpoint_steps],
        "position_target_mode": str(
            getattr(args, "position_target_mode", "absolute")
        ),
        "threshold_position_mae_cm": float(args.threshold_position_mae_cm),
    }


def validate_resume_report(
    report: dict[str, Any], expected_config: dict[str, Any]
) -> list[dict[str, Any]]:
    saved_config = report.get("scan_config")
    if saved_config is None:
        raise ValueError(
            "Resume report has no scan_config. It was produced by an older scanner; "
            "use a new output path or rerun without --resume."
        )
    if saved_config != expected_config:
        keys = sorted(set(saved_config) | set(expected_config))
        mismatches = [
            key for key in keys if saved_config.get(key) != expected_config.get(key)
        ]
        raise ValueError(
            "Resume configuration does not match the current scan: "
            + ", ".join(mismatches)
        )

    selected_steps = set(expected_config["checkpoint_steps"])
    results = report.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Resume report results must be a list")
    completed_steps = [int(item["step"]) for item in results]
    if len(completed_steps) != len(set(completed_steps)):
        raise ValueError("Resume report contains duplicate checkpoint steps")
    unexpected = sorted(set(completed_steps) - selected_steps)
    if unexpected:
        raise ValueError(f"Resume report contains unrequested steps: {unexpected}")
    order = {step: offset for offset, step in enumerate(expected_config["checkpoint_steps"])}
    return sorted(results, key=lambda item: order[int(item["step"])])


def build_report(
    args: argparse.Namespace,
    scan_config: dict[str, Any],
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ordered_results = list(results)
    completed_steps = [int(item["step"]) for item in ordered_results]
    completed_set = set(completed_steps)
    pending_steps = [
        step
        for step in scan_config["checkpoint_steps"]
        if step not in completed_set
    ]
    eligible = [
        item
        for item in ordered_results
        if item["position_mae_cm"] <= args.threshold_position_mae_cm
    ]
    best = (
        min(ordered_results, key=lambda item: item["position_mae_cm"])
        if ordered_results
        else None
    )
    split_name = args.dataset_root.name.lower()
    held_out = split_name in {"validation", "val", "test"}
    return {
        "status": (
            "VLA_CHECKPOINT_SCAN_COMPLETE"
            if not pending_steps
            else "VLA_CHECKPOINT_SCAN_PARTIAL"
        ),
        "scan_config": scan_config,
        "dataset_split": split_name,
        "evaluation_scope": (
            "fixed deterministic samples from an episode-disjoint held-out split"
            if held_out
            else "fixed deterministic samples from the current exported dataset"
        ),
        "contamination_warning": (
            None
            if held_out
            else (
                "This split is not named validation/test; confirm it was not used for "
                "training before interpreting the result as generalization."
            )
        ),
        "seed": args.seed,
        "indices": scan_config["indices"],
        "threshold_position_mae_cm": args.threshold_position_mae_cm,
        "threshold_passed": bool(eligible),
        "best_checkpoint": best,
        "completed_steps": completed_steps,
        "pending_steps": pending_steps,
        "results": ordered_results,
    }


def discover_peft_checkpoints(
    checkpoint_root: Path,
    requested_steps: Sequence[int] | None = None,
) -> list[Path]:
    """Find PEFT checkpoints and reject incomplete explicit scan requests."""

    available = {
        int(path.name): path / "pretrained_model"
        for path in checkpoint_root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]")
        if (path / "pretrained_model" / "adapter_model.safetensors").is_file()
    }
    if requested_steps:
        normalized = [int(step) for step in requested_steps]
        if len(normalized) != len(set(normalized)):
            raise ValueError("--checkpoint-steps contains duplicate steps")
        missing = sorted(set(normalized) - set(available))
        if missing:
            raise RuntimeError(
                "Requested PEFT checkpoints are missing or incomplete under "
                f"{checkpoint_root}: {missing}"
            )
        return [available[step] for step in sorted(normalized)]
    return [available[step] for step in sorted(available)]


def load_policy(checkpoint: Path, device: str):
    from peft import PeftConfig, PeftModel
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_config = load_lerobot_policy_config(checkpoint)
    peft_config = PeftConfig.from_pretrained(str(checkpoint))
    base_path = peft_config.base_model_name_or_path
    if not base_path:
        raise ValueError(f"No base model in {checkpoint / 'adapter_config.json'}")
    policy = SmolVLAPolicy.from_pretrained(base_path, config=policy_config)
    policy = PeftModel.from_pretrained(policy, str(checkpoint), config=peft_config)
    policy.to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocess, postprocess


def evaluate_checkpoint(
    *,
    torch,
    checkpoint_step: int,
    policy,
    preprocess,
    postprocess,
    dataset,
    selected_indices: Sequence[int],
    cached_samples: dict[int, Any] | None,
    episode_levels: dict[int, str],
    seed: int,
    batch_size: int,
    progress_every: int,
    checkpoint_started: float,
    position_target_mode: str = "absolute",
) -> dict[str, Any]:
    if position_target_mode not in {"absolute", "delta"}:
        raise ValueError(
            "position_target_mode must be either 'absolute' or 'delta'"
        )
    config = policy.config
    chunk_size = int(config.chunk_size)
    max_action_dim = int(config.max_action_dim)
    n_obs_steps = int(getattr(config, "n_obs_steps", 1))
    if batch_size > 1 and n_obs_steps != 1:
        raise ValueError(
            "Batched independent-frame scanning requires policy.config.n_obs_steps == 1"
        )

    position_errors: list[float] = []
    orientation_errors: list[float] = []
    gripper_errors: list[float] = []
    gripper_correct: list[bool] = []
    axis_errors: list[np.ndarray] = []
    position_baseline_errors: list[float] = []
    level_position_errors: dict[str, list[float]] = {}
    phase_errors: dict[str, list[np.ndarray]] = {}
    horizon_position_errors: list[list[float]] = [
        [] for _ in range(chunk_size)
    ]
    horizon_axis_errors: list[list[np.ndarray]] = [
        [] for _ in range(chunk_size)
    ]
    horizon_orientation_errors: list[list[float]] = [
        [] for _ in range(chunk_size)
    ]
    horizon_gripper_errors: list[list[float]] = [
        [] for _ in range(chunk_size)
    ]
    horizon_gripper_correct: list[list[bool]] = [
        [] for _ in range(chunk_size)
    ]
    horizon_finite: list[int] = [0 for _ in range(chunk_size)]
    finite = 0
    inference_total_ms = 0.0
    inference_batches = 0
    total = len(selected_indices)
    processed_count = 0
    progress_started = time.perf_counter()
    next_progress = progress_every if progress_every > 0 else total + 1

    for batch_indices in iter_index_batches(selected_indices, batch_size):
        batch_samples = [
            (
                cached_samples[sample_index]
                if cached_samples is not None
                else dataset[sample_index]
            )
            for sample_index in batch_indices
        ]
        raw_batch = build_inference_batch(torch, batch_samples)
        processed = preprocess(raw_batch)
        state_tensor = processed["observation.state"]
        noise = deterministic_action_noise(
            torch,
            batch_indices,
            seed=seed,
            chunk_size=chunk_size,
            max_action_dim=max_action_dim,
            device=state_tensor.device,
        )

        reset_policy = getattr(policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        infer_started = time.perf_counter()
        with torch.inference_mode():
            chunk = postprocess(policy.predict_action_chunk(processed, noise=noise))
        if state_tensor.device.type == "cuda":
            torch.cuda.synchronize(state_tensor.device)
        batch_inference_ms = (time.perf_counter() - infer_started) * 1000.0
        inference_total_ms += batch_inference_ms
        inference_batches += 1
        chunk_array = to_numpy(chunk)

        for batch_offset, sample in enumerate(batch_samples):
            predicted_chunk = chunk_array[batch_offset].astype(np.float32)
            target_chunk = to_numpy(sample["action"]).astype(np.float32)
            four_dim = predicted_chunk.shape[-1] == 4
            if predicted_chunk.shape[-1] not in (4, 8) or target_chunk.shape[-1] != predicted_chunk.shape[-1]:
                raise ValueError("Checkpoint and dataset action contracts differ")
            gripper_index = 3 if four_dim else 7
            if target_chunk.ndim == 1:
                target_chunk = target_chunk[None, :]
            padding = sample.get("action_is_pad")
            if padding is None:
                padding_mask = np.zeros(target_chunk.shape[0], dtype=bool)
            else:
                padding_mask = to_numpy(padding).astype(bool).reshape(-1)
            available_horizon = min(
                chunk_size,
                predicted_chunk.shape[0],
                target_chunk.shape[0],
                padding_mask.shape[0],
            )
            for horizon_index in range(available_horizon):
                if bool(padding_mask[horizon_index]):
                    continue
                horizon_prediction = predicted_chunk[horizon_index]
                horizon_target = target_chunk[horizon_index]
                horizon_finite[horizon_index] += int(
                    np.isfinite(horizon_prediction).all()
                )
                horizon_position_errors[horizon_index].append(
                    float(
                        np.linalg.norm(
                            horizon_prediction[:3] - horizon_target[:3]
                        )
                    )
                )
                horizon_axis_errors[horizon_index].append(
                    np.abs(horizon_prediction[:3] - horizon_target[:3])
                )
                if not four_dim:
                    horizon_orientation_errors[horizon_index].append(
                        quaternion_error_deg(horizon_prediction[3:7], horizon_target[3:7]))
                horizon_gripper_errors[horizon_index].append(
                    float(abs(horizon_prediction[gripper_index] - horizon_target[gripper_index]))
                )
                horizon_gripper_correct[horizon_index].append(
                    bool(
                        (horizon_prediction[gripper_index] >= 0.0)
                        == (horizon_target[gripper_index] >= 0.0)
                    )
                )

            predicted = predicted_chunk[0]
            target = target_chunk[0]
            state = to_numpy(sample["observation.state"]).astype(np.float32)
            diagnostic_phase = sample.get("training.controller_phase")
            phase = (int(to_numpy(diagnostic_phase).reshape(-1)[0]) if diagnostic_phase is not None
                     else int(state[29]) if len(state) == 30 else -1)
            phase_errors.setdefault(str(phase), []).append(np.asarray([
                np.linalg.norm(predicted[:3]-target[:3])*100,
                abs(predicted[2]-target[2])*100,
                float((predicted[gripper_index] >= 0) == (target[gripper_index] >= 0))]))
            finite += int(np.isfinite(predicted).all())
            position_error = float(np.linalg.norm(predicted[:3] - target[:3]))
            position_errors.append(position_error)
            axis_errors.append(np.abs(predicted[:3] - target[:3]))
            if position_target_mode == "delta":
                # A zero delta is the exact no-motion baseline for current-EEF-
                # relative action labels.
                baseline_position = np.zeros(3, dtype=np.float32)
            else:
                # Joint position/velocity occupy 0:18; EEF xyz is fixed at
                # 18:21. Visual-conditioned datasets append values afterwards.
                baseline_position = state[18:21]
            position_baseline_errors.append(
                float(np.linalg.norm(baseline_position - target[:3]))
            )
            episode_index = int(
                to_numpy(sample["episode_index"]).reshape(-1)[0]
            )
            level = episode_levels.get(episode_index, "unknown")
            level_position_errors.setdefault(level, []).append(position_error)
            if not four_dim:
                orientation_errors.append(quaternion_error_deg(predicted[3:7], target[3:7]))
            gripper_errors.append(float(abs(predicted[gripper_index] - target[gripper_index])))
            gripper_correct.append(
                bool((predicted[gripper_index] >= 0.0) == (target[gripper_index] >= 0.0))
            )

        processed_count += len(batch_indices)
        should_report = (
            processed_count == total
            or (progress_every > 0 and processed_count >= next_progress)
        )
        if should_report:
            elapsed = time.perf_counter() - progress_started
            eta = (
                elapsed / processed_count * (total - processed_count)
                if processed_count
                else 0.0
            )
            progress = {
                "step": checkpoint_step,
                "processed": processed_count,
                "total": total,
                "percent": processed_count / total * 100.0 if total else 100.0,
                "elapsed_s": elapsed,
                "eta_s": eta,
                "batch_size": batch_size,
            }
            print(
                "CHECKPOINT_PROGRESS="
                + json.dumps(progress, ensure_ascii=False),
                flush=True,
            )
            while next_progress <= processed_count:
                next_progress += progress_every if progress_every > 0 else total + 1

    baseline_mae_cm = float(np.mean(position_baseline_errors) * 100.0)
    action_horizon_metrics: list[dict[str, Any]] = []
    for horizon_index in range(chunk_size):
        horizon_errors = horizon_position_errors[horizon_index]
        valid_samples = len(horizon_errors)
        if not valid_samples:
            action_horizon_metrics.append(
                {
                    "horizon_index": horizon_index,
                    "valid_samples": 0,
                    "finite_rate": None,
                    "position_mae_cm": None,
                    "position_p90_cm": None,
                    "position_axis_mae_cm": None,
                    "orientation_mae_deg": None,
                    "gripper_mae": None,
                    "gripper_sign_accuracy": None,
                }
            )
            continue
        action_horizon_metrics.append(
            {
                "horizon_index": horizon_index,
                "valid_samples": valid_samples,
                "finite_rate": horizon_finite[horizon_index] / valid_samples,
                "position_mae_cm": float(np.mean(horizon_errors) * 100.0),
                "position_p90_cm": float(
                    np.percentile(horizon_errors, 90) * 100.0
                ),
                "position_axis_mae_cm": (
                    np.mean(
                        np.asarray(horizon_axis_errors[horizon_index]), axis=0
                    )
                    * 100.0
                ).tolist(),
                "orientation_mae_deg": float(
                    np.mean(horizon_orientation_errors[horizon_index])
                ) if horizon_orientation_errors[horizon_index] else None,
                "gripper_mae": float(
                    np.mean(horizon_gripper_errors[horizon_index])
                ),
                "gripper_sign_accuracy": float(
                    np.mean(horizon_gripper_correct[horizon_index])
                ),
            }
        )
    return {
        "step": checkpoint_step,
        "samples": total,
        "position_target_mode": position_target_mode,
        "position_mae_cm": float(np.mean(position_errors) * 100.0),
        "position_p90_cm": float(np.percentile(position_errors, 90) * 100.0),
        "position_axis_mae_cm": (
            np.mean(np.asarray(axis_errors), axis=0) * 100.0
        ).tolist(),
        "position_baseline_mode": (
            "zero_delta" if position_target_mode == "delta" else "current_eef"
        ),
        "position_baseline_mae_cm": baseline_mae_cm,
        "current_eef_baseline_mae_cm": (
            baseline_mae_cm if position_target_mode == "absolute" else None
        ),
        "zero_delta_baseline_mae_cm": (
            baseline_mae_cm if position_target_mode == "delta" else None
        ),
        "position_mae_cm_by_level": {
            level: float(np.mean(values) * 100.0)
            for level, values in sorted(level_position_errors.items())
        },
        "orientation_mae_deg": float(np.mean(orientation_errors)) if orientation_errors else None,
        "model_action_dimension": int(predicted_chunk.shape[-1]),
        "phase_metrics": {phase: {"samples": len(values),
            "position_mae_cm": float(np.asarray(values)[:, 0].mean()),
            "z_mae_cm": float(np.asarray(values)[:, 1].mean()),
            "gripper_sign_accuracy": float(np.asarray(values)[:, 2].mean())}
            for phase, values in phase_errors.items()},
        "gripper_mae": float(np.mean(gripper_errors)),
        "gripper_sign_accuracy": float(np.mean(gripper_correct)),
        "action_horizon_metrics": action_horizon_metrics,
        "finite_rate": finite / total,
        # Preserve the old field's per-sample meaning under batched inference.
        "inference_mean_ms": inference_total_ms / total,
        "inference_batch_mean_ms": inference_total_ms / inference_batches,
        "inference_throughput_frames_s": (
            total / (inference_total_ms / 1000.0)
            if inference_total_ms > 0.0
            else 0.0
        ),
        "batch_size_effective": batch_size,
        "scan_wall_time_s": time.perf_counter() - checkpoint_started,
    }


def main() -> None:
    args = parse_args()
    if args.progress_every < 0:
        raise ValueError("--progress-every must be zero or positive")

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    checkpoints = discover_peft_checkpoints(
        args.checkpoint_root,
        args.checkpoint_steps,
    )
    if not checkpoints:
        raise RuntimeError(f"No PEFT checkpoints found under {args.checkpoint_root}")

    checkpoint_steps = [int(checkpoint.parent.name) for checkpoint in checkpoints]
    checkpoint_order = {
        step: offset for offset, step in enumerate(checkpoint_steps)
    }
    chunk_sizes = {checkpoint_chunk_size(checkpoint) for checkpoint in checkpoints}
    if len(chunk_sizes) != 1:
        raise ValueError(
            f"All scanned checkpoints must share one chunk_size; got {sorted(chunk_sizes)}"
        )
    action_chunk_size = chunk_sizes.pop()
    dataset_info = json.loads(
        (args.dataset_root / "meta" / "info.json").read_text(encoding="utf-8")
    )
    dataset_fps = float(dataset_info["fps"])
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps={
            "action": [
                horizon_index / dataset_fps
                for horizon_index in range(action_chunk_size)
            ]
        },
    )
    rng = np.random.default_rng(args.seed)
    sample_count = len(dataset) if args.samples <= 0 else min(args.samples, len(dataset))
    if sample_count <= 0:
        raise RuntimeError(f"No validation frames found under {args.dataset_root}")
    indices = (
        np.arange(len(dataset), dtype=np.int64)
        if sample_count == len(dataset)
        else np.sort(rng.choice(len(dataset), size=sample_count, replace=False))
    )
    selected_indices = [int(index) for index in indices]

    # Cache small fixed scans for checkpoint-to-checkpoint speed. Full-split
    # gates decode lazily so 1,000+ float images do not occupy over 1 GB RAM.
    cached_samples = (
        {index: dataset[index] for index in selected_indices}
        if sample_count <= 256
        else None
    )
    manifest_path = args.dataset_root / "balanced_split_manifest.json"
    episode_levels: dict[int, str] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episode_levels = {
            episode_index: str(record["level"])
            for episode_index, record in enumerate(manifest.get("records", []))
        }

    scan_config = make_scan_config(
        args,
        checkpoint_steps,
        selected_indices,
        len(dataset),
    )
    scan_config["action_chunk_size"] = action_chunk_size
    scan_config["dataset_fps"] = dataset_fps
    results: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        try:
            saved_report = json.loads(args.output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot resume from invalid report {args.output}: {error}"
            ) from error
        results = validate_resume_report(saved_report, scan_config)
        print(
            "SCAN_RESUME="
            + json.dumps(
                {
                    "output": str(args.output),
                    "completed_steps": [int(item["step"]) for item in results],
                    "pending_steps": [
                        step
                        for step in checkpoint_steps
                        if step not in {int(item["step"]) for item in results}
                    ],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elif args.resume:
        print(
            "SCAN_RESUME="
            + json.dumps(
                {
                    "output": str(args.output),
                    "completed_steps": [],
                    "pending_steps": checkpoint_steps,
                    "note": "No prior report found; starting a fresh scan.",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    completed_steps = {int(item["step"]) for item in results}
    for checkpoint in checkpoints:
        step = int(checkpoint.parent.name)
        if step in completed_steps:
            print(
                "CHECKPOINT_SKIPPED="
                + json.dumps({"step": step, "reason": "resume_complete"}),
                flush=True,
            )
            continue

        checkpoint_started = time.perf_counter()
        policy = preprocess = postprocess = None
        try:
            policy, preprocess, postprocess = load_policy(checkpoint, args.device)
            try:
                result = evaluate_checkpoint(
                    torch=torch,
                    checkpoint_step=step,
                    policy=policy,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    dataset=dataset,
                    selected_indices=selected_indices,
                    cached_samples=cached_samples,
                    episode_levels=episode_levels,
                    seed=args.seed,
                    batch_size=args.batch_size,
                    progress_every=args.progress_every,
                    checkpoint_started=checkpoint_started,
                    position_target_mode=args.position_target_mode,
                )
            except RuntimeError as error:
                if args.batch_size == 2 and is_cuda_oom(error):
                    # The exception traceback owns the failed batch tensors.
                    # Break that reference before emptying the CUDA allocator.
                    error.__traceback__ = None
                    print(
                        "CHECKPOINT_BATCH_FALLBACK="
                        + json.dumps(
                            {
                                "step": step,
                                "requested_batch_size": 2,
                                "effective_batch_size": 1,
                                "reason": "cuda_out_of_memory",
                            }
                        ),
                        flush=True,
                    )
                    reset_policy = getattr(policy, "reset", None)
                    if callable(reset_policy):
                        reset_policy()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        result = evaluate_checkpoint(
                            torch=torch,
                            checkpoint_step=step,
                            policy=policy,
                            preprocess=preprocess,
                            postprocess=postprocess,
                            dataset=dataset,
                            selected_indices=selected_indices,
                            cached_samples=cached_samples,
                            episode_levels=episode_levels,
                            seed=args.seed,
                            batch_size=1,
                            progress_every=args.progress_every,
                            checkpoint_started=checkpoint_started,
                            position_target_mode=args.position_target_mode,
                        )
                    except RuntimeError as retry_error:
                        if is_cuda_oom(retry_error):
                            raise RuntimeError(
                                f"CUDA out of memory while scanning checkpoint {step} "
                                "even at --batch-size 1. Close other GPU applications "
                                "or reduce the validation sample count."
                            ) from retry_error
                        raise
                elif is_cuda_oom(error):
                    raise RuntimeError(
                        f"CUDA out of memory while scanning checkpoint {step} at "
                        "--batch-size 1. Close other GPU applications or reduce the "
                        "validation sample count."
                    ) from error
                else:
                    raise

            results.append(result)
            results.sort(key=lambda item: checkpoint_order[int(item["step"])])
            completed_steps.add(step)
            report = build_report(args, scan_config, results)
            atomic_write_json(args.output, report)
            print(
                "CHECKPOINT_RESULT=" + json.dumps(result, ensure_ascii=False),
                flush=True,
            )
        finally:
            policy = preprocess = postprocess = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = build_report(args, scan_config, results)
    atomic_write_json(args.output, report)
    print("VLA_CHECKPOINT_SCAN=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
