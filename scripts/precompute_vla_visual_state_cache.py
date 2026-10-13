"""Precompute RGB-D visual target state for the balanced causal VLA export.

This script reads only raw RGB-D, calibration, end-effector poses, and
controller phase for policy state construction.  ``trajectory.object_pose`` is
loaded after each cache is constructed and is used exclusively for the JSON
accuracy audit; it is never written to a cache or passed to the estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vision import RGBDPoseEstimator  # noqa: E402
from isaac_lab_data_engine.vla.dataset import align_episode  # noqa: E402
from isaac_lab_data_engine.vla.visual_cache import (  # noqa: E402
    apply_pregrasp_ema,
    camera_predictions_to_visual_position_b,
    derive_robot_root_poses_w,
    freeze_reference_after_grasp,
    workspace_pixel_mask,
)


DEFAULT_EXPORT_ROOT = (
    PROJECT_ROOT / "exports/lerobot/pick_lift_vla_balanced_504_causal_v2"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results/vision/rgbd_keypoint_supplement_160x120_v3/best_model.pt"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results/vla/visual_state_cache_balanced504_causal_restfreeze_v2"
)
PHASE_NAMES = {
    0: "rest",
    1: "approach_above",
    2: "approach_object",
    3: "grasp_close",
    4: "lift",
}


@dataclass(frozen=True)
class EpisodeInputs:
    """Non-privileged tensors used to construct one episode cache."""

    source: Path
    rgb: np.ndarray
    depth: np.ndarray
    camera_intrinsics: np.ndarray
    camera_pose_w: np.ndarray
    root_pose_w: np.ndarray
    frame_step: np.ndarray
    state_trajectory_index: np.ndarray
    action_trajectory_index: np.ndarray
    controller_state: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.rgb.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    parser.add_argument("--action-offset-steps", type=int, default=1)
    parser.add_argument("--ema-alpha", type=float, default=0.45)
    parser.add_argument(
        "--freeze-state-id",
        type=int,
        default=1,
        help="Latch visual xyz once approach_above starts (controller state 1).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already complete cache files; never overwrite episode caches.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output must be inspected first: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Create one cache atomically and refuse to replace an existing cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Stale partial output must be inspected first: {temporary}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_episode_inputs(
    source: Path, *, action_offset_steps: int, expected_frames: int
) -> EpisodeInputs:
    """Load and causally align tensors without exposing object pose to inference."""

    aligned = align_episode(source, action_offset_steps=action_offset_steps)
    if aligned.frames != expected_frames:
        raise ValueError(
            f"{source}: manifest declares {expected_frames} frames, causal alignment "
            f"produced {aligned.frames}"
        )
    action_indices = aligned.trajectory_indices
    state_indices = action_indices - action_offset_steps

    with np.load(source / "trajectory.npz", allow_pickle=False) as trajectory:
        eef_pose_b = np.asarray(trajectory["eef_pose"], dtype=np.float32)[state_indices]
        controller_state = np.asarray(
            trajectory["controller_state"], dtype=np.int64
        )[state_indices]
    with np.load(source / "camera.npz", allow_pickle=False) as camera:
        all_frame_steps = np.asarray(camera["frame_step"], dtype=np.int64)
        camera_state_indices = all_frame_steps - 1
        camera_action_indices = camera_state_indices + action_offset_steps
        keep = camera_action_indices < (
            int(action_indices[-1]) + 1 if len(action_indices) else 0
        )
        # The central aligner has already applied the authoritative trajectory
        # length check.  Matching the exact indices catches any silent drift.
        if not np.array_equal(camera_action_indices[keep], action_indices):
            raise ValueError(f"{source}: camera/trajectory causal indices disagree")
        rgb = np.asarray(camera["rgb"], dtype=np.uint8)[keep]
        depth = np.asarray(camera["depth"], dtype=np.float32)[keep]
        intrinsics = np.asarray(camera["camera_intrinsics"], dtype=np.float32)[keep]
        camera_pose_w = np.asarray(camera["camera_pose_w"], dtype=np.float32)[keep]
        eef_pose_w = np.asarray(camera["eef_pose_w"], dtype=np.float32)[keep]
        frame_step = all_frame_steps[keep]

    if not np.array_equal(rgb, aligned.rgb):
        raise ValueError(f"{source}: aligned RGB frames differ from export alignment")
    root_pose_w = derive_robot_root_poses_w(eef_pose_w, eef_pose_b)
    return EpisodeInputs(
        source=source,
        rgb=rgb,
        depth=depth,
        camera_intrinsics=intrinsics,
        camera_pose_w=camera_pose_w,
        root_pose_w=root_pose_w,
        frame_step=frame_step,
        state_trajectory_index=state_indices,
        action_trajectory_index=action_indices,
        controller_state=controller_state,
    )


@torch.inference_mode()
def _infer_episode(
    estimator: RGBDPoseEstimator,
    inputs: EpisodeInputs,
    *,
    device: torch.device,
    batch_size: int,
    ema_alpha: float,
    freeze_state_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_positions: list[np.ndarray] = []
    raw_validity: list[np.ndarray] = []
    for start in range(0, inputs.frames, batch_size):
        stop = min(start + batch_size, inputs.frames)
        rgb = torch.from_numpy(inputs.rgb[start:stop]).to(device)
        depth = torch.from_numpy(inputs.depth[start:stop]).to(device)
        intrinsics = torch.from_numpy(inputs.camera_intrinsics[start:stop]).to(device)
        camera_pose_w = torch.from_numpy(inputs.camera_pose_w[start:stop]).to(device)
        root_pose_w = torch.from_numpy(inputs.root_pose_w[start:stop]).to(device)
        mask = workspace_pixel_mask(
            depth, intrinsics, camera_pose_w, root_pose_w
        )
        empty_mask = ~mask.flatten(1).any(dim=-1)
        if bool(empty_mask.any()):
            indices = (torch.nonzero(empty_mask, as_tuple=False).squeeze(-1) + start)
            raise ValueError(
                f"{inputs.source}: workspace mask is empty for frames "
                f"{indices.detach().cpu().tolist()}"
            )
        position_camera = estimator.predict(
            rgb, depth, intrinsics, valid_pixel_mask=mask
        )
        position_b, valid = camera_predictions_to_visual_position_b(
            position_camera, camera_pose_w, root_pose_w
        )
        raw_positions.append(position_b.detach().cpu().numpy().astype(np.float32))
        raw_validity.append(valid.detach().cpu().numpy())

    raw_position_b = np.concatenate(raw_positions, axis=0)
    raw_valid = np.concatenate(raw_validity, axis=0)
    return apply_pregrasp_ema(
        raw_position_b,
        raw_valid,
        inputs.controller_state,
        alpha=ema_alpha,
        freeze_state_id=freeze_state_id,
    )


def _load_existing_cache(
    cache_path: Path, inputs: EpisodeInputs
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"source", "visual_position_b", "validity"}
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"{cache_path}: missing keys {sorted(missing)}")
        source = str(np.asarray(cache["source"]).item())
        visual_position_b = np.asarray(cache["visual_position_b"], dtype=np.float32)
        validity = np.asarray(cache["validity"], dtype=np.float32)
    if Path(source).resolve() != inputs.source.resolve():
        raise ValueError(f"{cache_path}: source mismatch: {source}")
    if visual_position_b.shape != (inputs.frames, 3):
        raise ValueError(f"{cache_path}: invalid position shape {visual_position_b.shape}")
    if validity.shape != (inputs.frames, 1):
        raise ValueError(f"{cache_path}: invalid validity shape {validity.shape}")
    return visual_position_b, validity


def _load_audit_truth(
    inputs: EpisodeInputs, *, freeze_state_id: int
) -> np.ndarray:
    """Load a metrics-only object target with online latch semantics."""

    with np.load(inputs.source / "trajectory.npz", allow_pickle=False) as trajectory:
        position_b = np.asarray(trajectory["object_pose"], dtype=np.float32)[
            inputs.state_trajectory_index, :3
        ]
    return freeze_reference_after_grasp(
        position_b, inputs.controller_state, freeze_state_id=freeze_state_id
    )


def _error_metrics(
    errors_m: np.ndarray, axis_abs_error_m: np.ndarray, validity: np.ndarray
) -> dict[str, Any]:
    errors_m = np.asarray(errors_m, dtype=np.float64).reshape(-1)
    axis_abs_error_m = np.asarray(axis_abs_error_m, dtype=np.float64).reshape(-1, 3)
    validity = np.asarray(validity, dtype=bool).reshape(-1)
    if not (len(errors_m) == len(axis_abs_error_m) == len(validity)):
        raise ValueError("Audit arrays have inconsistent lengths")

    def describe(values: np.ndarray) -> dict[str, float | None]:
        if len(values) == 0:
            return {
                "mean_cm": None,
                "median_cm": None,
                "p90_cm": None,
                "p95_cm": None,
                "max_cm": None,
            }
        return {
            "mean_cm": float(values.mean() * 100.0),
            "median_cm": float(np.median(values) * 100.0),
            "p90_cm": float(np.percentile(values, 90) * 100.0),
            "p95_cm": float(np.percentile(values, 95) * 100.0),
            "max_cm": float(values.max() * 100.0),
        }

    return {
        "frames": int(len(errors_m)),
        "valid_frames": int(validity.sum()),
        "valid_fraction": float(validity.mean()) if len(validity) else 0.0,
        "all_frames": describe(errors_m),
        "valid_frames_only": describe(errors_m[validity]),
        "mean_axis_absolute_error_cm": (
            axis_abs_error_m.mean(axis=0) * 100.0
        ).tolist()
        if len(axis_abs_error_m)
        else [None, None, None],
    }


def _summarize_groups(
    groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, batches in sorted(groups.items()):
        errors = np.concatenate([batch[0] for batch in batches])
        axis_errors = np.concatenate([batch[1] for batch in batches])
        validity = np.concatenate([batch[2] for batch in batches])
        result[name] = _error_metrics(errors, axis_errors, validity)
    return result


def main() -> None:
    args = parse_args()
    export_root = args.export_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.action_offset_steps != 1:
        raise ValueError("This visual cache requires causal action_offset_steps=1")
    if not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be in (0, 1]")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.freeze_state_id < 1:
        raise ValueError("--freeze-state-id must be at least 1")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    export_summary = _load_json(export_root / "balanced_export_summary.json")
    if int(export_summary.get("action_offset_steps", -1)) != args.action_offset_steps:
        raise ValueError("Balanced export is not the required causal offset=1 export")
    if output_root.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite {output_root}; choose a new path or pass --resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    estimator = RGBDPoseEstimator.load(checkpoint, device)

    overall_groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    level_groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    phase_groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    split_summaries: dict[str, Any] = {}
    total_episodes = 0
    total_frames = 0

    for split in args.splits:
        split_manifest_path = export_root / split / "balanced_split_manifest.json"
        manifest = _load_json(split_manifest_path)
        if int(manifest.get("action_offset_steps", -1)) != args.action_offset_steps:
            raise ValueError(f"{split_manifest_path} is not causally aligned at offset=1")
        records = manifest.get("records", [])
        if len(records) != int(manifest.get("episodes", -1)):
            raise ValueError(f"{split_manifest_path}: episode count mismatch")

        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_records = []
        split_groups: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = (
            defaultdict(list)
        )
        for episode_index, record in enumerate(records):
            source = Path(record["source"]).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(source)
            inputs = _load_episode_inputs(
                source,
                action_offset_steps=args.action_offset_steps,
                expected_frames=int(record["frames"]),
            )
            cache_path = split_dir / f"episode_{episode_index:06d}.npz"
            if cache_path.exists():
                if not args.resume:
                    raise FileExistsError(cache_path)
                visual_position_b, validity = _load_existing_cache(cache_path, inputs)
                status = "reused"
            else:
                visual_position_b, validity = _infer_episode(
                    estimator,
                    inputs,
                    device=device,
                    batch_size=args.batch_size,
                    ema_alpha=args.ema_alpha,
                    freeze_state_id=args.freeze_state_id,
                )
                _atomic_cache(
                    cache_path,
                    {
                        "source": np.asarray(str(source), dtype=np.str_),
                        "visual_position_b": visual_position_b.astype(np.float32),
                        "validity": validity.astype(np.float32),
                        "frame_step": inputs.frame_step.astype(np.int64),
                        "state_trajectory_index": inputs.state_trajectory_index.astype(
                            np.int64
                        ),
                        "action_trajectory_index": inputs.action_trajectory_index.astype(
                            np.int64
                        ),
                        "controller_state": inputs.controller_state.astype(np.int64),
                    },
                )
                status = "created"

            # Metrics-only privileged read: this tensor has no route back into
            # visual_position_b or the cache payload above.
            true_position_b = _load_audit_truth(
                inputs, freeze_state_id=args.freeze_state_id
            )
            axis_error = np.abs(visual_position_b - true_position_b)
            error = np.linalg.norm(axis_error, axis=-1)
            valid_flat = validity.reshape(-1).astype(bool)
            batch = (error, axis_error, valid_flat)
            split_groups[split].append(batch)
            overall_groups["overall"].append(batch)
            level_groups[str(record["level"])].append(batch)
            for phase_id in np.unique(inputs.controller_state):
                phase_mask = inputs.controller_state == phase_id
                phase_name = PHASE_NAMES.get(int(phase_id), f"unknown_{int(phase_id)}")
                phase_groups[phase_name].append(
                    (error[phase_mask], axis_error[phase_mask], valid_flat[phase_mask])
                )

            episode_metrics = _error_metrics(error, axis_error, valid_flat)
            output_records.append(
                {
                    "episode_index": episode_index,
                    "level": str(record["level"]),
                    "source": str(source),
                    "cache": str(cache_path.resolve()),
                    "frames": inputs.frames,
                    "status": status,
                    "audit_mean_error_cm": episode_metrics["all_frames"]["mean_cm"],
                    "audit_valid_fraction": episode_metrics["valid_fraction"],
                }
            )
            total_episodes += 1
            total_frames += inputs.frames
            if (episode_index + 1) % 10 == 0 or episode_index + 1 == len(records):
                print(
                    f"VISUAL_CACHE_PROGRESS={split} {episode_index + 1}/{len(records)}",
                    flush=True,
                )

        split_audit = _summarize_groups(split_groups)[split]
        split_summary = {
            "format_version": "1.0.0",
            "split": split,
            "source_manifest": str(split_manifest_path.resolve()),
            "action_offset_steps": args.action_offset_steps,
            "episodes": len(output_records),
            "frames": sum(int(item["frames"]) for item in output_records),
            "cache_keys": {
                "source": "scalar unicode source episode path",
                "visual_position_b": "float32 [F,3], robot-root frame",
                "validity": "float32 [F,1]",
            },
            "ground_truth_in_cache": False,
            "audit": split_audit,
            "records": output_records,
        }
        _atomic_json(split_dir / "visual_state_cache_manifest.json", split_summary)
        split_summaries[split] = split_summary | {"records": "omitted"}

    report = {
        "status": "VLA_VISUAL_STATE_CACHE_OK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "export_root": str(export_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "output_root": str(output_root),
        "device": str(device),
        "batch_size": args.batch_size,
        "action_offset_steps": args.action_offset_steps,
        "ema_alpha": args.ema_alpha,
        "freeze_state_id": args.freeze_state_id,
        "freeze_at_controller_state": "approach_above (state >= 1)",
        "episodes": total_episodes,
        "frames": total_frames,
        "cache_contains_object_ground_truth": False,
        "privileged_object_pose_usage": (
            "metrics_only; loaded after cache construction, never passed to inference "
            "or filtering, and never saved in episode caches"
        ),
        "audit_reference": (
            "trajectory object_pose during unobstructed rest; frozen at the last "
            "controller_state < 1 to match the cached target anchor"
        ),
        "overall_audit": _summarize_groups(overall_groups)["overall"],
        "audit_by_level": _summarize_groups(level_groups),
        "audit_by_phase": _summarize_groups(phase_groups),
        "splits": split_summaries,
    }
    _atomic_json(output_root / "visual_state_cache_audit.json", report)
    print("VLA_VISUAL_STATE_CACHE=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
