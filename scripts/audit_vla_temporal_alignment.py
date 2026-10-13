"""Quantify the post-step observation / previous-action mismatch in VLA exports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_NAMES = {
    0: "rest",
    1: "approach_above",
    2: "approach_object",
    3: "grasp_close",
    4: "lift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=PROJECT_ROOT / "exports/lerobot/pick_lift_vla_balanced_504_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/vla/temporal_alignment_audit.json",
    )
    return parser.parse_args()


def _new_bucket() -> dict[str, list[float]]:
    return defaultdict(list)


def _summarize(bucket: dict[str, list[float]]) -> dict[str, float | int]:
    if not bucket["label_shift_m"]:
        return {"frames": 0}
    shift = np.asarray(bucket["label_shift_m"], dtype=np.float64)
    old_eef = np.asarray(bucket["old_eef_error_m"], dtype=np.float64)
    causal_eef = np.asarray(bucket["causal_eef_error_m"], dtype=np.float64)
    return {
        "frames": int(len(shift)),
        "position_label_shift_mean_cm": float(shift.mean() * 100.0),
        "position_label_shift_p90_cm": float(np.percentile(shift, 90) * 100.0),
        "position_label_changed_over_1mm_fraction": float(np.mean(shift > 0.001)),
        "gripper_label_flip_fraction": float(np.mean(bucket["gripper_flip"])),
        "controller_phase_change_fraction": float(np.mean(bucket["phase_change"])),
        "current_eef_to_old_label_mean_cm": float(old_eef.mean() * 100.0),
        "current_eef_to_causal_label_mean_cm": float(causal_eef.mean() * 100.0),
    }


def main() -> None:
    args = parse_args()
    overall = _new_bucket()
    by_split: dict[str, dict[str, list[float]]] = {}
    by_level: dict[str, dict[str, list[float]]] = {}
    by_phase: dict[str, dict[str, list[float]]] = {}
    episodes = 0
    frames_before = 0
    dropped_terminal_frames = 0

    for split in ("train", "validation", "test"):
        manifest_path = args.export_root / split / "balanced_split_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_bucket = by_split.setdefault(split, _new_bucket())
        for record in manifest["records"]:
            episodes += 1
            level = str(record["level"])
            level_bucket = by_level.setdefault(level, _new_bucket())
            episode_dir = Path(record["source"])
            with np.load(episode_dir / "trajectory.npz") as trajectory:
                eef_pose = np.asarray(trajectory["eef_pose"], dtype=np.float32)
                action = np.asarray(trajectory["action"], dtype=np.float32)
                controller_state = np.asarray(
                    trajectory["controller_state"], dtype=np.int64
                )
            with np.load(episode_dir / "camera.npz") as camera:
                frame_steps = np.asarray(camera["frame_step"], dtype=np.int64)

            state_indices = frame_steps - 1
            frames_before += len(state_indices)
            valid = state_indices + 1 < len(action)
            dropped_terminal_frames += int((~valid).sum())
            state_indices = state_indices[valid]
            next_indices = state_indices + 1
            old_action = action[state_indices]
            causal_action = action[next_indices]
            current_eef = eef_pose[state_indices]
            old_phase = controller_state[state_indices]
            causal_phase = controller_state[next_indices]

            values = {
                "label_shift_m": np.linalg.norm(
                    causal_action[:, :3] - old_action[:, :3], axis=1
                ),
                "gripper_flip": (
                    (causal_action[:, 7] >= 0.0) != (old_action[:, 7] >= 0.0)
                ).astype(np.float32),
                "phase_change": (causal_phase != old_phase).astype(np.float32),
                "old_eef_error_m": np.linalg.norm(
                    old_action[:, :3] - current_eef[:, :3], axis=1
                ),
                "causal_eef_error_m": np.linalg.norm(
                    causal_action[:, :3] - current_eef[:, :3], axis=1
                ),
            }
            for name, array in values.items():
                data = array.astype(np.float64).tolist()
                overall[name].extend(data)
                split_bucket[name].extend(data)
                level_bucket[name].extend(data)

            for phase_id in np.unique(causal_phase):
                phase_name = PHASE_NAMES.get(int(phase_id), f"unknown_{int(phase_id)}")
                phase_bucket = by_phase.setdefault(phase_name, _new_bucket())
                phase_mask = causal_phase == phase_id
                for name, array in values.items():
                    phase_bucket[name].extend(
                        array[phase_mask].astype(np.float64).tolist()
                    )

    report = {
        "status": "VLA_TEMPORAL_ALIGNMENT_AUDIT_COMPLETE",
        "finding": (
            "Raw rows pair each post-step observation with the action already applied. "
            "Causal BC must pair that observation with trajectory action index +1."
        ),
        "export_root": str(args.export_root.resolve()),
        "episodes": episodes,
        "camera_frames_before": frames_before,
        "terminal_frames_dropped_for_causal_alignment": dropped_terminal_frames,
        "overall": _summarize(overall),
        "by_split": {name: _summarize(bucket) for name, bucket in by_split.items()},
        "by_level": {name: _summarize(bucket) for name, bucket in by_level.items()},
        "by_causal_phase": {
            name: _summarize(bucket) for name, bucket in by_phase.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VLA_TEMPORAL_ALIGNMENT_AUDIT=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
