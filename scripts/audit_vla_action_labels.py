"""Audit absolute-IK normalization, gripper labels, and phase coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.dataset import (  # noqa: E402
    ACTION_NAMES,
    align_episode,
    discover_demonstrations,
)


PHASE_NAMES = {
    0: "rest",
    1: "approach_above",
    2: "approach_object",
    3: "grasp_close",
    4: "lift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--action-offset-steps", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def describe(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "p01": np.percentile(values, 1, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    episodes = discover_demonstrations(args.source, successes_only=True)
    if not episodes:
        raise RuntimeError("No successful camera demonstrations found")

    action_batches = []
    phase_counter: Counter[str] = Counter()
    phase_gripper: dict[str, list[float]] = {name: [] for name in PHASE_NAMES.values()}
    per_episode_frames = []

    for episode_dir in episodes:
        aligned = align_episode(
            episode_dir, action_offset_steps=args.action_offset_steps
        )
        with np.load(episode_dir / "trajectory.npz") as trajectory:
            controller_state = np.asarray(trajectory["controller_state"], dtype=np.int64)
        phases = controller_state[aligned.trajectory_indices]
        action_batches.append(aligned.action)
        per_episode_frames.append(aligned.frames)
        for phase_id, gripper in zip(phases.tolist(), aligned.action[:, 7].tolist(), strict=True):
            name = PHASE_NAMES.get(int(phase_id), f"unknown_{phase_id}")
            phase_counter[name] += 1
            phase_gripper.setdefault(name, []).append(float(gripper))

    actions = np.concatenate(action_batches, axis=0)
    gripper = actions[:, 7]
    open_mask = gripper > 0.5
    close_mask = gripper < -0.5
    intermediate_mask = ~(open_mask | close_mask)
    component_stats = describe(actions)

    # Mean/std is the exact affine normalization used by the exported dataset.
    normalized = (actions - actions.mean(axis=0)) / np.maximum(actions.std(axis=0), 1e-8)
    normalization_check = {
        "normalized_mean_max_abs": float(np.abs(normalized.mean(axis=0)).max()),
        "normalized_std_max_abs_error": float(np.abs(normalized.std(axis=0) - 1.0).max()),
        "near_zero_std_dimensions": [
            ACTION_NAMES[index]
            for index, value in enumerate(actions.std(axis=0))
            if value < 1e-5
        ],
    }

    phase_summary = {}
    for name, count in sorted(phase_counter.items()):
        values = np.asarray(phase_gripper[name], dtype=np.float32)
        phase_summary[name] = {
            "frames": count,
            "fraction": count / len(actions),
            "open_fraction": float(np.mean(values > 0.5)),
            "closed_fraction": float(np.mean(values < -0.5)),
        }

    report = {
        "status": "VLA_ACTION_AUDIT_OK",
        "sources": [str(path.resolve()) for path in args.source],
        "action_offset_steps": args.action_offset_steps,
        "successful_episodes": len(episodes),
        "aligned_frames": len(actions),
        "frames_per_episode": {
            "min": int(min(per_episode_frames)),
            "max": int(max(per_episode_frames)),
            "mean": float(np.mean(per_episode_frames)),
        },
        "action_names": list(ACTION_NAMES),
        "action_statistics": component_stats,
        "normalization_check": normalization_check,
        "gripper_labels": {
            "open_count": int(open_mask.sum()),
            "closed_count": int(close_mask.sum()),
            "intermediate_count": int(intermediate_mask.sum()),
            "open_fraction": float(open_mask.mean()),
            "closed_fraction": float(close_mask.mean()),
            "class_imbalance_open_to_closed": float(open_mask.sum() / max(close_mask.sum(), 1)),
        },
        "phase_coverage": phase_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VLA_ACTION_AUDIT=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
