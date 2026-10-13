"""Aggregate the immutable v5 train split with train-only DAgger recovery data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LEVELS = ("easy", "medium", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    return parser.parse_args()


def _read_train_manifest(root: Path) -> dict[str, object]:
    path = root / "balanced_split_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "train":
        raise ValueError(f"Expected a train manifest at {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest has no records: {path}")
    if int(payload.get("episodes", -1)) != len(records):
        raise ValueError(f"Episode count mismatch in {path}")
    if int(payload.get("frames", -1)) != sum(int(item["frames"]) for item in records):
        raise ValueError(f"Frame count mismatch in {path}")
    return payload


def main() -> None:
    args = parse_args()
    base_root = args.base_root.expanduser().resolve()
    recovery_root = args.recovery_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    base = _read_train_manifest(base_root)
    recovery = _read_train_manifest(recovery_root)
    if not all(bool(record.get("dagger_recovery")) for record in recovery["records"]):
        raise ValueError("Every recovery manifest record must be marked dagger_recovery")

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    aggregate_datasets(
        repo_ids=[str(base["repo_id"]), str(recovery["repo_id"])],
        roots=[base_root, recovery_root],
        aggr_repo_id=args.repo_id,
        aggr_root=output_root,
    )
    metadata = LeRobotDatasetMetadata(args.repo_id, root=output_root)
    records = [dict(record) for record in base["records"]] + [
        dict(record) for record in recovery["records"]
    ]
    frames = sum(int(record["frames"]) for record in records)
    if metadata.total_episodes != len(records) or metadata.total_frames != frames:
        raise RuntimeError(
            "Aggregated LeRobot metadata does not match source manifests: "
            f"episodes={metadata.total_episodes}/{len(records)}, "
            f"frames={metadata.total_frames}/{frames}"
        )
    level_episode_counts = {
        level: sum(record["level"] == level for record in records) for level in LEVELS
    }
    level_frame_counts = {
        level: sum(
            int(record["frames"]) for record in records if record["level"] == level
        )
        for level in LEVELS
    }
    base_phase = base.get("controller_phase_frame_counts") or {}
    recovery_frames = int(recovery["frames"])
    phase_counts = {
        str(phase): int(base_phase.get(str(phase), 0))
        + (recovery_frames if phase == 1 else 0)
        for phase in range(5)
    }
    merged = {
        **{
            key: value
            for key, value in base.items()
            if key not in {
                "repo_id",
                "output",
                "episodes",
                "frames",
                "level_episode_counts",
                "level_frame_counts",
                "controller_phase_frame_counts",
                "records",
            }
        },
        "split": "train",
        "repo_id": args.repo_id,
        "output": str(output_root),
        "episodes": len(records),
        "frames": frames,
        "level_episode_counts": level_episode_counts,
        "level_frame_counts": level_frame_counts,
        "controller_phase_frame_counts": phase_counts,
        "dagger_recovery": {
            "base_root": str(base_root),
            "recovery_root": str(recovery_root),
            "episode_start_index": int(base["episodes"]),
            "episodes": int(recovery["episodes"]),
            "frames": recovery_frames,
            "train_only": True,
            "validation_policy": "evaluate original v5 validation and separate held-out recovery validation",
        },
        "records": records,
    }
    manifest_path = output_root / "balanced_split_manifest.json"
    manifest_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "VLA_TRAIN_AGGREGATION_OK",
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "episodes": len(records),
        "frames": frames,
        "base_episodes": int(base["episodes"]),
        "base_frames": int(base["frames"]),
        "recovery_episodes": int(recovery["episodes"]),
        "recovery_frames": recovery_frames,
        "manifest": str(manifest_path),
    }
    print("VLA_TRAIN_AGGREGATION=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
