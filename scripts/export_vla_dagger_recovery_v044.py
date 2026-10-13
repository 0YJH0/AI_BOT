"""Export compact pre-grasp DAgger recovery episodes to LeRobot 0.4.4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.dataset import (  # noqa: E402
    DELTA_ACTION_NAMES,
    VISUAL_FSM_CONDITIONED_STATE_NAMES,
)
from isaac_lab_data_engine.vla.recovery import (  # noqa: E402
    RECOVERY_FORMAT_VERSION,
    validate_recovery_episode,
)


LEVELS = ("easy", "medium", "hard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation"), default=["train", "validation"]
    )
    return parser.parse_args(argv)


def discover_records(source_root: Path, split: str) -> list[tuple[str, Path, Path]]:
    records: list[tuple[str, Path, Path]] = []
    for level in LEVELS:
        for npz_path in sorted((source_root / split / level).glob("episode_*.npz")):
            json_path = npz_path.with_suffix(".json")
            if not json_path.is_file():
                raise FileNotFoundError(json_path)
            records.append((level, npz_path, json_path))
    if not records:
        raise FileNotFoundError(f"No recovery episodes found for split {split!r}")
    return records


def _load_record(
    level: str, npz_path: Path, json_path: Path, split: str
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if int(metadata.get("format_version", -1)) != RECOVERY_FORMAT_VERSION:
        raise ValueError(f"Unsupported recovery format in {json_path}")
    if metadata.get("split") != split or metadata.get("level") != level:
        raise ValueError(f"Path/metadata split or level mismatch in {json_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    frames = validate_recovery_episode(arrays)
    if int(metadata.get("frames", -1)) != frames:
        raise ValueError(f"Frame-count mismatch in {npz_path}")
    return arrays, metadata


def export_split(args: argparse.Namespace, split: str) -> dict[str, object]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = args.output_root / split
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    source_records = discover_records(args.source_root, split)
    first_arrays, _ = _load_record(*source_records[0], split)
    height, width, channels = first_arrays["rgb"].shape[1:]
    features = {
        "observation.images.table": {
            "dtype": "image",
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(VISUAL_FSM_CONDITIONED_STATE_NAMES),),
            "names": list(VISUAL_FSM_CONDITIONED_STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(DELTA_ACTION_NAMES),),
            "names": list(DELTA_ACTION_NAMES),
        },
        "training.controller_phase": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["controller_phase"],
        },
        "training.difficulty_id": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["difficulty_id"],
        },
        "training.loss_weight": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["loss_weight"],
        },
    }
    repo_id = f"{args.repo_prefix}-{split}"
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        root=output,
        robot_type="franka_isaac_lab_absolute_ik",
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )
    manifest_records: list[dict[str, object]] = []
    try:
        for level, npz_path, json_path in source_records:
            arrays, metadata = _load_record(level, npz_path, json_path, split)
            if tuple(arrays["rgb"].shape[1:]) != (height, width, channels):
                raise ValueError(f"Inconsistent RGB shape in {npz_path}")
            frames = int(arrays["rgb"].shape[0])
            learner_indices = np.flatnonzero(
                arrays["learner_visited"].reshape(-1).astype(bool)
            ).tolist()
            band = metadata.get("recovery_band_m") or {
                "minimum": 0.005,
                "target": 0.015,
                "maximum": 0.025,
            }
            distances = arrays["distance_to_pregrasp_m"].reshape(-1)
            near_indices = np.flatnonzero(
                (distances >= float(band["minimum"]))
                & (distances <= float(band["maximum"]))
            ).tolist()
            for index in range(frames):
                dataset.add_frame(
                    {
                        "observation.images.table": arrays["rgb"][index],
                        "observation.state": arrays["state"][index],
                        "action": arrays["action"][index],
                        "training.controller_phase": arrays["controller_phase"][index],
                        "training.difficulty_id": arrays["difficulty_id"][index],
                        "training.loss_weight": arrays["loss_weight"][index],
                        "task": str(metadata["task"]),
                    }
                )
            dataset.save_episode()
            manifest_records.append(
                {
                    "level": level,
                    "source": str(npz_path.resolve()),
                    "frames": frames,
                    "dagger_recovery": True,
                    "near_pregrasp_target_m": 0.015,
                    "learner_visited_frames": int(metadata["learner_visited_frames"]),
                    "learner_visited_frame_indices": learner_indices,
                    "near_pregrasp_frame_indices": near_indices,
                    "recovery_band_m": band,
                    "policy_checkpoint_role": metadata.get("policy_checkpoint_role"),
                }
            )
    finally:
        dataset.finalize()

    summary = {
        "split": split,
        "repo_id": repo_id,
        "output": str(output.resolve()),
        "episodes": len(manifest_records),
        "frames": sum(int(record["frames"]) for record in manifest_records),
        "level_episode_counts": {
            level: sum(record["level"] == level for record in manifest_records)
            for level in LEVELS
        },
        "level_frame_counts": {
            level: sum(
                int(record["frames"])
                for record in manifest_records
                if record["level"] == level
            )
            for level in LEVELS
        },
        "rgb_shape": [height, width, channels],
        "state_dim": len(VISUAL_FSM_CONDITIONED_STATE_NAMES),
        "action_dim": len(DELTA_ACTION_NAMES),
        "action_names": list(DELTA_ACTION_NAMES),
        "privileged_object_state": False,
        "visual_pose_conditioning": True,
        "action_target_mode": "visual_fsm_bounded_delta",
        "controller_phase_frame_counts": {
            "0": 0,
            "1": sum(int(record["frames"]) for record in manifest_records),
            "2": 0,
            "3": 0,
            "4": 0,
        },
        "training_metadata": {
            "policy_input": False,
            "features": [
                "training.controller_phase",
                "training.difficulty_id",
                "training.loss_weight",
            ],
            "dagger_recovery": True,
            "teacher": "visual_scripted_ik_delta_xyz",
            "trainer_must_consume_loss_weight": True,
        },
        "records": manifest_records,
    }
    (output / "balanced_split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "DAGGER_RECOVERY_SPLIT="
        + json.dumps(summary | {"records": "omitted"}, ensure_ascii=False),
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    if args.fps < 1:
        raise ValueError("--fps must be positive")
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    summaries = {split: export_split(args, split) for split in args.splits}
    summary = {
        "status": "VLA_DAGGER_RECOVERY_EXPORT_OK",
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "splits": {
            split: {
                "episodes": item["episodes"],
                "frames": item["frames"],
                "level_frame_counts": item["level_frame_counts"],
            }
            for split, item in summaries.items()
        },
    }
    (args.output_root / "recovery_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DAGGER_RECOVERY_EXPORT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
