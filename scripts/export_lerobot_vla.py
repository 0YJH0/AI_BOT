"""Export successful Isaac demonstrations to a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.dataset import (  # noqa: E402
    ACTION_NAMES,
    STATE_NAMES,
    align_episode,
    discover_demonstrations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        default=None,
        help="Dataset root; repeat for Easy/Medium/Hard roots.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "exports/lerobot/pick_lift_vla",
    )
    parser.add_argument("--repo-id", default="local/isaac-pick-lift-vla")
    parser.add_argument("--task", default="Pick up the red cube and lift it 10 centimeters.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without requiring LeRobot or writing videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = args.source or [PROJECT_ROOT / "dataset/phase12_rgbd_hard_raw"]
    episode_dirs = discover_demonstrations(
        sources,
        successes_only=not args.include_failures,
        minimum_camera_frames=2,
    )
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise RuntimeError("No camera demonstrations matched the requested filters")

    aligned = [align_episode(path, task=args.task) for path in episode_dirs]
    shapes = sorted({tuple(item.rgb.shape[1:]) for item in aligned})
    if len(shapes) != 1:
        raise ValueError(f"LeRobot video feature requires one RGB resolution, found {shapes}")
    height, width, channels = shapes[0]
    summary = {
        "status": "validated" if args.dry_run else "exported",
        "sources": [str(path.resolve()) for path in sources],
        "output_root": str(args.output_root.resolve()),
        "repo_id": args.repo_id,
        "task": args.task,
        "fps": args.fps,
        "num_episodes": len(aligned),
        "num_frames": sum(item.frames for item in aligned),
        "rgb_shape": [height, width, channels],
        "state_dim": len(STATE_NAMES),
        "action_dim": len(ACTION_NAMES),
        "privileged_object_state": False,
    }
    if args.dry_run:
        print("VLA_DATASET_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
        return
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output_root}. Choose a new --output-root."
        )

    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot is not installed. Run this script inside the separate lerobot-vla environment."
        ) from error

    features = {
        "observation.images.table": {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": list(ACTION_NAMES),
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.output_root,
        robot_type="franka_isaac_lab_absolute_ik",
        features=features,
        use_videos=True,
        image_writer_threads=2,
    )
    try:
        for episode in aligned:
            for frame_index in range(episode.frames):
                dataset.add_frame(
                    {
                        "observation.images.table": episode.rgb[frame_index],
                        "observation.state": episode.state[frame_index],
                        "action": episode.action[frame_index],
                        "task": episode.task,
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "isaac_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("VLA_DATASET_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
