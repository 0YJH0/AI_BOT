"""Export Isaac demonstrations as image-backed LeRobot v3 for Windows.

LeRobot 0.4.4's MP4 reader depends on torchvision.io.VideoReader, which is no
longer available in torchvision 0.26.  Storing frames as the native ``image``
feature avoids that decoder while preserving the exact SmolVLA input contract.
"""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="Pick up the red cube and lift it 10 centimeters.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    episode_dirs = discover_demonstrations(args.source, successes_only=True, minimum_camera_frames=2)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]
    if not episode_dirs:
        raise RuntimeError("No complete successful RGB demonstrations were found")
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")

    aligned = [align_episode(path, task=args.task) for path in episode_dirs]
    rgb_shapes = sorted({tuple(item.rgb.shape[1:]) for item in aligned})
    if len(rgb_shapes) != 1:
        raise ValueError(f"All RGB frames must have one resolution, found {rgb_shapes}")
    height, width, channels = rgb_shapes[0]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.table": {
            "dtype": "image",
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
        use_videos=False,
        image_writer_threads=2,
    )
    try:
        for episode in aligned:
            for index in range(episode.frames):
                dataset.add_frame(
                    {
                        "observation.images.table": episode.rgb[index],
                        "observation.state": episode.state[index],
                        "action": episode.action[index],
                        "task": episode.task,
                    }
                )
            dataset.save_episode()
    finally:
        dataset.finalize()

    summary = {
        "status": "exported",
        "storage": "image-backed LeRobot v3 (Windows torchvision 0.26 compatible)",
        "repo_id": args.repo_id,
        "output_root": str(args.output_root.resolve()),
        "num_episodes": len(aligned),
        "num_frames": sum(item.frames for item in aligned),
        "rgb_shape": [height, width, channels],
        "state_dim": len(STATE_NAMES),
        "action_dim": len(ACTION_NAMES),
        "privileged_object_state": False,
    }
    (args.output_root / "isaac_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("VLA_IMAGE_DATASET_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
