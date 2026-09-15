"""Plot camera/object/end-effector trajectories from a Phase 10 episode."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaac_lab_data_engine.sensors import plot_camera_object_eef_trajectory


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--episode-dir",
    type=Path,
    default=ROOT / "dataset" / "phase10_rgbd_validation" / "episode_000001",
)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "results" / "phase10" / "camera_object_eef_trajectory.png",
)
args = parser.parse_args()


if __name__ == "__main__":
    output = plot_camera_object_eef_trajectory(args.episode_dir / "camera.npz", args.output)
    print(f"PHASE10_TRAJECTORY_PLOT={output}")
