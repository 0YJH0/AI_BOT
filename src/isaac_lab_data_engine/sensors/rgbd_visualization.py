"""Visual checks for synchronized RGB-D episode labels."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def plot_camera_object_eef_trajectory(
    camera_npz: str | Path,
    output_path: str | Path,
) -> Path:
    """Plot camera, object, and end-effector positions in the world frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    camera_npz = Path(camera_npz).resolve()
    output_path = Path(output_path).resolve()
    with np.load(camera_npz) as data:
        camera = data["camera_pose_w"][:, :3]
        target_object = data["object_pose_w"][:, :3]
        end_effector = data["eef_pose_w"][:, :3]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(7.2, 5.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*target_object.T, color="#D95F59", linewidth=2.5, label="Object")
    axis.plot(*end_effector.T, color="#4C78A8", linewidth=2.0, label="End-effector")
    axis.scatter(*camera[0], color="#222222", marker="^", s=70, label="Camera")
    axis.scatter(*target_object[0], color="#D95F59", marker="o", s=35)
    axis.scatter(*target_object[-1], color="#D95F59", marker="x", s=55)
    axis.set_title("Camera, object, and end-effector trajectory", loc="left", weight="bold")
    axis.set_xlabel("World X (m)")
    axis.set_ylabel("World Y (m)")
    axis.set_zlabel("World Z (m)")
    axis.legend(frameon=False, loc="upper left")
    all_positions = np.concatenate((camera, target_object, end_effector), axis=0)
    spans = np.maximum(np.ptp(all_positions, axis=0), 0.2)
    axis.set_box_aspect(spans)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    os.replace(temporary, output_path)
    return output_path
