"""Test Phase 10 RGB-D trajectory visualization."""

import numpy as np

from isaac_lab_data_engine.sensors import plot_camera_object_eef_trajectory


def test_trajectory_plot_is_created(tmp_path):
    camera_file = tmp_path / "camera.npz"
    positions = np.array([[0, 0, 0], [0.1, 0, 0.1]], dtype=np.float32)
    pose = np.concatenate((positions, np.tile([[1, 0, 0, 0]], (2, 1))), axis=1)
    np.savez_compressed(
        camera_file,
        camera_pose_w=pose + np.array([1, -2, 1.5, 0, 0, 0, 0]),
        object_pose_w=pose,
        eef_pose_w=pose + np.array([0, 0, 0.1, 0, 0, 0, 0]),
    )
    output = plot_camera_object_eef_trajectory(camera_file, tmp_path / "trajectory.png")
    assert output.stat().st_size > 1000
