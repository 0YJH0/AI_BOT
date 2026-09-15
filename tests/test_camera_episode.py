"""Simulator-free RGB-D episode serialization tests."""

import json

import numpy as np

from isaac_lab_data_engine.data_collection import (
    CameraFrameBuffer,
    EpisodeBuffer,
    EpisodeWriter,
    validate_episode,
)


def test_camera_episode_round_trip(tmp_path):
    trajectory = EpisodeBuffer()
    camera = CameraFrameBuffer()
    for step in range(1, 4):
        trajectory.append(
            q=np.zeros(9),
            dq=np.zeros(9),
            eef_pose=np.array([0, 0, 0, 1, 0, 0, 0]),
            object_pose=np.array([0.5, 0, 0.5, 1, 0, 0, 0]),
            action=np.array([0, 0, 0, 1, 0, 0, 0, 1]),
            reward=1.0,
            timestamp=step * 0.02,
            controller_state=step,
        )
    for step in (1, 3):
        camera.append(
            frame_step=step,
            timestamp=step * 0.02,
            rgb=np.full((8, 10, 3), 64 + step, dtype=np.uint8),
            depth=np.full((8, 10, 1), 1.5, dtype=np.float32),
            camera_intrinsics=np.array([[10, 0, 5], [0, 10, 4], [0, 0, 1]]),
            camera_pose_w=np.array([1, 0, 1, 1, 0, 0, 0]),
            object_pose_w=np.array([0.5, 0, 0.5, 1, 0, 0, 0]),
            eef_pose_w=np.array([0.5, 0, 0.7, 1, 0, 0, 0]),
        )
    episode = EpisodeWriter(tmp_path).write(
        1,
        trajectory,
        {
            "success": True,
            "failure_reason": "none",
            "steps": 3,
            "completion_time_s": 0.06,
            "total_reward": 3.0,
        },
        camera_buffer=camera,
    )
    validation = validate_episode(episode)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))

    assert validation["camera_data"] is True
    assert validation["camera_tensor_shapes"]["rgb"] == [2, 8, 10, 3]
    assert validation["finite_depth_fraction"] == 1.0
    assert metadata["camera_frames"] == 2
    assert (episode / "camera_rgb_preview.png").is_file()
    with np.load(episode / "camera.npz") as data:
        assert data["rgb"].dtype == np.uint8
        assert data["depth"].dtype == np.float32
        assert data["object_pose_camera"].shape == (2, 7)
