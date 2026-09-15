"""Fast, simulator-free tests for the Phase 3 episode format."""

from __future__ import annotations

import json

import numpy as np

from isaac_lab_data_engine.data_collection import EpisodeBuffer, EpisodeWriter, validate_episode


def test_episode_round_trip(tmp_path):
    buffer = EpisodeBuffer()
    for step in range(3):
        buffer.append(
            q=np.full(9, step, dtype=np.float32),
            dq=np.zeros(9, dtype=np.float32),
            eef_pose=np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            object_pose=np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            action=np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
            reward=float(step),
            timestamp=(step + 1) * 0.02,
            controller_state=step,
        )

    writer = EpisodeWriter(tmp_path)
    episode_dir = writer.write(
        1,
        buffer,
        {
            "success": True,
            "failure_reason": "none",
            "steps": 3,
            "completion_time_s": 0.06,
            "total_reward": 3.0,
        },
    )
    validation = validate_episode(episode_dir)
    manifest = json.loads((tmp_path / "dataset_metadata.json").read_text(encoding="utf-8"))

    assert validation["status"] == "EPISODE_VALIDATION_OK"
    assert validation["tensor_shapes"]["q"] == [3, 9]
    assert validation["tensor_shapes"]["action"] == [3, 8]
    assert manifest["num_episodes"] == 1
    assert manifest["num_success"] == 1
    assert manifest["num_failure"] == 0
    assert manifest["episodes"][0]["domain_randomization"] is False


def test_randomized_episode_requires_and_indexes_sidecar(tmp_path):
    buffer = EpisodeBuffer()
    buffer.append(
        q=np.zeros(9),
        dq=np.zeros(9),
        eef_pose=np.array([0, 0, 0, 1, 0, 0, 0]),
        object_pose=np.array([1, 0, 0, 1, 0, 0, 0]),
        action=np.array([0, 0, 0, 1, 0, 0, 0, 1]),
        reward=0.0,
        timestamp=0.02,
        controller_state=0,
    )
    writer = EpisodeWriter(tmp_path)
    episode_dir = writer.write(
        1,
        buffer,
        {
            "success": False,
            "failure_reason": "test",
            "steps": 1,
            "completion_time_s": 0.02,
            "total_reward": 0.0,
            "domain_randomization": True,
            "domain_params_file": "domain_params.json",
        },
        sidecar_json={
            "domain_params.json": {
                "object": {},
                "camera": {},
                "lighting": {},
                "applied_values": {},
            }
        },
    )

    validation = validate_episode(episode_dir)
    manifest = json.loads((tmp_path / "dataset_metadata.json").read_text(encoding="utf-8"))
    assert validation["domain_randomization"] is True
    assert manifest["episodes"][0]["domain_params_file"] == "domain_params.json"
