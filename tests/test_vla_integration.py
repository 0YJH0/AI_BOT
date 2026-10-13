from __future__ import annotations

import json

import numpy as np
import pytest

from isaac_lab_data_engine.vla.dataset import (
    align_episode,
    build_visual_fsm_bounded_actions,
    pack_proprioceptive_state,
)
from isaac_lab_data_engine.vla.hybrid_controller import (
    HybridActionGuard,
    decode_vla_position_target,
)
from isaac_lab_data_engine.vla.rpc import VLARequest, VLAResponse, decode_arrays, encode_arrays


def test_pack_proprioception_excludes_object_state() -> None:
    state = pack_proprioceptive_state(
        np.arange(9, dtype=np.float32),
        np.arange(9, dtype=np.float32) + 10,
        np.arange(7, dtype=np.float32) + 20,
    )
    assert state.shape == (25,)
    np.testing.assert_array_equal(state[:9], np.arange(9, dtype=np.float32))
    np.testing.assert_array_equal(state[-7:], np.arange(7, dtype=np.float32) + 20)


def test_pack_proprioception_can_append_visual_target() -> None:
    state = pack_proprioceptive_state(
        np.zeros(9, dtype=np.float32),
        np.zeros(9, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        np.asarray([0.52, -0.03, 0.03], dtype=np.float32),
        True,
    )
    assert state.shape == (29,)
    np.testing.assert_allclose(state[-4:], [0.52, -0.03, 0.03, 1.0])


def test_pack_proprioception_can_append_visual_target_and_phase() -> None:
    state = pack_proprioceptive_state(
        np.zeros(9, dtype=np.float32),
        np.zeros(9, dtype=np.float32),
        np.asarray([0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.52, -0.03, 0.03], dtype=np.float32),
        True,
        controller_phase=3,
    )

    assert state.shape == (30,)
    np.testing.assert_allclose(state[-5:], [0.52, -0.03, 0.03, 1.0, 3.0])
    with pytest.raises(ValueError, match="requires visual"):
        pack_proprioceptive_state(
            np.zeros(9), np.zeros(9), np.zeros(7), controller_phase=1
        )


def test_align_episode_uses_one_based_camera_steps(tmp_path) -> None:
    episode_dir = tmp_path / "episode_000001"
    episode_dir.mkdir()
    trajectory_length = 6
    np.savez_compressed(
        episode_dir / "trajectory.npz",
        q=np.arange(trajectory_length * 9, dtype=np.float32).reshape(trajectory_length, 9),
        dq=np.zeros((trajectory_length, 9), dtype=np.float32),
        eef_pose=np.zeros((trajectory_length, 7), dtype=np.float32),
        object_pose=np.full((trajectory_length, 7), 999.0, dtype=np.float32),
        action=np.arange(trajectory_length * 8, dtype=np.float32).reshape(trajectory_length, 8),
    )
    np.savez_compressed(
        episode_dir / "camera.npz",
        rgb=np.zeros((3, 4, 5, 3), dtype=np.uint8),
        frame_step=np.asarray([1, 3, 6], dtype=np.int64),
        timestamp=np.asarray([0.02, 0.06, 0.12], dtype=np.float64),
    )
    (episode_dir / "metadata.json").write_text(
        json.dumps({"success": True, "camera_frames": 3}), encoding="utf-8"
    )

    episode = align_episode(episode_dir)

    np.testing.assert_array_equal(episode.state_trajectory_indices, [0, 2, 5])
    np.testing.assert_array_equal(episode.observation_trajectory_indices, [0, 2, 5])
    np.testing.assert_array_equal(episode.trajectory_indices, [0, 2, 5])
    np.testing.assert_array_equal(episode.action_trajectory_indices, [0, 2, 5])
    np.testing.assert_array_equal(episode.action[:, 0], [0, 16, 40])
    assert episode.state.shape == (3, 25)
    assert not np.any(episode.state == 999.0)


def test_align_episode_supports_causal_next_action(tmp_path) -> None:
    episode_dir = tmp_path / "episode_000002"
    episode_dir.mkdir()
    trajectory_length = 6
    q = np.arange(trajectory_length * 9, dtype=np.float32).reshape(trajectory_length, 9)
    actions = np.arange(trajectory_length * 8, dtype=np.float32).reshape(trajectory_length, 8)
    np.savez_compressed(
        episode_dir / "trajectory.npz",
        q=q,
        dq=np.zeros_like(q),
        eef_pose=np.zeros((trajectory_length, 7), dtype=np.float32),
        action=actions,
    )
    np.savez_compressed(
        episode_dir / "camera.npz",
        rgb=np.zeros((3, 4, 5, 3), dtype=np.uint8),
        frame_step=np.asarray([1, 3, 6], dtype=np.int64),
        timestamp=np.asarray([0.02, 0.06, 0.12], dtype=np.float64),
    )
    (episode_dir / "metadata.json").write_text(
        json.dumps({"success": True, "camera_frames": 3}), encoding="utf-8"
    )

    episode = align_episode(episode_dir, action_offset_steps=1)

    # The final camera frame has no subsequently recorded action and is dropped.
    np.testing.assert_array_equal(episode.state_trajectory_indices, [0, 2])
    np.testing.assert_array_equal(episode.trajectory_indices, [1, 3])
    np.testing.assert_array_equal(episode.action[:, 0], [8, 24])
    np.testing.assert_array_equal(episode.state[:, 0], [0, 18])
    assert episode.rgb.shape[0] == 2


def test_visual_fsm_bounded_actions_cover_all_phases_and_step_boundary() -> None:
    eef_pose = np.asarray(
        [
            [0.40, 0.00, 0.30, 1.0, 0.0, 0.0, 0.0],
            [0.50, 0.00, 0.11, 1.0, 0.0, 0.0, 0.0],
            [0.50, 0.00, 0.07, 1.0, 0.0, 0.0, 0.0],
            [0.50, 0.00, 0.10, 1.0, 0.0, 0.0, 0.0],
            [0.50, 0.00, 0.10, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    visual_position = np.asarray(
        [
            [0.70, 0.20, 0.03],
            [0.50, 0.00, 0.03],
            [0.50, 0.00, 0.03],
            [0.50, 0.00, 0.03],
            [0.50, 0.00, 0.03],
        ],
        dtype=np.float32,
    )
    phase = np.arange(5, dtype=np.int64)

    actions = build_visual_fsm_bounded_actions(
        eef_pose, visual_position, phase, maximum_position_step_m=0.04
    )

    np.testing.assert_allclose(
        actions[:, :3],
        [
            [0.40, 0.00, 0.30],
            [0.50, 0.00, 0.15],
            [0.50, 0.00, 0.03],
            [0.50, 0.00, 0.06],
            [0.50, 0.00, 0.14],
        ],
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(actions[0, 3:7], eef_pose[0, 3:7])
    np.testing.assert_array_equal(
        actions[1:, 3:7],
        np.tile(np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32), (4, 1)),
    )
    np.testing.assert_array_equal(actions[:, 7], [1.0, 1.0, 1.0, -1.0, -1.0])
    step_lengths = np.linalg.norm(actions[:, :3] - eef_pose[:, :3], axis=1)
    assert np.all(step_lengths <= 0.040001)


def test_visual_fsm_bounded_actions_reject_invalid_phase() -> None:
    with pytest.raises(ValueError, match=r"within \[0, 4\]"):
        build_visual_fsm_bounded_actions(
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([5], dtype=np.int64),
        )


def test_visual_fsm_bounded_delta_actions_are_current_eef_relative() -> None:
    eef_pose = np.asarray(
        [
            [0.40, 0.00, 0.30, 1.0, 0.0, 0.0, 0.0],
            [0.50, 0.00, 0.10, 1.0, 0.0, 0.0, 0.0],
            [0.50, 0.00, 0.10, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    visual_position = np.asarray(
        [[0.70, 0.20, 0.03], [0.50, 0.00, 0.03], [0.50, 0.00, 0.03]],
        dtype=np.float32,
    )
    phase = np.asarray([0, 3, 4], dtype=np.int64)

    absolute = build_visual_fsm_bounded_actions(
        eef_pose, visual_position, phase, maximum_position_step_m=0.04
    )
    relative = build_visual_fsm_bounded_actions(
        eef_pose,
        visual_position,
        phase,
        maximum_position_step_m=0.04,
        position_target_mode="delta",
    )

    np.testing.assert_allclose(relative[:, :3], absolute[:, :3] - eef_pose[:, :3])
    np.testing.assert_allclose(relative[0, :3], 0.0)
    assert np.all(np.linalg.norm(relative[:, :3], axis=1) <= 0.040001)
    np.testing.assert_array_equal(relative[:, 3:], absolute[:, 3:])


def test_visual_fsm_bounded_actions_reject_invalid_position_mode() -> None:
    with pytest.raises(ValueError, match="position_target_mode"):
        build_visual_fsm_bounded_actions(
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            position_target_mode="robot_frame_velocity",
        )


def test_hybrid_guard_falls_back_for_invalid_vla_action() -> None:
    guard = HybridActionGuard()
    fallback = np.asarray([0.5, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    decision = guard.filter(
        np.full(8, np.nan, dtype=np.float32),
        fallback,
        current_eef_pose=fallback[:7],
        visual_object_position=np.asarray([0.5, 0.0, 0.03], dtype=np.float32),
        perception_valid=True,
    )
    assert decision.source == "scripted_ik"
    assert decision.reason == "invalid_vla_action"
    np.testing.assert_array_equal(decision.action, fallback)


def test_hybrid_guard_limits_motion_and_blocks_early_close() -> None:
    guard = HybridActionGuard(maximum_position_step_m=0.04)
    fallback = np.asarray([0.5, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    candidate = np.asarray([0.8, 0.2, 0.5, 0.0, 2.0, 0.0, 0.0, -1.0], dtype=np.float32)
    decision = guard.filter(
        candidate,
        fallback,
        current_eef_pose=fallback[:7],
        visual_object_position=np.asarray([0.7, 0.2, 0.03], dtype=np.float32),
        perception_valid=True,
    )
    assert decision.source == "vla"
    assert decision.position_was_clipped
    assert decision.gripper_was_blocked
    assert np.linalg.norm(decision.action[:3] - fallback[:3]) <= 0.040001
    assert np.isclose(np.linalg.norm(decision.action[3:7]), 1.0)
    assert decision.action[7] == 1.0


def test_delta_xyz_is_decoded_before_guard_at_execution_time() -> None:
    guard = HybridActionGuard(maximum_position_step_m=0.04)
    current_eef = np.asarray(
        [0.50, -0.10, 0.20, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
    )
    fallback = np.concatenate((current_eef, [1.0])).astype(np.float32)
    delta_action = np.asarray(
        [0.02, -0.01, 0.03, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32
    )

    decoded = decode_vla_position_target(
        delta_action,
        current_eef,
        position_target_mode="delta",
    )
    decision = guard.filter(
        decoded,
        fallback,
        current_eef_pose=current_eef,
        visual_object_position=np.asarray([0.52, -0.11, 0.03], dtype=np.float32),
        perception_valid=True,
    )

    np.testing.assert_allclose(decision.action[:3], [0.52, -0.11, 0.23])
    assert decision.source == "vla"


def test_hybrid_guard_latches_grasp() -> None:
    guard = HybridActionGuard(grasp_enable_distance_m=0.075)
    eef = np.asarray([0.5, 0.0, 0.04, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    fallback = np.concatenate((eef, [1.0])).astype(np.float32)
    close = np.concatenate((eef, [-1.0])).astype(np.float32)
    first = guard.filter(
        close,
        fallback,
        current_eef_pose=eef,
        visual_object_position=np.asarray([0.5, 0.0, 0.03], dtype=np.float32),
        perception_valid=True,
    )
    reopen = guard.filter(
        fallback,
        fallback,
        current_eef_pose=eef,
        visual_object_position=np.asarray([0.5, 0.0, 0.03], dtype=np.float32),
        perception_valid=True,
    )
    assert first.action[7] == -1.0
    assert reopen.action[7] == -1.0


def test_rpc_numpy_payload_roundtrip() -> None:
    request = VLARequest(
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        state=np.arange(25, dtype=np.float32),
        task="lift the cube",
    )
    decoded = decode_arrays(encode_arrays(request.as_arrays()))
    assert decoded["rgb"].shape == (8, 10, 3)
    assert str(decoded["task"].item()) == "lift the cube"
    response = VLAResponse.from_arrays(
        {
            "actions": np.zeros((8, 8), dtype=np.float32),
            "inference_ms": np.asarray(12.5, dtype=np.float32),
        }
    )
    assert response.actions.shape == (8, 8)
    assert response.inference_ms == 12.5


def test_rpc_request_accepts_visual_conditioned_29d_state() -> None:
    request = VLARequest(
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        state=np.arange(29, dtype=np.float32),
        task="lift the cube",
    )

    decoded = decode_arrays(encode_arrays(request.as_arrays()))

    assert decoded["state"].shape == (29,)


def test_rpc_request_accepts_phase_conditioned_30d_state() -> None:
    request = VLARequest(
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        state=np.arange(30, dtype=np.float32),
        task="lift the cube",
    )

    decoded = decode_arrays(encode_arrays(request.as_arrays()))

    assert decoded["state"].shape == (30,)
