"""Saved labels must not silently become a stronger online expert."""

import json

import numpy as np
import pytest

from isaac_lab_data_engine.vla.label_replay import TrainingLabelReplay


def make_bundle(tmp_path):
    states = np.zeros((2, 30), dtype=np.float32)
    states[:, 18:25] = [0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    states[:, 25:28] = [0.55, 0.0, 0.03]
    states[:, 29] = 1
    actions = np.array([[0.02, 0, -0.03, 0, 1, 0, 0, 1],
                        [0.01, 0, -0.02, 0, 1, 0, 0, 1]], dtype=np.float32)
    path = tmp_path / "bundle.npz"
    np.savez(path, easy_states=states, easy_actions=actions,
             easy_domain_json=np.asarray(json.dumps({"sample_index": 92})),
             easy_source=np.asarray("original source"), easy_episode_id=np.asarray(0),
             easy_frame_indices=np.asarray([0, 1]),
             semantics_json=np.asarray(json.dumps({
                 "maximum_position_step_m": 0.04, "approach_height_m": 0.12, "lift_height_m": 0.18
             })))
    return path, states, actions


def test_stored_replay_returns_exact_labels_without_relabeling(tmp_path):
    path, states, actions = make_bundle(tmp_path)
    replay = TrainingLabelReplay(path, "easy", "stored")
    live_eef = states[0, 18:25].copy()
    live_eef[:3] += 0.1
    for action in actions:
        np.testing.assert_array_equal(replay.next_action(live_eef, np.zeros(3), 4), action)
    assert replay.next_action(live_eef, np.zeros(3), 4) is None
    assert replay.next_action(live_eef, np.zeros(3), 4) is None
    assert replay.cursor == len(actions)


def test_online_labels_use_live_state_and_bound_displacement(tmp_path):
    path, states, _ = make_bundle(tmp_path)
    replay = TrainingLabelReplay(path, "easy", "online")
    eef = states[0, 18:25]
    target = states[0, 25:28]
    action = replay.next_action(eef, target, 1)
    expected_direction = target + [0, 0, 0.12] - eef[:3]
    expected = expected_direction / np.linalg.norm(expected_direction) * 0.04
    np.testing.assert_allclose(action[:3], expected, atol=1e-7)
    assert action[7] == 1
    rest = replay.next_action(eef, target, 0)
    np.testing.assert_array_equal(rest[:3], [0, 0, 0])


def test_invalid_replay_mode_is_rejected(tmp_path):
    path, _, _ = make_bundle(tmp_path)
    with pytest.raises(ValueError, match="stored or online"):
        TrainingLabelReplay(path, "easy", "invalid")
