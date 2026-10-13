from __future__ import annotations

import numpy as np
import pytest

from isaac_lab_data_engine.vla.recovery import (
    RecoveryBand,
    build_pregrasp_recovery_actions,
    distance_to_pregrasp,
    pregrasp_waypoint,
    validate_recovery_episode,
)


def test_pregrasp_distance_and_band_are_centered_at_15_mm() -> None:
    visual = np.asarray([0.55, 0.02, 0.03], dtype=np.float32)
    waypoint = pregrasp_waypoint(visual)
    np.testing.assert_allclose(waypoint, [0.55, 0.02, 0.15], atol=1.0e-7)
    eef = np.asarray([0.565, 0.02, 0.15, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert float(distance_to_pregrasp(eef, visual)) == pytest.approx(0.015)
    band = RecoveryBand(minimum_m=0.01, target_m=0.015, maximum_m=0.02)
    assert bool(band.contains(0.015))
    assert not bool(band.contains(0.021))


def test_recovery_action_is_delta_top_down_and_open() -> None:
    visual = np.asarray([0.55, 0.0, 0.03], dtype=np.float32)
    eef = np.asarray([0.58, 0.0, 0.15, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    action = build_pregrasp_recovery_actions(eef, visual)
    np.testing.assert_allclose(action[:3], [-0.03, 0.0, 0.0], atol=1.0e-7)
    np.testing.assert_array_equal(action[3:7], [0.0, 1.0, 0.0, 0.0])
    assert action[7] == 1.0


def test_recovery_action_clips_long_delta() -> None:
    visual = np.asarray([0.55, 0.0, 0.03], dtype=np.float32)
    eef = np.asarray([0.65, 0.0, 0.15, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    action = build_pregrasp_recovery_actions(
        eef, visual, maximum_position_step_m=0.04
    )
    assert np.linalg.norm(action[:3]) == pytest.approx(0.04)


def test_compact_recovery_episode_contract() -> None:
    frames = 2
    state = np.zeros((frames, 30), dtype=np.float32)
    state[:, -1] = 1.0
    arrays = {
        "rgb": np.zeros((frames, 12, 16, 3), dtype=np.uint8),
        "state": state,
        "action": np.zeros((frames, 8), dtype=np.float32),
        "controller_phase": np.ones((frames, 1), dtype=np.float32),
        "difficulty_id": np.zeros((frames, 1), dtype=np.float32),
        "loss_weight": np.full((frames, 1), 2.0, dtype=np.float32),
        "distance_to_pregrasp_m": np.full((frames, 1), 0.015, dtype=np.float32),
        "learner_visited": np.asarray([[False], [True]], dtype=np.bool_),
    }
    assert validate_recovery_episode(arrays) == frames
    arrays["controller_phase"] = np.zeros((frames, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="phase 1"):
        validate_recovery_episode(arrays)


def test_recovery_band_rejects_invalid_order() -> None:
    with pytest.raises(ValueError, match="minimum"):
        RecoveryBand(minimum_m=0.02, target_m=0.015, maximum_m=0.03)
