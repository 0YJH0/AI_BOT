from __future__ import annotations

import pytest
import torch

from isaac_lab_data_engine.envs.ppo_reward_math import (
    normalized_height_progress,
    proximity_gated_close_score,
)


def test_height_progress_is_clamped_and_normalized() -> None:
    height = torch.tensor([0.45, 0.48, 0.515, 0.55, 0.60])
    result = normalized_height_progress(height, start_height=0.48, target_height=0.55)
    torch.testing.assert_close(result, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_height_progress_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="target_height"):
        normalized_height_progress(torch.tensor([0.5]), start_height=0.5, target_height=0.5)


def test_gripper_close_reward_requires_both_proximity_and_closure() -> None:
    result = proximity_gated_close_score(
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.04, 0.0]),
        distance_std=0.06,
        open_finger_position=0.04,
    )
    assert result[0].item() == pytest.approx(1.0)
    assert result[1].item() == pytest.approx(0.0)
    assert result[2].item() < 1.0e-6
