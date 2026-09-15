"""Pure tensor math for PPO shaping rewards, separated for unit testing."""

from __future__ import annotations

import torch


def proximity_gated_close_score(
    distance: torch.Tensor,
    finger_mean: torch.Tensor,
    *,
    distance_std: float,
    open_finger_position: float,
) -> torch.Tensor:
    if min(distance_std, open_finger_position) <= 0.0:
        raise ValueError("distance_std and open_finger_position must be positive")
    proximity = 1.0 - torch.tanh(distance / distance_std)
    closed_fraction = 1.0 - torch.clamp(finger_mean / open_finger_position, 0.0, 1.0)
    return proximity * closed_fraction


def normalized_height_progress(
    height: torch.Tensor, *, start_height: float, target_height: float
) -> torch.Tensor:
    if target_height <= start_height:
        raise ValueError("target_height must be greater than start_height")
    return torch.clamp(
        (height - start_height) / (target_height - start_height), 0.0, 1.0
    )
