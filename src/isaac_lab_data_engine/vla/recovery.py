"""Pure NumPy helpers for targeted pre-grasp DAgger recovery samples.

The online collector deliberately keeps Isaac Sim and LeRobot out of this
module.  That makes the distance gate, expert labels, and compact episode
contract testable in the ordinary project test environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .dataset import ACTION_DIM, build_visual_fsm_bounded_actions


RECOVERY_FORMAT_VERSION = 1
APPROACH_PHASE = 1
TOP_DOWN_QUATERNION = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32)


@dataclass(frozen=True)
class RecoveryBand:
    """Distance shell around the visual pre-grasp waypoint."""

    minimum_m: float = 0.005
    target_m: float = 0.015
    maximum_m: float = 0.030

    def __post_init__(self) -> None:
        values = (self.minimum_m, self.target_m, self.maximum_m)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Recovery-band distances must be finite")
        if not 0.0 <= self.minimum_m <= self.target_m <= self.maximum_m:
            raise ValueError(
                "Recovery band must satisfy 0 <= minimum <= target <= maximum"
            )
        if self.maximum_m <= 0.0:
            raise ValueError("Recovery-band maximum must be positive")

    def contains(self, distance_m: np.ndarray | float) -> np.ndarray:
        distance = np.asarray(distance_m, dtype=np.float32)
        return (distance >= self.minimum_m) & (distance <= self.maximum_m)


def pregrasp_waypoint(
    visual_object_position: np.ndarray,
    *,
    approach_height_m: float = 0.12,
) -> np.ndarray:
    """Return the phase-1 waypoint in the robot-root frame."""

    position = np.asarray(visual_object_position, dtype=np.float32)
    if position.shape[-1:] != (3,):
        raise ValueError(
            "visual_object_position must have final dimension 3, got "
            f"{position.shape}"
        )
    if not np.isfinite(position).all():
        raise ValueError("visual_object_position must be finite")
    if not np.isfinite(approach_height_m) or approach_height_m < 0.0:
        raise ValueError("approach_height_m must be finite and non-negative")
    offset = np.asarray((0.0, 0.0, approach_height_m), dtype=np.float32)
    return (position + offset).astype(np.float32, copy=False)


def distance_to_pregrasp(
    eef_pose: np.ndarray,
    visual_object_position: np.ndarray,
    *,
    approach_height_m: float = 0.12,
) -> np.ndarray:
    """Measure EEF distance to the visual phase-1 waypoint."""

    eef = np.asarray(eef_pose, dtype=np.float32)
    if eef.shape[-1:] != (7,):
        raise ValueError(f"eef_pose must have final dimension 7, got {eef.shape}")
    if not np.isfinite(eef).all():
        raise ValueError("eef_pose must be finite")
    waypoint = pregrasp_waypoint(
        visual_object_position, approach_height_m=approach_height_m
    )
    try:
        delta = np.broadcast_arrays(eef[..., :3], waypoint)[0] - np.broadcast_arrays(
            eef[..., :3], waypoint
        )[1]
    except ValueError as exc:
        raise ValueError("EEF and visual-position batch shapes are incompatible") from exc
    return np.linalg.norm(delta, axis=-1).astype(np.float32, copy=False)


def build_pregrasp_recovery_actions(
    eef_pose: np.ndarray,
    visual_object_position: np.ndarray,
    *,
    maximum_position_step_m: float = 0.04,
    approach_height_m: float = 0.12,
) -> np.ndarray:
    """Build phase-1 expert labels using the deployment delta convention."""

    eef = np.asarray(eef_pose, dtype=np.float32)
    visual = np.asarray(visual_object_position, dtype=np.float32)
    single = eef.ndim == 1
    if single:
        eef = eef[None, :]
    if visual.ndim == 1:
        visual = visual[None, :]
    if eef.ndim != 2 or eef.shape[1] != 7:
        raise ValueError(f"Expected eef_pose [F,7], got {eef.shape}")
    if visual.shape != (eef.shape[0], 3):
        raise ValueError(
            f"Expected visual_object_position [{eef.shape[0]},3], got {visual.shape}"
        )
    actions = build_visual_fsm_bounded_actions(
        eef,
        visual,
        np.full((eef.shape[0],), APPROACH_PHASE, dtype=np.int64),
        maximum_position_step_m=maximum_position_step_m,
        approach_height_m=approach_height_m,
        position_target_mode="delta",
    )
    return actions[0] if single else actions


def validate_recovery_episode(arrays: Mapping[str, np.ndarray]) -> int:
    """Validate a compact recovery episode and return its frame count."""

    required = {
        "rgb",
        "state",
        "action",
        "controller_phase",
        "difficulty_id",
        "loss_weight",
        "distance_to_pregrasp_m",
        "learner_visited",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Recovery episode is missing arrays: {sorted(missing)}")

    rgb = np.asarray(arrays["rgb"])
    state = np.asarray(arrays["state"])
    action = np.asarray(arrays["action"])
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8 [F,H,W,3], got {rgb.shape}/{rgb.dtype}")
    frames = int(rgb.shape[0])
    if frames < 1:
        raise ValueError("Recovery episode must contain at least one frame")
    if state.shape != (frames, 30):
        raise ValueError(f"state must be [{frames},30], got {state.shape}")
    if action.shape != (frames, ACTION_DIM):
        raise ValueError(
            f"action must be [{frames},{ACTION_DIM}], got {action.shape}"
        )
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state and action must be finite")

    vector_fields = {
        "controller_phase": np.float32,
        "difficulty_id": np.float32,
        "loss_weight": np.float32,
        "distance_to_pregrasp_m": np.float32,
        "learner_visited": np.bool_,
    }
    for name, dtype in vector_fields.items():
        value = np.asarray(arrays[name])
        if value.shape not in {(frames,), (frames, 1)}:
            raise ValueError(f"{name} must have one value per frame, got {value.shape}")
        if dtype is not np.bool_ and not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    phases = np.asarray(arrays["controller_phase"]).reshape(-1)
    if not np.all(phases == APPROACH_PHASE):
        raise ValueError("Recovery episodes must contain only controller phase 1")
    weights = np.asarray(arrays["loss_weight"]).reshape(-1)
    if np.any(weights <= 0.0):
        raise ValueError("loss_weight must be positive")
    distances = np.asarray(arrays["distance_to_pregrasp_m"]).reshape(-1)
    if np.any(distances < 0.0):
        raise ValueError("distance_to_pregrasp_m must be non-negative")
    if not np.allclose(state[:, -1], phases, atol=0.0, rtol=0.0):
        raise ValueError("state controller phase does not match controller_phase")
    return frames
