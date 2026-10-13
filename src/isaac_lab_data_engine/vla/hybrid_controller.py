"""Safety and fallback policy for VLA-generated absolute IK actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridActionDecision:
    action: np.ndarray
    source: str
    reason: str
    position_was_clipped: bool = False
    gripper_was_blocked: bool = False


def decode_vla_position_target(
    vla_action: np.ndarray | None,
    current_eef_pose: np.ndarray,
    *,
    position_target_mode: str = "absolute",
) -> np.ndarray | None:
    """Decode VLA xyz into the absolute target consumed by Isaac control.

    Call this when a predicted chunk element is dequeued for execution. For a
    delta checkpoint, anchoring the whole chunk to the EEF pose at RPC request
    time would make all later elements use a stale origin.

    Invalid action shapes/values are deliberately left for
    :class:`HybridActionGuard` to reject and route to the fallback controller.
    """

    if position_target_mode not in {"absolute", "delta"}:
        raise ValueError("position_target_mode must be 'absolute' or 'delta'")
    if vla_action is None:
        return None
    action = np.asarray(vla_action, dtype=np.float32).copy()
    if position_target_mode == "absolute":
        return action
    eef = np.asarray(current_eef_pose, dtype=np.float32)
    if eef.shape != (7,) or not np.isfinite(eef).all():
        raise ValueError("current_eef_pose must be a finite [7] vector")
    if action.shape == (8,):
        action[:3] += eef[:3]
    return action


class HybridActionGuard:
    """Validate VLA actions and fall back to deterministic visual Scripted IK.

    The guard is intentionally model-agnostic so it can be unit-tested without
    Isaac Sim, Torch, or LeRobot. Actions use xyz + quaternion(wxyz) + gripper.
    """

    def __init__(
        self,
        *,
        workspace_min: tuple[float, float, float] = (0.32, -0.35, 0.015),
        workspace_max: tuple[float, float, float] = (0.82, 0.35, 0.60),
        maximum_position_step_m: float = 0.04,
        grasp_enable_distance_m: float = 0.075,
        close_threshold: float = 0.0,
        use_visual_grasp_gate: bool = True,
    ) -> None:
        self.workspace_min = np.asarray(workspace_min, dtype=np.float32)
        self.workspace_max = np.asarray(workspace_max, dtype=np.float32)
        self.maximum_position_step_m = float(maximum_position_step_m)
        self.grasp_enable_distance_m = float(grasp_enable_distance_m)
        self.close_threshold = float(close_threshold)
        self.use_visual_grasp_gate = bool(use_visual_grasp_gate)
        if np.any(self.workspace_min >= self.workspace_max):
            raise ValueError("workspace_min must be smaller than workspace_max")
        if self.maximum_position_step_m <= 0.0 or self.grasp_enable_distance_m <= 0.0:
            raise ValueError("Safety distances must be positive")
        self._grasp_latched = False

    def reset(self) -> None:
        self._grasp_latched = False

    @property
    def grasp_latched(self) -> bool:
        return self._grasp_latched

    def filter(
        self,
        vla_action: np.ndarray | None,
        fallback_action: np.ndarray,
        *,
        current_eef_pose: np.ndarray,
        visual_object_position: np.ndarray,
        perception_valid: bool,
    ) -> HybridActionDecision:
        fallback = self._as_action(fallback_action, "fallback_action")
        eef = np.asarray(current_eef_pose, dtype=np.float32)
        visual_position = np.asarray(visual_object_position, dtype=np.float32)
        if eef.shape != (7,) or visual_position.shape != (3,):
            raise ValueError("current_eef_pose must be [7] and visual_object_position must be [3]")
        if not perception_valid or (self.use_visual_grasp_gate and not np.isfinite(visual_position).all()):
            return HybridActionDecision(fallback.copy(), "scripted_ik", "invalid_perception")
        if vla_action is None:
            return HybridActionDecision(fallback.copy(), "scripted_ik", "vla_unavailable")
        try:
            candidate = self._as_action(vla_action, "vla_action").copy()
        except ValueError:
            return HybridActionDecision(fallback.copy(), "scripted_ik", "invalid_vla_action")

        clipped_position = np.clip(candidate[:3], self.workspace_min, self.workspace_max)
        position_was_clipped = not np.allclose(clipped_position, candidate[:3])
        delta = clipped_position - eef[:3]
        distance = float(np.linalg.norm(delta))
        if distance > self.maximum_position_step_m:
            clipped_position = eef[:3] + delta * (self.maximum_position_step_m / distance)
            position_was_clipped = True
        candidate[:3] = clipped_position

        quaternion_norm = float(np.linalg.norm(candidate[3:7]))
        if quaternion_norm < 1.0e-6:
            return HybridActionDecision(fallback.copy(), "scripted_ik", "invalid_quaternion")
        candidate[3:7] /= quaternion_norm

        wants_closed = bool(candidate[7] < self.close_threshold)
        object_distance = float(np.linalg.norm(eef[:3] - visual_position)) if self.use_visual_grasp_gate else None
        gripper_was_blocked = False
        if wants_closed and not self._grasp_latched:
            if not self.use_visual_grasp_gate or object_distance <= self.grasp_enable_distance_m:
                self._grasp_latched = True
            else:
                candidate[7] = 1.0
                gripper_was_blocked = True
        elif self._grasp_latched:
            candidate[7] = -1.0
        else:
            candidate[7] = 1.0

        return HybridActionDecision(
            action=candidate.astype(np.float32, copy=False),
            source="vla",
            reason="accepted_with_safety_filter" if position_was_clipped or gripper_was_blocked else "accepted",
            position_was_clipped=position_was_clipped,
            gripper_was_blocked=gripper_was_blocked,
        )

    @staticmethod
    def _as_action(value: np.ndarray, name: str) -> np.ndarray:
        action = np.asarray(value, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(f"{name} must be a finite [8] vector")
        return action
