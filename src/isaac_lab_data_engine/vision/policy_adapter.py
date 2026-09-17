"""Replace privileged object-state terms with visual position estimates."""

from __future__ import annotations

import torch

from .rgbd_pose import quaternion_apply, quaternion_conjugate


class VisualPolicyObservationAdapter:
    """Build the 57-D IK policy observation without object ground truth.

    The trained DAgger/PPO actor expects the project's fixed concatenation:
    joint position (9), joint velocity (9), object position (3), previous
    action (8), end-effector pose (7), object pose (7), object-in-EEF (3),
    grasp state (6), and adaptive phase (5).  Robot proprioception remains
    available; every feature that depends on the object is recomputed from the
    RGB-D position estimate.  Object orientation is intentionally set to the
    neutral quaternion because the first visual model predicts position only.
    """

    OBSERVATION_DIM = 57
    OBJECT_POSITION = slice(18, 21)
    END_EFFECTOR_POSE = slice(29, 36)
    OBJECT_POSE = slice(36, 43)
    OBJECT_IN_EEF = slice(43, 46)
    GRASP_STATE = slice(46, 52)
    ADAPTIVE_PHASE = slice(52, 57)

    def __init__(
        self,
        *,
        ema_alpha: float = 0.45,
        maximum_xy_error_m: float = 0.025,
        maximum_grasp_distance_m: float = 0.04,
        maximum_held_distance_m: float = 0.08,
    ) -> None:
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.ema_alpha = float(ema_alpha)
        self.maximum_xy_error_m = float(maximum_xy_error_m)
        self.maximum_grasp_distance_m = float(maximum_grasp_distance_m)
        self.maximum_held_distance_m = float(maximum_held_distance_m)
        self._filtered_position: torch.Tensor | None = None

    def reset(self) -> None:
        self._filtered_position = None

    def filter_position(self, position_b: torch.Tensor) -> torch.Tensor:
        if position_b.ndim != 2 or position_b.shape[1] != 3:
            raise ValueError(f"Expected object positions [N,3], got {tuple(position_b.shape)}")
        if self._filtered_position is None or self._filtered_position.shape != position_b.shape:
            self._filtered_position = position_b.detach().clone()
        else:
            self._filtered_position.lerp_(position_b, self.ema_alpha)
        return self._filtered_position.clone()

    def patch(
        self,
        policy_observation: torch.Tensor,
        visual_object_position_b: torch.Tensor,
        *,
        elapsed_s: torch.Tensor,
        apply_filter: bool = True,
    ) -> torch.Tensor:
        """Return an actor observation whose object terms come from RGB-D."""

        if policy_observation.ndim != 2 or policy_observation.shape[1] != self.OBSERVATION_DIM:
            raise ValueError(
                f"Expected policy observations [N,{self.OBSERVATION_DIM}], "
                f"got {tuple(policy_observation.shape)}"
            )
        count = policy_observation.shape[0]
        if visual_object_position_b.shape != (count, 3):
            raise ValueError("visual_object_position_b must have shape [N,3]")
        if elapsed_s.shape not in {(count,), (count, 1)}:
            raise ValueError("elapsed_s must have shape [N] or [N,1]")
        elapsed_s = elapsed_s.reshape(count)
        object_position = (
            self.filter_position(visual_object_position_b)
            if apply_filter
            else visual_object_position_b
        )
        result = policy_observation.clone()
        ee_pose = result[:, self.END_EFFECTOR_POSE]
        ee_position = ee_pose[:, :3]
        ee_quaternion = ee_pose[:, 3:7]

        result[:, self.OBJECT_POSITION] = object_position
        result[:, self.OBJECT_POSE] = 0.0
        result[:, 36:39] = object_position
        result[:, 39] = 1.0
        result[:, self.OBJECT_IN_EEF] = quaternion_apply(
            quaternion_conjugate(ee_quaternion), object_position - ee_position
        )

        delta = ee_position - object_position
        xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        distance = torch.linalg.vector_norm(delta, dim=-1)
        finger_mean = result[:, 7:9].mean(dim=-1)
        finger_speed = result[:, 16:18].abs().mean(dim=-1)
        held = (
            (finger_mean < 0.025)
            & (finger_mean > 0.005)
            & (distance < self.maximum_held_distance_m)
        )
        result[:, self.GRASP_STATE] = torch.stack(
            (xy_distance, delta[:, 2], distance, finger_mean, finger_speed, held.float()),
            dim=-1,
        )

        phase = torch.ones(count, dtype=torch.long, device=result.device)
        phase[xy_distance < self.maximum_xy_error_m] = 2
        phase[(distance < self.maximum_grasp_distance_m) & ~held] = 3
        phase[held] = 4
        phase[elapsed_s < 0.20] = 0
        result[:, self.ADAPTIVE_PHASE] = torch.nn.functional.one_hot(
            phase, num_classes=5
        ).to(result.dtype)
        return result
