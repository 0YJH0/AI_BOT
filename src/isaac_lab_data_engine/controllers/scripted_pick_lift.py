"""Vectorized scripted state machine for deterministic pick-and-lift rollouts."""

from __future__ import annotations

from enum import IntEnum

import torch


class PickAndLiftState(IntEnum):
    """Discrete phases of the scripted manipulation policy."""

    REST = 0
    APPROACH_ABOVE_OBJECT = 1
    APPROACH_OBJECT = 2
    GRASP_OBJECT = 3
    LIFT_OBJECT = 4


class PickAndLiftStateMachine:
    """Generate absolute end-effector pose and binary gripper actions.

    All buffers are Torch tensors on the simulation device. The implementation
    therefore remains vectorized when ``num_envs`` is increased in later phases.
    Poses use ``(x, y, z, qw, qx, qy, qz)`` ordering.
    """

    OPEN_GRIPPER = 1.0
    CLOSE_GRIPPER = -1.0

    def __init__(
        self,
        dt: float,
        num_envs: int,
        device: torch.device | str,
        position_threshold: float = 0.015,
        approach_height: float = 0.12,
    ) -> None:
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.position_threshold = float(position_threshold)
        self.approach_height = float(approach_height)

        self.state = torch.full(
            (num_envs,), int(PickAndLiftState.REST), dtype=torch.int64, device=self.device
        )
        self.elapsed = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.desired_pose = torch.zeros((num_envs, 7), dtype=torch.float32, device=self.device)
        self.gripper_action = torch.full(
            (num_envs,), self.OPEN_GRIPPER, dtype=torch.float32, device=self.device
        )
        self.top_down_quat = torch.tensor(
            (0.0, 1.0, 0.0, 0.0), dtype=torch.float32, device=self.device
        ).repeat(num_envs, 1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all or selected state-machine instances."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.state[env_ids] = int(PickAndLiftState.REST)
        self.elapsed[env_ids] = 0.0
        self.gripper_action[env_ids] = self.OPEN_GRIPPER

    def _transition(self, mask: torch.Tensor, next_state: PickAndLiftState) -> None:
        self.state[mask] = int(next_state)
        self.elapsed[mask] = 0.0

    def compute(
        self,
        ee_pose_b: torch.Tensor,
        object_pose_b: torch.Tensor,
        lift_pose_b: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the state machine and return an ``(N, 8)`` action tensor."""

        if ee_pose_b.shape != (self.num_envs, 7):
            raise ValueError(f"Expected end-effector pose {(self.num_envs, 7)}, got {tuple(ee_pose_b.shape)}")
        if object_pose_b.shape != (self.num_envs, 7):
            raise ValueError(f"Expected object pose {(self.num_envs, 7)}, got {tuple(object_pose_b.shape)}")
        if lift_pose_b.shape != (self.num_envs, 7):
            raise ValueError(f"Expected lift pose {(self.num_envs, 7)}, got {tuple(lift_pose_b.shape)}")

        object_grasp_pose = object_pose_b.clone()
        object_grasp_pose[:, 3:7] = self.top_down_quat
        approach_pose = object_grasp_pose.clone()
        approach_pose[:, 2] += self.approach_height
        lift_pose = lift_pose_b.clone()
        lift_pose[:, 3:7] = self.top_down_quat

        self.desired_pose.copy_(ee_pose_b)
        self.gripper_action.fill_(self.OPEN_GRIPPER)

        rest = self.state == int(PickAndLiftState.REST)
        above = self.state == int(PickAndLiftState.APPROACH_ABOVE_OBJECT)
        descend = self.state == int(PickAndLiftState.APPROACH_OBJECT)
        grasp = self.state == int(PickAndLiftState.GRASP_OBJECT)
        lift = self.state == int(PickAndLiftState.LIFT_OBJECT)

        self.desired_pose[above] = approach_pose[above]
        self.desired_pose[descend] = object_grasp_pose[descend]
        self.desired_pose[grasp] = object_grasp_pose[grasp]
        self.desired_pose[lift] = lift_pose[lift]
        self.gripper_action[grasp | lift] = self.CLOSE_GRIPPER

        position_error = torch.linalg.vector_norm(ee_pose_b[:, :3] - self.desired_pose[:, :3], dim=-1)
        self._transition(rest & (self.elapsed >= 0.20), PickAndLiftState.APPROACH_ABOVE_OBJECT)
        self._transition(
            above & (position_error < self.position_threshold) & (self.elapsed >= 0.35),
            PickAndLiftState.APPROACH_OBJECT,
        )
        self._transition(
            descend & (position_error < self.position_threshold) & (self.elapsed >= 0.25),
            PickAndLiftState.GRASP_OBJECT,
        )
        self._transition(grasp & (self.elapsed >= 0.55), PickAndLiftState.LIFT_OBJECT)

        self.elapsed += self.dt
        return torch.cat((self.desired_pose, self.gripper_action.unsqueeze(-1)), dim=-1)

