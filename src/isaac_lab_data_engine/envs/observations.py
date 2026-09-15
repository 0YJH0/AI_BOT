"""Task-specific observation terms for the Pick-and-Lift environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def end_effector_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return end-effector position and wxyz quaternion in the robot root frame."""

    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    position, orientation = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
    )
    return torch.cat((position, orientation), dim=-1)


def object_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return object position and wxyz quaternion in the robot root frame."""

    robot = env.scene[robot_cfg.name]
    target_object = env.scene[object_cfg.name]
    position, orientation = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_object.data.root_pos_w,
        target_object.data.root_quat_w,
    )
    return torch.cat((position, orientation), dim=-1)


def object_position_in_end_effector_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object position expressed in the end-effector frame."""

    ee_frame = env.scene[ee_frame_cfg.name]
    target_object = env.scene[object_cfg.name]
    position, _ = subtract_frame_transforms(
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
        target_object.data.root_pos_w,
        target_object.data.root_quat_w,
    )
    return position
