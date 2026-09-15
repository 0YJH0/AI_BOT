"""Small reward-shaping terms used only by the optional PPO baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .ppo_reward_math import normalized_height_progress, proximity_gated_close_score

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def proximity_gated_gripper_close(
    env: ManagerBasedRLEnv,
    distance_std: float,
    open_finger_position: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the fingers only while the end effector is near the object."""

    robot: Articulation = env.scene[robot_cfg.name]
    target_object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    distance = torch.linalg.vector_norm(
        ee_frame.data.target_pos_w[:, 0, :] - target_object.data.root_pos_w, dim=-1
    )
    finger_mean = robot.data.joint_pos[:, robot_cfg.joint_ids].mean(dim=-1)
    return proximity_gated_close_score(
        distance,
        finger_mean,
        distance_std=distance_std,
        open_finger_position=open_finger_position,
    )


def object_height_progress(
    env: ManagerBasedRLEnv,
    start_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous progress for physically raising the object toward the formal success height."""

    target_object: RigidObject = env.scene[object_cfg.name]
    return normalized_height_progress(
        target_object.data.root_pos_w[:, 2],
        start_height=start_height,
        target_height=target_height,
    )
