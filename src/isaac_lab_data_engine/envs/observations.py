"""Task-specific observation terms for the Pick-and-Lift environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _finger_joint_values(
    robot: object, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read Franka finger position/velocity in managed and direct calls."""

    joint_ids = robot_cfg.joint_ids
    # ObservationManager resolves the configured regex to [7, 8].  The
    # adaptive scripted teacher also calls these functions directly, where an
    # unresolved SceneEntityCfg still contains slice(None); use the known
    # Franka convention in that case instead of accidentally averaging all
    # seven arm joints together with the fingers.
    if joint_ids == slice(None):
        joint_ids = slice(-2, None)
    return robot.data.joint_pos[:, joint_ids], robot.data.joint_vel[:, joint_ids]


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


def episode_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return elapsed episode time in seconds as a one-dimensional clock.

    Pick-and-lift contains dwell phases (rest and grasp closure) that cannot be
    inferred uniquely from an instantaneous geometric state.  Supplying the
    clock makes those phases observable to a feed-forward policy while keeping
    the value independent of the configured episode horizon.
    """

    return (env.episode_length_buf.float() * env.step_dt).unsqueeze(-1)


def grasp_state_features(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    maximum_held_distance_m: float = 0.08,
    maximum_closed_finger_position_m: float = 0.025,
    minimum_contact_finger_position_m: float = 0.005,
) -> torch.Tensor:
    """Return geometry and grasp feedback used by the adaptive policy.

    The final value is a deterministic held/contact proxy: the fingers are
    closed and the object remains close to the end effector.  It is available
    without adding an expensive force sensor to every parallel environment.
    """

    robot = env.scene[robot_cfg.name]
    ee_position = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
    object_position = env.scene[object_cfg.name].data.root_pos_w
    delta = ee_position - object_position
    xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=-1)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    finger_position, finger_velocity = _finger_joint_values(robot, robot_cfg)
    finger_mean = finger_position.mean(dim=-1)
    finger_speed = finger_velocity.abs().mean(dim=-1)
    held = (
        (finger_mean < maximum_closed_finger_position_m)
        & (finger_mean > minimum_contact_finger_position_m)
        & (distance < maximum_held_distance_m)
    )
    return torch.stack(
        (
            xy_distance,
            delta[:, 2],
            distance,
            finger_mean,
            finger_speed,
            held.float(),
        ),
        dim=-1,
    )


def adaptive_manipulation_phase(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rest_time_s: float = 0.20,
    maximum_xy_error_m: float = 0.025,
    maximum_grasp_distance_m: float = 0.04,
    maximum_held_distance_m: float = 0.08,
    maximum_closed_finger_position_m: float = 0.025,
    minimum_contact_finger_position_m: float = 0.005,
) -> torch.Tensor:
    """Return a recoverable rest/approach/descend/grasp/lift phase one-hot.

    Unlike the previous fixed clock, transitions depend on the current robot
    and object state. If tracking is slow in a randomized scene, the phase
    waits; if the object slips, it can fall back from lift to re-grasp.
    """

    robot = env.scene[robot_cfg.name]
    ee_position = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
    object_position = env.scene[object_cfg.name].data.root_pos_w
    delta = ee_position - object_position
    xy_distance = torch.linalg.vector_norm(delta[:, :2], dim=-1)
    distance = torch.linalg.vector_norm(delta, dim=-1)
    finger_position, _ = _finger_joint_values(robot, robot_cfg)
    finger_mean = finger_position.mean(dim=-1)
    elapsed = env.episode_length_buf.float() * env.step_dt

    phase = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
    # Once horizontally aligned, remain in descend until the grasp pose is
    # reached.  Requiring a minimum vertical gap here would send the policy
    # back to approach in the final centimetres and create a phase oscillation.
    aligned_above = xy_distance < maximum_xy_error_m
    near_object = distance < maximum_grasp_distance_m
    gripper_closed = finger_mean < maximum_closed_finger_position_m
    held = (
        gripper_closed
        & (finger_mean > minimum_contact_finger_position_m)
        & (distance < maximum_held_distance_m)
    )
    phase[aligned_above] = 2
    phase[near_object & ~held] = 3
    phase[held] = 4
    phase[elapsed < rest_time_s] = 0
    return torch.nn.functional.one_hot(phase, num_classes=5).float()
