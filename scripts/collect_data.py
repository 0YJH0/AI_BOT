"""Collect Phase 3-5 Pick-and-Lift trajectories, including vectorized batches."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect vectorized Pick-and-Lift episode batches.")
parser.add_argument(
    "--num_envs", "--num-envs", dest="num_envs", type=int, default=1,
    help="Number of environments simulated in parallel.",
)
parser.add_argument(
    "--episodes", type=int, default=1,
    help="Number of synchronized batches; total written episodes equals batches times num_envs.",
)
parser.add_argument("--max_steps", type=int, default=600, help="Maximum environment steps per episode.")
parser.add_argument(
    "--output_dir",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "dataset",
    help="Dataset root directory.",
)
parser.add_argument(
    "--randomization_config",
    type=Path,
    default=None,
    help="Optional Phase 4 YAML configuration applied at every episode reset.",
)
parser.add_argument(
    "--camera_frame_interval",
    type=int,
    default=5,
    help="When cameras are enabled, save one RGB-D frame every N control steps (default: 5).",
)
parser.add_argument(
    "--camera_ready_attempts",
    type=int,
    default=8,
    help="Maximum render/update attempts used to reject stale or blank camera frames.",
)
parser.add_argument(
    "--camera_min_rgb_std",
    type=float,
    default=10.0,
    help="Minimum per-environment RGB standard deviation required before recording.",
)
parser.add_argument(
    "--camera_min_rgb_range",
    type=float,
    default=32.0,
    help="Minimum per-environment RGB dynamic range required before recording.",
)
parser.add_argument(
    "--camera_min_depth_fraction",
    type=float,
    default=0.10,
    help="Minimum fraction of finite positive depth pixels required before recording.",
)
parser.add_argument(
    "--camera_warmup_steps",
    type=int,
    default=60,
    help="Maximum unrecorded control steps used to obtain consecutive valid camera frames.",
)
parser.add_argument(
    "--camera_warmup_consecutive_frames",
    type=int,
    default=3,
    help="Consecutive valid rendered frames required before an episode is recorded.",
)
parser.add_argument(
    "--camera_batch_retries",
    type=int,
    default=5,
    help="Number of full reset/randomize retries after a camera-quality rejection.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Phase 3-5 record compact state trajectories. ``--enable_cameras`` creates
# the sensor so Phase 4 can also apply and verify camera-pose randomization;
# image frame export itself remains a Phase 10 feature.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.controllers import PickAndLiftStateMachine
    from isaac_lab_data_engine.data_collection import (
        CameraFrameBuffer,
        EpisodeBuffer,
        EpisodeWriter,
        validate_episode,
    )
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.randomization import DomainRandomizer, create_domain_randomizer
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


def _task_poses(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    target_object = env.scene["object"]
    ee_position, ee_orientation = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
    )
    object_position, object_orientation = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_object.data.root_pos_w,
        target_object.data.root_quat_w,
    )
    return (
        torch.cat((ee_position, ee_orientation), dim=-1),
        torch.cat((object_position, object_orientation), dim=-1),
    )


def _cpu_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def _camera_rgb_quality(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a conservative per-environment blank-frame quality check."""

    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise RuntimeError(f"Expected batched RGB/RGBA tensor, got shape {tuple(rgb.shape)}")
    rgb_float = rgb[..., :3].to(dtype=torch.float32)
    flattened = rgb_float.reshape(rgb_float.shape[0], -1)
    rgb_std = flattened.std(dim=1, unbiased=False)
    rgb_range = flattened.amax(dim=1) - flattened.amin(dim=1)
    finite = torch.isfinite(flattened).all(dim=1)
    return rgb_std, rgb_range, finite


class CameraQualityError(RuntimeError):
    """Raised when an RGB-D batch cannot be made safe to record."""


def _read_camera_snapshot_until_ready(
    env: ManagerBasedRLEnv,
    *,
    max_attempts: int,
    minimum_rgb_std: float,
    minimum_rgb_range: float,
    minimum_depth_fraction: float,
    minimum_attempts: int,
    render_before_first: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    """Read valid per-env camera rows, retrying only rows whose annotators dropped."""

    camera = env.scene["table_camera"]
    if minimum_attempts < 1 or minimum_attempts > max_attempts:
        raise ValueError("minimum_attempts must be within [1, max_attempts]")
    accepted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    accepted_std = torch.full((env.num_envs,), float("nan"), device=env.device)
    accepted_range = torch.full((env.num_envs,), float("nan"), device=env.device)
    accepted_depth_fraction = torch.full((env.num_envs,), float("nan"), device=env.device)
    snapshot: dict[str, torch.Tensor] | None = None
    last_std: torch.Tensor | None = None
    last_range: torch.Tensor | None = None
    last_depth_fraction: torch.Tensor | None = None
    last_valid: torch.Tensor | None = None
    for attempt in range(1, max_attempts + 1):
        if render_before_first or attempt > 1:
            env.sim.render()
            camera.update(0.0, force_recompute=True)
        camera_data = camera.data
        current = {
            "rgb": camera_data.output["rgb"],
            "depth": camera_data.output["distance_to_image_plane"],
            "intrinsics": camera_data.intrinsic_matrices,
            "camera_pose": torch.cat((camera_data.pos_w, camera_data.quat_w_ros), dim=-1),
        }
        last_std, last_range, rgb_finite = _camera_rgb_quality(current["rgb"])
        depth = current["depth"].to(dtype=torch.float32).reshape(env.num_envs, -1)
        depth_valid = torch.isfinite(depth) & (depth > 0.0)
        last_depth_fraction = depth_valid.to(dtype=torch.float32).mean(dim=1)
        intrinsics = current["intrinsics"]
        camera_pose = current["camera_pose"]
        sensor_state_valid = (
            torch.isfinite(intrinsics).reshape(env.num_envs, -1).all(dim=1)
            & (intrinsics[:, 0, 0] > 0.0)
            & (intrinsics[:, 1, 1] > 0.0)
            & torch.isfinite(camera_pose).all(dim=1)
        )
        quaternion_norm = torch.linalg.vector_norm(camera_pose[:, 3:7], dim=1)
        sensor_state_valid &= (quaternion_norm > 0.5) & (quaternion_norm < 1.5)
        last_valid = (
            rgb_finite
            & (last_std >= minimum_rgb_std)
            & (last_range >= minimum_rgb_range)
            & (last_depth_fraction >= minimum_depth_fraction)
            & sensor_state_valid
        )
        if snapshot is None:
            snapshot = {key: torch.empty_like(value) for key, value in current.items()}
        newly_valid = last_valid & ~accepted
        if bool(newly_valid.any()):
            for key, value in current.items():
                snapshot[key][newly_valid] = value[newly_valid].clone()
            accepted_std[newly_valid] = last_std[newly_valid]
            accepted_range[newly_valid] = last_range[newly_valid]
            accepted_depth_fraction[newly_valid] = last_depth_fraction[newly_valid]
            accepted |= newly_valid
        if attempt >= minimum_attempts and bool(accepted.all()):
            return snapshot, {
                "attempts": attempt,
                "forced_renders": attempt if render_before_first else attempt - 1,
                "minimum_rgb_std": float(accepted_std.min().item()),
                "minimum_rgb_range": float(accepted_range.min().item()),
                "minimum_depth_fraction": float(accepted_depth_fraction.min().item()),
            }

    assert (
        last_std is not None
        and last_range is not None
        and last_depth_fraction is not None
        and last_valid is not None
    )
    invalid_ids = torch.nonzero(~accepted, as_tuple=False).squeeze(-1).cpu().tolist()
    std_values = last_std.detach().cpu().tolist()
    range_values = last_range.detach().cpu().tolist()
    depth_values = last_depth_fraction.detach().cpu().tolist()
    raise CameraQualityError(
        "Camera remained blank/stale after "
        f"{max_attempts} reads; unresolved envs={invalid_ids}, "
        f"last_rgb_std={std_values}, last_rgb_range={range_values}, "
        f"last_depth_fraction={depth_values}. "
        "No episodes from this batch were written."
    )


def _advance_camera_warmup(
    env: ManagerBasedRLEnv,
    hold_action: torch.Tensor,
    *,
    maximum_control_steps: int,
    required_consecutive_frames: int,
    minimum_rgb_std: float,
    minimum_rgb_range: float,
    minimum_depth_fraction: float,
) -> dict[str, float | int]:
    """Advance unrecorded simulation steps until RTX output is stably valid."""

    consecutive = 0
    last_event: dict[str, float | int] | None = None
    last_error: CameraQualityError | None = None
    for warmup_step in range(1, maximum_control_steps + 1):
        env.step(hold_action)
        try:
            _, last_event = _read_camera_snapshot_until_ready(
                env,
                max_attempts=1,
                minimum_rgb_std=minimum_rgb_std,
                minimum_rgb_range=minimum_rgb_range,
                minimum_depth_fraction=minimum_depth_fraction,
                minimum_attempts=1,
                render_before_first=False,
            )
            consecutive += 1
        except CameraQualityError as exc:
            last_error = exc
            consecutive = 0
        if consecutive >= required_consecutive_frames:
            # Warm-up steps are outside the recorded episode horizon.
            env.episode_length_buf.zero_()
            assert last_event is not None
            return {
                **last_event,
                "warmup_control_steps": warmup_step,
                "warmup_consecutive_frames": consecutive,
            }

    detail = f" Last check: {last_error}" if last_error is not None else ""
    raise CameraQualityError(
        "Camera did not produce "
        f"{required_consecutive_frames} consecutive valid frames within "
        f"{maximum_control_steps} unrecorded control steps.{detail}"
    )


def _collect_batch(
    env: ManagerBasedRLEnv,
    env_cfg: PickLiftEnvCfg,
    max_steps: int,
    randomizer: DomainRandomizer | None,
    camera_frame_interval: int,
    camera_ready_attempts: int,
    camera_min_rgb_std: float,
    camera_min_rgb_range: float,
    camera_min_depth_fraction: float,
    camera_warmup_steps: int,
    camera_warmup_consecutive_frames: int,
) -> tuple[
    list[
        tuple[
            EpisodeBuffer,
            dict[str, object],
            dict[str, object] | None,
            CameraFrameBuffer | None,
        ]
    ],
    dict[str, float | int],
]:
    """Collect one synchronized vectorized batch and split it into episodes."""

    env.reset()
    domain_params = (
        randomizer.apply_batch(env, randomizer.sample_batch(env.num_envs))
        if randomizer is not None
        else None
    )
    camera_enabled = "table_camera" in env.scene.sensors
    if camera_enabled and randomizer is None:
        camera = env.scene["table_camera"]
        eye = torch.tensor((1.60, -2.00, 1.65), device=env.device).repeat(env.num_envs, 1)
        target = torch.tensor((0.25, 0.0, 0.85), device=env.device).repeat(env.num_envs, 1)
        camera.set_world_poses_from_view(env.scene.env_origins + eye, env.scene.env_origins + target)
    if camera_enabled:
        camera = env.scene["table_camera"]
        env.sim.forward()
        camera.reset()
    state_machine = PickAndLiftStateMachine(
        dt=env.step_dt, num_envs=env.num_envs, device=env.device
    )
    ee_pose_b, _ = _task_poses(env)
    actions = torch.cat(
        (
            ee_pose_b,
            torch.full((env.num_envs, 1), state_machine.OPEN_GRIPPER, device=env.device),
        ),
        dim=-1,
    )
    camera_quality_events: list[dict[str, float | int]] = []
    if camera_enabled:
        warmup_event = _advance_camera_warmup(
            env,
            actions,
            maximum_control_steps=camera_warmup_steps,
            required_consecutive_frames=camera_warmup_consecutive_frames,
            minimum_rgb_std=camera_min_rgb_std,
            minimum_rgb_range=camera_min_rgb_range,
            minimum_depth_fraction=camera_min_depth_fraction,
        )
        camera_quality_events.append(warmup_event)
    # Re-read the hold pose after the unrecorded settling period.
    ee_pose_b, _ = _task_poses(env)
    actions = torch.cat(
        (
            ee_pose_b,
            torch.full((env.num_envs, 1), state_machine.OPEN_GRIPPER, device=env.device),
        ),
        dim=-1,
    )
    buffers = [EpisodeBuffer() for _ in range(env.num_envs)]
    camera_buffers = (
        [CameraFrameBuffer() for _ in range(env.num_envs)] if camera_enabled else [None] * env.num_envs
    )
    success_streak = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    failure_reasons = ["max_steps"] * env.num_envs
    completion_steps = torch.full(
        (env.num_envs,), max_steps, dtype=torch.int64, device=env.device
    )
    initial_object_height = env.scene["object"].data.root_pos_w[:, 2].clone()
    maximum_object_height = initial_object_height.clone()
    final_object_height = initial_object_height.clone()
    vector_steps = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize(env.device)
    rollout_start = time.perf_counter()

    # ``no_grad`` keeps multi-episode reset buffers mutable.  Using
    # ``inference_mode`` here would mark Isaac Lab's internal tensors as
    # inference tensors and make the next episode reset fail in-place.
    with torch.no_grad():
        for step in range(1, max_steps + 1):
            vector_steps = step
            active = ~done
            applied_action = actions.clone()
            applied_controller_state = state_machine.state.clone()
            _, reward, terminated, truncated, _ = env.step(applied_action)

            ee_pose_b, object_pose_b = _task_poses(env)
            lift_position_b = env.command_manager.get_command("object_pose")[:, :3]
            lift_pose_b = torch.cat(
                (
                    lift_position_b,
                    torch.tensor((0.0, 1.0, 0.0, 0.0), device=env.device).repeat(
                        env.num_envs, 1
                    ),
                ),
                dim=-1,
            )
            actions = state_machine.compute(ee_pose_b, object_pose_b, lift_pose_b)

            robot = env.scene["robot"]
            object_position_w = env.scene["object"].data.root_pos_w
            ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            finger_position = robot.data.joint_pos[:, -2:]
            ee_object_distance = torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1)

            maximum_object_height = torch.where(
                active,
                torch.maximum(maximum_object_height, object_position_w[:, 2]),
                maximum_object_height,
            )
            current_success = (
                (object_position_w[:, 2] >= TABLE_TOP_Z + 0.10)
                & (finger_position.mean(dim=-1) < 0.030)
                & (ee_object_distance < 0.08)
            )
            next_streak = torch.where(
                current_success, success_streak + 1, torch.zeros_like(success_streak)
            )
            success_streak = torch.where(active, next_streak, success_streak)
            newly_successful = active & (success_streak >= 10)
            newly_terminated = active & terminated & ~newly_successful
            newly_truncated = active & truncated & ~newly_successful & ~newly_terminated
            newly_done = newly_successful | newly_terminated | newly_truncated

            # Transfer each batched tensor once, then distribute its rows to
            # per-environment buffers. This avoids one GPU synchronization per
            # tensor per environment.
            active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1).cpu().tolist()
            q_cpu = _cpu_numpy(robot.data.joint_pos)
            dq_cpu = _cpu_numpy(robot.data.joint_vel)
            ee_cpu = _cpu_numpy(ee_pose_b)
            object_cpu = _cpu_numpy(object_pose_b)
            action_cpu = _cpu_numpy(applied_action)
            reward_cpu = _cpu_numpy(reward)
            state_cpu = _cpu_numpy(applied_controller_state)
            for env_index in active_indices:
                buffers[env_index].append(
                    q=q_cpu[env_index],
                    dq=dq_cpu[env_index],
                    eef_pose=ee_cpu[env_index],
                    object_pose=object_cpu[env_index],
                    action=action_cpu[env_index],
                    reward=float(reward_cpu[env_index]),
                    timestamp=step * env.step_dt,
                    controller_state=int(state_cpu[env_index]),
                )

            capture_camera = camera_enabled and (
                step == 1 or step % camera_frame_interval == 0 or bool(newly_done.any())
            )
            if capture_camera:
                camera_snapshot, camera_event = _read_camera_snapshot_until_ready(
                    env,
                    max_attempts=camera_ready_attempts,
                    minimum_rgb_std=camera_min_rgb_std,
                    minimum_rgb_range=camera_min_rgb_range,
                    minimum_depth_fraction=camera_min_depth_fraction,
                    minimum_attempts=1,
                    render_before_first=False,
                )
                if int(camera_event["forced_renders"]) > 0:
                    camera_quality_events.append(camera_event)
                rgb_cpu = _cpu_numpy(camera_snapshot["rgb"])
                depth_cpu = _cpu_numpy(camera_snapshot["depth"])
                intrinsics_cpu = _cpu_numpy(camera_snapshot["intrinsics"])
                camera_pose_cpu = _cpu_numpy(camera_snapshot["camera_pose"])
                object_pose_w_cpu = _cpu_numpy(
                    torch.cat(
                        (
                            env.scene["object"].data.root_pos_w,
                            env.scene["object"].data.root_quat_w,
                        ),
                        dim=-1,
                    )
                )
                eef_pose_w_cpu = _cpu_numpy(
                    torch.cat(
                        (
                            env.scene["ee_frame"].data.target_pos_w[:, 0, :],
                            env.scene["ee_frame"].data.target_quat_w[:, 0, :],
                        ),
                        dim=-1,
                    )
                )
                for env_index in active_indices:
                    camera_buffers[env_index].append(
                        frame_step=step,
                        timestamp=step * env.step_dt,
                        rgb=rgb_cpu[env_index],
                        depth=depth_cpu[env_index],
                        camera_intrinsics=intrinsics_cpu[env_index],
                        camera_pose_w=camera_pose_cpu[env_index],
                        object_pose_w=object_pose_w_cpu[env_index],
                        eef_pose_w=eef_pose_w_cpu[env_index],
                    )

            if bool(newly_done.any()):
                completion_steps[newly_done] = step
                final_object_height[newly_done] = object_position_w[newly_done, 2]
                for env_index in torch.nonzero(newly_successful, as_tuple=False).squeeze(-1).cpu().tolist():
                    failure_reasons[env_index] = "none"
                for env_index in torch.nonzero(newly_terminated, as_tuple=False).squeeze(-1).cpu().tolist():
                    failure_reasons[env_index] = "terminated"
                for env_index in torch.nonzero(newly_truncated, as_tuple=False).squeeze(-1).cpu().tolist():
                    failure_reasons[env_index] = "timeout"
                success |= newly_successful
                done |= newly_done

            # Completed environments are held at their current pose while the
            # remaining environments finish; their trajectory buffers stop.
            actions[done, :7] = ee_pose_b[done]
            actions[done, 7] = state_machine.OPEN_GRIPPER
            if bool(done.all()):
                break

    remaining = ~done
    final_object_height[remaining] = env.scene["object"].data.root_pos_w[remaining, 2]
    if torch.cuda.is_available():
        torch.cuda.synchronize(env.device)
    rollout_wall_time_s = time.perf_counter() - rollout_start

    initial_height_cpu = initial_object_height.detach().cpu().tolist()
    maximum_height_cpu = maximum_object_height.detach().cpu().tolist()
    final_height_cpu = final_object_height.detach().cpu().tolist()
    success_cpu = success.detach().cpu().tolist()
    completion_steps_cpu = completion_steps.detach().cpu().tolist()
    records: list[
        tuple[
            EpisodeBuffer,
            dict[str, object],
            dict[str, object] | None,
            CameraFrameBuffer | None,
        ]
    ] = []
    for env_index, buffer in enumerate(buffers):
        arrays = buffer.as_arrays()
        metadata: dict[str, object] = {
            "task": "Franka-Pick-and-Lift-v0",
            "controller": "scripted_absolute_differential_ik",
            "seed": env_cfg.seed,
            "parallel_env_index": env_index,
            "parallel_num_envs": env.num_envs,
            "success": bool(success_cpu[env_index]),
            "failure_reason": failure_reasons[env_index],
            "steps": len(buffer),
            "physics_dt": env.physics_dt,
            "control_dt": env.step_dt,
            "completion_time_s": float(completion_steps_cpu[env_index] * env.step_dt),
            "total_reward": float(arrays["reward"].sum(dtype="float64")),
            "initial_object_height": float(initial_height_cpu[env_index]),
            "maximum_object_height": float(maximum_height_cpu[env_index]),
            "final_object_height": float(final_height_cpu[env_index]),
            "success_criteria": {
                "minimum_height_above_table_m": 0.10,
                "maximum_mean_finger_position": 0.030,
                "maximum_ee_object_distance_m": 0.08,
                "sustained_steps": 10,
            },
            "coordinate_conventions": {
                "pose_order": "x, y, z, qw, qx, qy, qz",
                "eef_pose_frame": "robot_root",
                "object_pose_frame": "robot_root",
            },
            "camera_data": camera_enabled,
            "camera_frame_interval": camera_frame_interval if camera_enabled else None,
            "camera_quality_gate": (
                {
                    "minimum_rgb_std": camera_min_rgb_std,
                    "minimum_rgb_range": camera_min_rgb_range,
                    "minimum_depth_fraction": camera_min_depth_fraction,
                    "maximum_refresh_attempts": camera_ready_attempts,
                    "maximum_warmup_control_steps": camera_warmup_steps,
                    "required_consecutive_warmup_frames": camera_warmup_consecutive_frames,
                }
                if camera_enabled
                else None
            ),
            "camera_coordinate_conventions": (
                {
                    "camera_pose_w": "ROS optical camera frame in world; xyz + quaternion wxyz",
                    "camera_extrinsics_w2c": "world-to-camera homogeneous transform",
                    "object_pose_camera": "object pose in ROS optical camera frame",
                    "depth": "distance_to_image_plane in meters; +inf denotes no hit",
                }
                if camera_enabled
                else None
            ),
            "domain_randomization": domain_params is not None,
            "domain_params_file": "domain_params.json" if domain_params is not None else None,
        }
        records.append(
            (
                buffer,
                metadata,
                domain_params[env_index] if domain_params is not None else None,
                camera_buffers[env_index],
            )
        )

    environment_transitions = sum(len(buffer) for buffer in buffers)
    metrics: dict[str, float | int] = {
        "vector_steps": vector_steps,
        "environment_transitions": environment_transitions,
        "rollout_wall_time_s": rollout_wall_time_s,
        "simulation_fps": vector_steps / rollout_wall_time_s,
        "environment_steps_per_second": environment_transitions / rollout_wall_time_s,
        "episodes_per_minute": env.num_envs * 60.0 / rollout_wall_time_s,
        "camera_refresh_events": len(camera_quality_events),
        "camera_refresh_attempts_total": sum(
            int(item["forced_renders"]) for item in camera_quality_events
        ),
        "camera_rgb_std_min": min(
            (float(item["minimum_rgb_std"]) for item in camera_quality_events),
            default=0.0,
        ),
        "camera_rgb_range_min": min(
            (float(item["minimum_rgb_range"]) for item in camera_quality_events),
            default=0.0,
        ),
        "camera_depth_fraction_min": min(
            (float(item["minimum_depth_fraction"]) for item in camera_quality_events),
            default=0.0,
        ),
        "camera_warmup_control_steps": max(
            (int(item.get("warmup_control_steps", 0)) for item in camera_quality_events),
            default=0,
        ),
    }
    return records, metrics


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1")
    if args_cli.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args_cli.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if args_cli.camera_frame_interval < 1:
        raise ValueError("--camera_frame_interval must be at least 1")
    if args_cli.camera_ready_attempts < 1:
        raise ValueError("--camera_ready_attempts must be at least 1")
    if args_cli.camera_min_rgb_std < 0.0 or args_cli.camera_min_rgb_range < 0.0:
        raise ValueError("Camera quality thresholds must be non-negative")
    if not 0.0 <= args_cli.camera_min_depth_fraction <= 1.0:
        raise ValueError("--camera_min_depth_fraction must be within [0, 1]")
    if args_cli.camera_warmup_steps < 1:
        raise ValueError("--camera_warmup_steps must be at least 1")
    if not 1 <= args_cli.camera_warmup_consecutive_frames <= args_cli.camera_warmup_steps:
        raise ValueError(
            "--camera_warmup_consecutive_frames must be within [1, camera_warmup_steps]"
        )
    if args_cli.camera_batch_retries < 0:
        raise ValueError("--camera_batch_retries must be non-negative")

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.enable_cameras:
        # Avoid reading Replicator annotators during physics-only decimation
        # substeps. Camera data is checked explicitly after the rendered step.
        env_cfg.scene.lazy_sensor_update = True
    physx_aggregate_pairs_capacity = configure_parallel_physx_capacity(
        env_cfg, args_cli.num_envs
    )
    if not args_cli.enable_cameras:
        env_cfg.scene.table_camera = None
        env_cfg.scene.robot.spawn.semantic_tags = None
        env_cfg.scene.object.spawn.semantic_tags = None
        env_cfg.scene.table.spawn.semantic_tags = None
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = (args_cli.max_steps + 10) * env_cfg.sim.dt * env_cfg.decimation
    env = ManagerBasedRLEnv(cfg=env_cfg)
    writer = EpisodeWriter(args_cli.output_dir)
    randomizer = (
        create_domain_randomizer(args_cli.randomization_config)
        if args_cli.randomization_config
        else None
    )
    collected: list[dict[str, object]] = []
    batch_metrics: list[dict[str, float | int]] = []

    try:
        for batch_index in range(args_cli.episodes):
            for retry_index in range(args_cli.camera_batch_retries + 1):
                try:
                    records, metrics = _collect_batch(
                        env,
                        env_cfg,
                        args_cli.max_steps,
                        randomizer,
                        args_cli.camera_frame_interval,
                        args_cli.camera_ready_attempts,
                        args_cli.camera_min_rgb_std,
                        args_cli.camera_min_rgb_range,
                        args_cli.camera_min_depth_fraction,
                        args_cli.camera_warmup_steps,
                        args_cli.camera_warmup_consecutive_frames,
                    )
                    metrics["camera_batch_retry_index"] = retry_index
                    break
                except CameraQualityError as exc:
                    if retry_index >= args_cli.camera_batch_retries:
                        raise
                    print(
                        "CAMERA_BATCH_RETRY="
                        + json.dumps(
                            {
                                "batch_index": batch_index,
                                "retry_index": retry_index + 1,
                                "reason": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            batch_metrics.append(metrics)
            next_episode_id = writer.next_episode_id()
            for env_index, (buffer, metadata, domain_params, camera_buffer) in enumerate(records):
                episode_id = next_episode_id + env_index
                metadata["parallel_batch_index"] = batch_index
                sidecars = {"domain_params.json": domain_params} if domain_params is not None else None
                episode_dir = writer.write(
                    episode_id,
                    buffer,
                    metadata,
                    sidecar_json=sidecars,
                    camera_buffer=camera_buffer,
                )
                validation = validate_episode(episode_dir)
                collected.append(
                    {
                        "episode_id": episode_id,
                        "path": str(episode_dir),
                        "success": metadata["success"],
                        "steps": metadata["steps"],
                        "parallel_batch_index": batch_index,
                        "parallel_env_index": env_index,
                        "camera_frames": (
                            validation["camera_tensor_shapes"]["rgb"][0]
                            if validation["camera_data"]
                            else 0
                        ),
                        "validation": validation["status"],
                    }
                )
    finally:
        env.close()

    total_rollout_time = sum(float(item["rollout_wall_time_s"]) for item in batch_metrics)
    total_vector_steps = sum(int(item["vector_steps"]) for item in batch_metrics)
    total_transitions = sum(int(item["environment_transitions"]) for item in batch_metrics)
    phase = (
        "PHASE10"
        if args_cli.enable_cameras
        else ("PHASE5" if args_cli.num_envs > 1 else ("PHASE4" if randomizer is not None else "PHASE3"))
    )
    episode_details_truncated = len(collected) > 16
    episode_preview = collected if not episode_details_truncated else collected[:3] + collected[-3:]
    summary = {
        "status": f"{phase}_COLLECTION_OK",
        "num_envs": args_cli.num_envs,
        "batches_requested": args_cli.episodes,
        "episodes_requested": args_cli.episodes * args_cli.num_envs,
        "episodes_written": len(collected),
        "num_success": sum(bool(item["success"]) for item in collected),
        "num_failure": sum(not bool(item["success"]) for item in collected),
        "domain_randomization": randomizer is not None,
        "randomization_seed": randomizer.seed if randomizer is not None else None,
        "camera_data": bool(args_cli.enable_cameras),
        "camera_frame_interval": args_cli.camera_frame_interval if args_cli.enable_cameras else None,
        "camera_quality_gate": (
            {
                "minimum_rgb_std": args_cli.camera_min_rgb_std,
                "minimum_rgb_range": args_cli.camera_min_rgb_range,
                "minimum_depth_fraction": args_cli.camera_min_depth_fraction,
                "maximum_refresh_attempts": args_cli.camera_ready_attempts,
                "maximum_warmup_control_steps": args_cli.camera_warmup_steps,
                "required_consecutive_warmup_frames": args_cli.camera_warmup_consecutive_frames,
                "maximum_batch_retries": args_cli.camera_batch_retries,
            }
            if args_cli.enable_cameras
            else None
        ),
        "camera_frames_written": (
            sum(int(item["camera_frames"]) for item in collected)
            if args_cli.enable_cameras
            else 0
        ),
        "physx_gpu_total_aggregate_pairs_capacity": physx_aggregate_pairs_capacity,
        "camera_batch_retries_total": sum(
            int(item.get("camera_batch_retry_index", 0)) for item in batch_metrics
        ),
        "rollout_wall_time_s": total_rollout_time,
        "simulation_fps": total_vector_steps / total_rollout_time,
        "environment_steps_per_second": total_transitions / total_rollout_time,
        "episodes_per_minute": len(collected) * 60.0 / total_rollout_time,
        "output_dir": str(args_cli.output_dir.resolve()),
        "batches": batch_metrics,
        "episode_details_truncated": episode_details_truncated,
        "episode_preview": episode_preview,
    }
    print(f"{phase}_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    sys.exit(exit_code)
