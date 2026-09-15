"""Run the deterministic Phase 2 scripted Pick-and-Lift baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the Phase 2 scripted Franka Pick-and-Lift task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_steps", type=int, default=600, help="Maximum manager-environment steps.")
parser.add_argument("--save_camera", action="store_true", help="Save the final RGB camera frame.")
parser.add_argument(
    "--realtime",
    action="store_true",
    help="Throttle the loop to simulation time so the motion is easy to watch in the GUI.",
)
parser.add_argument(
    "--realtime_speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier used with --realtime (for example, 0.5 is half speed).",
)
parser.add_argument(
    "--keep_open",
    action="store_true",
    help="After success, keep holding the cube until the visible Isaac Sim window is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Headless validation and image saving need the RGB-D sensor.  A visible
# debugging window only needs the viewport, so keep camera/Replicator
# extensions disabled unless the user explicitly requests them.
args_cli.enable_cameras = bool(args_cli.enable_cameras or args_cli.headless or args_cli.save_camera)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import torch
    from PIL import Image

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.controllers import PickAndLiftStateMachine
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        configure_parallel_physx_capacity,
    )
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


def _pose_in_robot_frame(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    eef_sensor = env.scene["ee_frame"]
    target_object = env.scene["object"]

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        eef_sensor.data.target_pos_w[:, 0, :],
        eef_sensor.data.target_quat_w[:, 0, :],
    )
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_object.data.root_pos_w,
        target_object.data.root_quat_w,
    )
    return torch.cat((ee_pos_b, ee_quat_b), dim=-1), torch.cat((object_pos_b, object_quat_b), dim=-1)


def _set_camera_pose(env: ManagerBasedRLEnv) -> None:
    camera = env.scene["table_camera"]
    origins = env.scene.env_origins
    eye = torch.tensor((1.60, -2.00, 1.65), device=env.device).repeat(env.num_envs, 1)
    target = torch.tensor((0.25, 0.0, 0.85), device=env.device).repeat(env.num_envs, 1)
    camera.set_world_poses_from_view(origins + eye, origins + target)


def _refresh_camera(env: ManagerBasedRLEnv) -> None:
    """Force a few RTX updates before reading a debug frame."""

    camera = env.scene["table_camera"]
    for _ in range(3):
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)


def _save_final_rgb(env: ManagerBasedRLEnv) -> str:
    output_dir = Path(__file__).resolve().parents[1] / "results" / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pick_lift_final_rgb.png"
    _refresh_camera(env)
    rgb = env.scene["table_camera"].data.output["rgb"][0].detach().cpu().numpy()
    Image.fromarray(rgb).save(output_path)
    return str(output_path)


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1")
    if args_cli.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if args_cli.realtime_speed <= 0.0:
        raise ValueError("--realtime_speed must be greater than zero")

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    configure_parallel_physx_capacity(env_cfg, args_cli.num_envs)
    env_cfg.sim.device = args_cli.device
    if not args_cli.enable_cameras:
        env_cfg.scene.table_camera = None
    if args_cli.keep_open:
        env_cfg.episode_length_s = 24.0 * 60.0 * 60.0
    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()
        if args_cli.enable_cameras:
            _set_camera_pose(env)

        state_machine = PickAndLiftStateMachine(
            dt=env_cfg.sim.dt * env_cfg.decimation,
            num_envs=env.num_envs,
            device=env.device,
        )
        actions = torch.zeros((env.num_envs, 8), device=env.device)
        actions[:, 3] = 1.0

        initial_object_height = env.scene["object"].data.root_pos_w[:, 2].clone()
        maximum_object_height = initial_object_height.clone()
        minimum_ee_object_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
        success_streak = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        success_step = -1
        step_count = 0

        with torch.inference_mode():
            for step_count in range(1, args_cli.max_steps + 1):
                wall_step_start = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(actions)

                ee_pose_b, object_pose_b = _pose_in_robot_frame(env)
                lift_position_b = env.command_manager.get_command("object_pose")[:, :3]
                lift_pose_b = torch.cat(
                    (
                        lift_position_b,
                        torch.tensor((0.0, 1.0, 0.0, 0.0), device=env.device).repeat(env.num_envs, 1),
                    ),
                    dim=-1,
                )
                actions = state_machine.compute(ee_pose_b, object_pose_b, lift_pose_b)

                object_position_w = env.scene["object"].data.root_pos_w
                ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
                finger_position = env.scene["robot"].data.joint_pos[:, -2:]
                ee_object_distance = torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1)

                maximum_object_height = torch.maximum(maximum_object_height, object_position_w[:, 2])
                minimum_ee_object_distance = torch.minimum(minimum_ee_object_distance, ee_object_distance)

                lifted = object_position_w[:, 2] >= TABLE_TOP_Z + 0.10
                gripper_closed = finger_position.mean(dim=-1) < 0.030
                object_held = ee_object_distance < 0.08
                current_success = lifted & gripper_closed & object_held
                success_streak = torch.where(current_success, success_streak + 1, torch.zeros_like(success_streak))
                success = success_streak >= 10

                if success.all():
                    success_step = step_count
                    break
                if (terminated | truncated).any():
                    break

                if args_cli.realtime:
                    target_wall_dt = env.step_dt / args_cli.realtime_speed
                    remaining = target_wall_dt - (time.perf_counter() - wall_step_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

            if bool(success.all()) and args_cli.keep_open:
                print(
                    "PHASE2_GUI_HOLD=success; close the Isaac Sim window to exit.",
                    flush=True,
                )
                while simulation_app.is_running():
                    wall_step_start = time.perf_counter()
                    env.step(actions)
                    if args_cli.realtime:
                        target_wall_dt = env.step_dt / args_cli.realtime_speed
                        remaining = target_wall_dt - (time.perf_counter() - wall_step_start)
                        if remaining > 0.0:
                            time.sleep(remaining)

        final_object_position = env.scene["object"].data.root_pos_w.clone()
        final_ee_position = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
        final_finger_position = env.scene["robot"].data.joint_pos[:, -2:].clone()
        final_distance = torch.linalg.vector_norm(final_ee_position - final_object_position, dim=-1)

        policy_observation = env.observation_manager.compute_group("policy")
        summary: dict[str, object] = {
            "status": "PHASE2_SUCCESS" if bool(success.all()) else "PHASE2_FAILED",
            "num_envs": env.num_envs,
            "steps": step_count,
            "success_step": success_step,
            "success": success.detach().cpu().tolist(),
            "state": state_machine.state.detach().cpu().tolist(),
            "action_shape": list(actions.shape),
            "policy_observation_shape": list(policy_observation.shape),
            "initial_object_height": initial_object_height.detach().cpu().tolist(),
            "maximum_object_height": maximum_object_height.detach().cpu().tolist(),
            "final_object_position": final_object_position.detach().cpu().tolist(),
            "final_end_effector_position": final_ee_position.detach().cpu().tolist(),
            "final_ee_object_distance": final_distance.detach().cpu().tolist(),
            "minimum_ee_object_distance": minimum_ee_object_distance.detach().cpu().tolist(),
            "final_finger_joint_position": final_finger_position.detach().cpu().tolist(),
        }
        if args_cli.enable_cameras:
            _refresh_camera(env)
            camera = env.scene["table_camera"]
            rgb = camera.data.output["rgb"]
            depth = camera.data.output["distance_to_image_plane"]
            summary["camera_position"] = camera.data.pos_w[0].detach().cpu().tolist()
            summary["camera_rgb_mean"] = float(rgb.float().mean().item())
            summary["camera_rgb_range"] = [int(rgb.min().item()), int(rgb.max().item())]
            summary["camera_finite_depth_fraction"] = float(torch.isfinite(depth).float().mean().item())
        if args_cli.save_camera:
            summary["camera_file"] = _save_final_rgb(env)

        print("PHASE2_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
        if not bool(success.all()):
            raise RuntimeError("Pick-and-Lift did not satisfy the sustained success criteria.")
    finally:
        env.close()


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
