"""Run closed-loop pick-and-lift using RGB-D object position instead of ground truth."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results/vision/rgbd_keypoint_depth_residual_hard_v2/best_model.pt")
parser.add_argument("--randomization-config", type=Path, default=PROJECT_ROOT / "configs/randomization_hard.yaml")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument("--camera-interval", type=int, default=5, help="Run RGB-D inference every N control steps.")
parser.add_argument("--ema-alpha", type=float, default=0.45)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--realtime-speed", type=float, default=1.0)
parser.add_argument("--keep-open", action="store_true")
parser.add_argument(
    "--output-summary",
    type=Path,
    default=PROJECT_ROOT / "results/vision/visual_control_smoke_summary.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.controllers import PickAndLiftState, PickAndLiftStateMachine
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import PickLiftEnvCfg, configure_parallel_physx_capacity
    from isaac_lab_data_engine.randomization import create_domain_randomizer
    from isaac_lab_data_engine.vision import (
        RGBDPoseEstimator,
        VisualPolicyObservationAdapter,
        camera_to_world_position,
        world_to_root_position,
    )
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


def _poses_in_robot_frame(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    target_object = env.scene["object"]
    ee_position, ee_quaternion = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
    )
    object_position, object_quaternion = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_object.data.root_pos_w,
        target_object.data.root_quat_w,
    )
    return (
        torch.cat((ee_position, ee_quaternion), dim=-1),
        torch.cat((object_position, object_quaternion), dim=-1),
    )


def _refresh_camera(env: ManagerBasedRLEnv) -> None:
    camera = env.scene["table_camera"]
    for _ in range(3):
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)


def _object_workspace_pixel_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select depth pixels lying on the configured tabletop object workspace."""

    camera = env.scene["table_camera"]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    intrinsics = camera.data.intrinsic_matrices
    batch, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=env.device, dtype=depth.dtype),
        torch.arange(width, device=env.device, dtype=depth.dtype),
        indexing="ij",
    )
    x_camera = (x[None] - intrinsics[:, 0, 2, None, None]) / intrinsics[:, 0, 0, None, None] * depth
    y_camera = (y[None] - intrinsics[:, 1, 2, None, None]) / intrinsics[:, 1, 1, None, None] * depth
    position_camera = torch.stack((x_camera, y_camera, depth), dim=-1)
    position_w = camera_to_world_position(
        position_camera,
        camera.data.pos_w[:, None, None, :],
        camera.data.quat_w_ros[:, None, None, :],
    )
    robot = env.scene["robot"]
    position_b = world_to_root_position(
        position_w,
        robot.data.root_pos_w[:, None, None, :],
        robot.data.root_quat_w[:, None, None, :],
    )
    return (
        torch.isfinite(depth)
        & (position_b[..., 0] >= 0.33)
        & (position_b[..., 0] <= 0.81)
        & (position_b[..., 1] >= -0.33)
        & (position_b[..., 1] <= 0.33)
        & (position_b[..., 2] >= -0.015)
        & (position_b[..., 2] <= 0.09)
    )


@torch.inference_mode()
def _visual_position_b(
    env: ManagerBasedRLEnv, estimator: RGBDPoseEstimator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camera = env.scene["table_camera"]
    position_camera = estimator.predict(
        camera.data.output["rgb"],
        camera.data.output["distance_to_image_plane"],
        camera.data.intrinsic_matrices,
        valid_pixel_mask=_object_workspace_pixel_mask(env),
    )
    position_w = camera_to_world_position(
        position_camera, camera.data.pos_w, camera.data.quat_w_ros
    )
    robot = env.scene["robot"]
    raw_position_b = world_to_root_position(
        position_w, robot.data.root_pos_w, robot.data.root_quat_w
    )
    # Reject only physically impossible outliers.  These limits are task
    # workspace constraints, not object-state information from the simulator.
    position_b = raw_position_b.clone()
    position_b[:, 0].clamp_(0.36, 0.78)
    position_b[:, 1].clamp_(-0.30, 0.30)
    # Perception is latched before motion while the 5 cm cube rests on the
    # known tabletop.  Using the known half-height prevents a one-pixel depth
    # error from commanding the gripper through the table.
    position_b[:, 2] = 0.03
    return position_b, position_camera, raw_position_b


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.max_steps < 1 or args_cli.camera_interval < 1:
        raise ValueError("num-envs, max-steps and camera-interval must be positive")
    if args_cli.realtime_speed <= 0.0:
        raise ValueError("realtime-speed must be positive")
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(args_cli.checkpoint)

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    configure_parallel_physx_capacity(env_cfg, args_cli.num_envs)
    if args_cli.keep_open:
        env_cfg.episode_length_s = 24.0 * 60.0 * 60.0
    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        randomizer = create_domain_randomizer(args_cli.randomization_config)
        domain_params = randomizer.apply_batch(env, randomizer.sample_batch(env.num_envs))
        _refresh_camera(env)

        estimator = RGBDPoseEstimator.load(args_cli.checkpoint, env.device)
        print(
            "PHASE12_VISUAL_MODEL="
            + json.dumps(
                {
                    "checkpoint": str(args_cli.checkpoint.resolve()),
                    "model": type(estimator.model).__name__,
                    "input": "table_camera RGB + distance_to_image_plane + intrinsics",
                    "controller_object_source": "visual_prediction_only",
                    "simulator_object_truth": "metrics_only",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        adapter = VisualPolicyObservationAdapter(ema_alpha=args_cli.ema_alpha)
        controller = PickAndLiftStateMachine(env.step_dt, env.num_envs, env.device)
        ee_pose_b, true_object_pose_b = _poses_in_robot_frame(env)
        raw_visual_position, initial_position_camera, initial_position_b_unclamped = _visual_position_b(
            env, estimator
        )
        visual_position = adapter.filter_position(raw_visual_position)
        lift_anchor = visual_position.clone()
        predicted_object_pose = torch.zeros((env.num_envs, 7), device=env.device)
        predicted_object_pose[:, :3] = visual_position
        predicted_object_pose[:, 3] = 1.0
        actions = torch.cat(
            (
                ee_pose_b,
                torch.full((env.num_envs, 1), controller.OPEN_GRIPPER, device=env.device),
            ),
            dim=-1,
        )

        prediction_errors = [
            torch.linalg.vector_norm(visual_position - true_object_pose_b[:, :3], dim=-1)
        ]
        print(
            "PHASE12_INITIAL_PERCEPTION="
            + json.dumps(
                {
                    "predicted_camera_m": initial_position_camera.cpu().tolist(),
                    "predicted_root_unclamped_m": initial_position_b_unclamped.cpu().tolist(),
                    "predicted_root_used_m": visual_position.cpu().tolist(),
                    "true_root_metrics_only_m": true_object_pose_b[:, :3].cpu().tolist(),
                    "camera_position_w_m": env.scene["table_camera"].data.pos_w.cpu().tolist(),
                    "camera_quaternion_wxyz_ros": env.scene["table_camera"].data.quat_w_ros.cpu().tolist(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        inference_times_ms: list[float] = []
        initial_height = env.scene["object"].data.root_pos_w[:, 2].clone()
        maximum_height = initial_height.clone()
        success_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        step_count = 0

        with torch.inference_mode():
            for step_count in range(1, args_cli.max_steps + 1):
                wall_start = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(actions)
                ee_pose_b, true_object_pose_b = _poses_in_robot_frame(env)

                # Fuse several unobstructed frames while resting, then latch
                # the target.  Once the arm enters the view, the wrist and
                # fingers can fully occlude this 5 cm cube; blindly updating
                # at that point caused the first closed-loop attempt to chase
                # a false target.
                resting = controller.state == int(PickAndLiftState.REST)
                if step_count % args_cli.camera_interval == 0 and bool(resting.any()):
                    inference_start = time.perf_counter()
                    raw_visual_position, _, _ = _visual_position_b(env, estimator)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(env.device)
                    inference_times_ms.append((time.perf_counter() - inference_start) * 1000.0)
                    filtered_position = adapter.filter_position(raw_visual_position)
                    visual_position[resting] = filtered_position[resting]
                    prediction_errors.append(
                        torch.linalg.vector_norm(
                            visual_position - true_object_pose_b[:, :3], dim=-1
                        )
                    )
                    lift_anchor[resting] = visual_position[resting]

                predicted_object_pose[:, :3] = visual_position
                lift_pose = predicted_object_pose.clone()
                lift_pose[:, :3] = lift_anchor
                lift_pose[:, 2] += 0.18
                actions = controller.compute(ee_pose_b, predicted_object_pose, lift_pose)

                object_position_w = env.scene["object"].data.root_pos_w
                ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
                finger_position = env.scene["robot"].data.joint_pos[:, -2:].mean(dim=-1)
                maximum_height = torch.maximum(maximum_height, object_position_w[:, 2])
                held = torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1) < 0.08
                valid = (
                    (object_position_w[:, 2] >= TABLE_TOP_Z + 0.10)
                    & (finger_position < 0.030)
                    & held
                )
                success_streak = torch.where(valid, success_streak + 1, torch.zeros_like(success_streak))
                success = success_streak >= 10
                if bool(success.all()) or bool((terminated | truncated).any()):
                    break
                if args_cli.realtime:
                    remaining = env.step_dt / args_cli.realtime_speed - (time.perf_counter() - wall_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

            if bool(success.all()) and args_cli.keep_open:
                print("VISUAL_GUI_HOLD=success; close Isaac Sim to exit.", flush=True)
                while simulation_app.is_running():
                    env.step(actions)

        errors = torch.cat(prediction_errors)
        summary = {
            "status": "PHASE12_VISUAL_CONTROL_SUCCESS" if bool(success.all()) else "PHASE12_VISUAL_CONTROL_FAILED",
            "checkpoint": str(args_cli.checkpoint.resolve()),
            "randomization_config": str(args_cli.randomization_config.resolve()),
            "num_envs": env.num_envs,
            "steps": step_count,
            "success": success.cpu().tolist(),
            "controller_state": controller.state.cpu().tolist(),
            "initial_prediction_error_cm": (prediction_errors[0] * 100.0).cpu().tolist(),
            "mean_prediction_error_cm": float(errors.mean().cpu() * 100.0),
            "p95_prediction_error_cm": float(torch.quantile(errors, 0.95).cpu() * 100.0),
            "mean_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
            "maximum_lift_cm": ((maximum_height - initial_height) * 100.0).cpu().tolist(),
            "sampled_object_positions_m": [item["object"]["position_m"] for item in domain_params],
            "ground_truth_usage": "metrics_only; controller receives RGB-D prediction and robot proprioception",
        }
        args_cli.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("PHASE12_VISUAL_CONTROL_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
        if not bool(success.all()):
            raise RuntimeError("Visual closed-loop policy did not reach sustained 10 cm lift success")
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 0
    try:
        import numpy as np

        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        simulation_app.close()
    sys.exit(exit_code)
