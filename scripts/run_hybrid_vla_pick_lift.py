"""Run SmolVLA with RGB-D validation, action safety, and Scripted IK fallback."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--vision-checkpoint",
    type=Path,
    default=PROJECT_ROOT / "results/vision/rgbd_keypoint_supplement_160x120_v3/best_model.pt",
)
parser.add_argument(
    "--randomization-config",
    type=Path,
    default=PROJECT_ROOT / "configs/randomization_hard.yaml",
)
parser.add_argument("--vla-host", default="127.0.0.1")
parser.add_argument("--vla-port", type=int, default=8765)
parser.add_argument("--vla-timeout", type=float, default=30.0)
parser.add_argument("--task", default="Pick up the red cube and lift it 10 centimeters.")
parser.add_argument("--execution-horizon", type=int, default=8)
parser.add_argument(
    "--vla-position-target-mode",
    choices=("absolute", "delta"),
    default="absolute",
    help=(
        "Interpret the VLA action's first three values as absolute xyz or as "
        "delta xyz relative to the current EEF. Delta is decoded when each "
        "queued action is executed."
    ),
)
parser.add_argument(
    "--action-repeat",
    type=int,
    default=5,
    help="Hold every 10 Hz VLA action for this many 50 Hz Isaac control steps.",
)
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--realtime-speed", type=float, default=1.0)
parser.add_argument("--keep-open", action="store_true")
parser.add_argument("--require-vla", action="store_true")
parser.add_argument(
    "--condition-vla-on-visual-pose",
    action="store_true",
    help=(
        "Append the RGB-D estimated object xyz and validity to VLA state. "
        "Required for checkpoints trained with 29-D visual-conditioned state."
    ),
)
parser.add_argument(
    "--condition-vla-on-controller-phase",
    action="store_true",
    help=(
        "Append the pre-action Scripted IK FSM phase to the 29-D visual state, "
        "producing the 30-D state used by phase-conditioned checkpoints."
    ),
)
parser.add_argument(
    "--output-summary",
    type=Path,
    default=PROJECT_ROOT / "results/vla/hybrid_vla_smoke_summary.json",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if (
    args_cli.condition_vla_on_controller_phase
    and not args_cli.condition_vla_on_visual_pose
):
    parser.error(
        "--condition-vla-on-controller-phase requires --condition-vla-on-visual-pose"
    )
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.controllers import PickAndLiftState, PickAndLiftStateMachine
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.randomization import create_domain_randomizer
    from isaac_lab_data_engine.vision import (
        RGBDPoseEstimator,
        VisualPolicyObservationAdapter,
        camera_to_world_position,
        world_to_root_position,
    )
    from isaac_lab_data_engine.vla import (
        HybridActionGuard,
        VLAClient,
        VLARequest,
        decode_vla_position_target,
        pack_proprioceptive_state,
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


def _workspace_pixel_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    camera = env.scene["table_camera"]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    intrinsics = camera.data.intrinsic_matrices
    _, height, width = depth.shape
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
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = env.scene["table_camera"]
    position_camera = estimator.predict(
        camera.data.output["rgb"],
        camera.data.output["distance_to_image_plane"],
        camera.data.intrinsic_matrices,
        valid_pixel_mask=_workspace_pixel_mask(env),
    )
    position_w = camera_to_world_position(
        position_camera, camera.data.pos_w, camera.data.quat_w_ros
    )
    robot = env.scene["robot"]
    raw_position_b = world_to_root_position(
        position_w, robot.data.root_pos_w, robot.data.root_quat_w
    )
    position_b = raw_position_b.clone()
    position_b[:, 0].clamp_(0.36, 0.78)
    position_b[:, 1].clamp_(-0.30, 0.30)
    position_b[:, 2] = 0.03
    valid = (
        torch.isfinite(raw_position_b).all(dim=-1)
        & (raw_position_b[:, 0] >= 0.32)
        & (raw_position_b[:, 0] <= 0.82)
        & (raw_position_b[:, 1].abs() <= 0.35)
        & (raw_position_b[:, 2] >= -0.03)
        & (raw_position_b[:, 2] <= 0.15)
    )
    return position_b, valid


def _vla_request(
    env: ManagerBasedRLEnv,
    ee_pose_b: torch.Tensor,
    task: str,
    visual_position_b: torch.Tensor,
    perception_valid: torch.Tensor,
    controller_phase: int | None = None,
) -> VLARequest:
    robot = env.scene["robot"]
    camera_rgb = env.scene["table_camera"].data.output["rgb"][0, ..., :3]
    state_args = (
        (
            visual_position_b[0].detach().cpu().numpy(),
            bool(perception_valid[0]),
        )
        if args_cli.condition_vla_on_visual_pose
        else (None, None)
    )
    state_kwargs = {}
    if args_cli.condition_vla_on_controller_phase:
        if controller_phase is None:
            raise ValueError("controller_phase is required for 30-D VLA conditioning")
        state_kwargs["controller_phase"] = float(controller_phase)
    state = pack_proprioceptive_state(
        robot.data.joint_pos[0].detach().cpu().numpy(),
        robot.data.joint_vel[0].detach().cpu().numpy(),
        ee_pose_b[0].detach().cpu().numpy(),
        *state_args,
        **state_kwargs,
    )
    return VLARequest(
        rgb=camera_rgb.detach().cpu().numpy(),
        state=state,
        task=task,
    )


def main() -> None:
    if args_cli.execution_horizon < 1 or args_cli.action_repeat < 1 or args_cli.max_steps < 1:
        raise ValueError("execution-horizon, action-repeat and max-steps must be positive")
    if args_cli.realtime_speed <= 0.0:
        raise ValueError("realtime-speed must be positive")
    if not args_cli.vision_checkpoint.is_file():
        raise FileNotFoundError(args_cli.vision_checkpoint)

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    configure_parallel_physx_capacity(env_cfg, 1)
    if args_cli.keep_open:
        env_cfg.episode_length_s = 24.0 * 60.0 * 60.0
    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        randomizer = create_domain_randomizer(args_cli.randomization_config)
        domain_params = randomizer.apply_batch(env, randomizer.sample_batch(1))
        _refresh_camera(env)

        estimator = RGBDPoseEstimator.load(args_cli.vision_checkpoint, env.device)
        visual_filter = VisualPolicyObservationAdapter(ema_alpha=0.45)
        raw_visual_position, perception_valid = _visual_position_b(env, estimator)
        visual_position = visual_filter.filter_position(raw_visual_position)
        lift_anchor = visual_position.clone()

        scripted = PickAndLiftStateMachine(env.step_dt, 1, env.device)
        guard = HybridActionGuard(maximum_position_step_m=0.04)
        client = VLAClient(args_cli.vla_host, args_cli.vla_port, args_cli.vla_timeout)
        action_queue: deque[np.ndarray] = deque()
        repeated_action: np.ndarray | None = None
        repeat_remaining = 0
        ee_pose_b, true_object_pose_b = _poses_in_robot_frame(env)
        actions = torch.cat(
            (ee_pose_b, torch.ones((1, 1), device=env.device)), dim=-1
        )

        initial_height = env.scene["object"].data.root_pos_w[:, 2].clone()
        maximum_height = initial_height.clone()
        success_streak = torch.zeros(1, dtype=torch.long, device=env.device)
        success = torch.zeros(1, dtype=torch.bool, device=env.device)
        source_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        request_phase_counts: Counter[str] = Counter()
        vla_errors: list[str] = []
        inference_times_ms: list[float] = []
        step_count = 0

        print(
            "HYBRID_VLA_CONFIGURATION="
            + json.dumps(
                {
                    "vision_checkpoint": str(args_cli.vision_checkpoint.resolve()),
                    "vla_server": f"{args_cli.vla_host}:{args_cli.vla_port}",
                    "state": (
                        "30-D proprioception + RGB-D target xyz/validity + pre-action "
                        "Scripted IK FSM phase; no simulator object pose"
                        if args_cli.condition_vla_on_controller_phase
                        else (
                            "29-D proprioception + RGB-D estimated target xyz/validity; "
                            "no simulator object pose"
                            if args_cli.condition_vla_on_visual_pose
                            else "25-D robot proprioception; no simulator object pose"
                        )
                    ),
                    "state_dimension": (
                        30
                        if args_cli.condition_vla_on_controller_phase
                        else 29 if args_cli.condition_vla_on_visual_pose else 25
                    ),
                    "controller_phase_source": (
                        "scripted_state_captured_before_compute"
                        if args_cli.condition_vla_on_controller_phase
                        else None
                    ),
                    "action": (
                        "8-D delta xyz + absolute quaternion/gripper; decoded "
                        "against current EEF at execution"
                        if args_cli.vla_position_target_mode == "delta"
                        else "8-D absolute IK target"
                    ),
                    "vla_position_target_mode": args_cli.vla_position_target_mode,
                    "fallback": "RGB-D target + Scripted IK",
                    "visual_target_latch": (
                        "EMA during REST; freeze when approach starts to avoid robot occlusion"
                    ),
                    "task": args_cli.task,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        with torch.inference_mode():
            for step_count in range(1, args_cli.max_steps + 1):
                wall_start = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(actions)
                ee_pose_b, true_object_pose_b = _poses_in_robot_frame(env)

                target_unoccluded = scripted.state < int(
                    PickAndLiftState.APPROACH_ABOVE_OBJECT
                )
                if (
                    step_count % args_cli.action_repeat == 0
                    and bool(target_unoccluded[0])
                ):
                    estimate, current_valid = _visual_position_b(env, estimator)
                    perception_valid = current_valid
                    if bool(current_valid[0]):
                        visual_position = visual_filter.filter_position(estimate)
                        lift_anchor.copy_(visual_position)

                predicted_object_pose = torch.zeros((1, 7), device=env.device)
                predicted_object_pose[:, :3] = visual_position
                predicted_object_pose[:, 3] = 1.0
                lift_pose = predicted_object_pose.clone()
                lift_pose[:, :3] = lift_anchor
                lift_pose[:, 2] += 0.18
                # Capture the state that generates this step's teacher action.
                # ``compute`` may transition the FSM, so reading it afterwards
                # would create a one-phase label/inference mismatch.
                controller_phase = int(scripted.state[0].item())
                fallback_action = scripted.compute(ee_pose_b, predicted_object_pose, lift_pose)

                if not action_queue and repeat_remaining == 0:
                    try:
                        request_phase_counts[str(controller_phase)] += 1
                        response = client.predict(
                            _vla_request(
                                env,
                                ee_pose_b,
                                args_cli.task,
                                visual_position,
                                perception_valid,
                                controller_phase,
                            )
                        )
                        inference_times_ms.append(response.inference_ms)
                        for item in response.actions[: args_cli.execution_horizon]:
                            action_queue.append(item.copy())
                    except (ConnectionError, OSError, TimeoutError, RuntimeError, ValueError) as error:
                        vla_errors.append(f"{type(error).__name__}: {error}")
                        if args_cli.require_vla:
                            raise

                if repeat_remaining == 0:
                    queued_action = action_queue.popleft() if action_queue else None
                    repeated_action = decode_vla_position_target(
                        queued_action,
                        ee_pose_b[0].detach().cpu().numpy(),
                        position_target_mode=args_cli.vla_position_target_mode,
                    )
                    repeat_remaining = args_cli.action_repeat if repeated_action is not None else 1
                repeat_remaining -= 1

                decision = guard.filter(
                    repeated_action,
                    fallback_action[0].detach().cpu().numpy(),
                    current_eef_pose=ee_pose_b[0].detach().cpu().numpy(),
                    visual_object_position=visual_position[0].detach().cpu().numpy(),
                    perception_valid=bool(perception_valid[0]),
                )
                source_counts[decision.source] += 1
                reason_counts[decision.reason] += 1
                actions = torch.as_tensor(decision.action, device=env.device).unsqueeze(0)

                object_position_w = env.scene["object"].data.root_pos_w
                ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
                finger_position = env.scene["robot"].data.joint_pos[:, -2:].mean(dim=-1)
                maximum_height = torch.maximum(maximum_height, object_position_w[:, 2])
                held = torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1) < 0.08
                valid_success = (
                    (object_position_w[:, 2] >= TABLE_TOP_Z + 0.10)
                    & (finger_position < 0.030)
                    & held
                )
                success_streak = torch.where(
                    valid_success, success_streak + 1, torch.zeros_like(success_streak)
                )
                success = success_streak >= 10
                if bool(success[0]) or bool((terminated | truncated)[0]):
                    break
                if args_cli.realtime:
                    remaining = env.step_dt / args_cli.realtime_speed - (time.perf_counter() - wall_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

            if bool(success[0]) and args_cli.keep_open:
                print("HYBRID_VLA_GUI_HOLD=success; close Isaac Sim to exit.", flush=True)
                while simulation_app.is_running():
                    env.step(actions)

        summary = {
            "status": "HYBRID_VLA_SUCCESS" if bool(success[0]) else "HYBRID_VLA_FAILED",
            "success": bool(success[0]),
            "steps": step_count,
            "maximum_lift_cm": float((maximum_height[0] - initial_height[0]).cpu() * 100.0),
            "source_counts": dict(source_counts),
            "decision_reasons": dict(reason_counts),
            "vla_request_phase_counts": dict(sorted(request_phase_counts.items())),
            "state_dimension": (
                30
                if args_cli.condition_vla_on_controller_phase
                else 29 if args_cli.condition_vla_on_visual_pose else 25
            ),
            "condition_vla_on_visual_pose": args_cli.condition_vla_on_visual_pose,
            "condition_vla_on_controller_phase": (
                args_cli.condition_vla_on_controller_phase
            ),
            "vla_position_target_mode": args_cli.vla_position_target_mode,
            "controller_phase_source": (
                "scripted_state_captured_before_compute"
                if args_cli.condition_vla_on_controller_phase
                else None
            ),
            "vla_requests": len(inference_times_ms),
            "mean_vla_inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
            "vla_error_count": len(vla_errors),
            "last_vla_error": vla_errors[-1] if vla_errors else None,
            "sampled_domain": domain_params[0],
            "object_truth_usage": "success metrics only",
        }
        args_cli.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("HYBRID_VLA_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
        if not bool(success[0]):
            raise RuntimeError("Hybrid VLA rollout did not achieve a sustained 10 cm lift")
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
