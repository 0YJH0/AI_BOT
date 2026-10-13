"""Collect targeted phase-1 VLA DAgger recovery trajectories near 1.5 cm.

Each compact episode is a continuous 10 Hz sequence.  The current VLA first
acts from a deliberately staged pre-grasp-shell state; Scripted IK then labels
and executes the recovery.  Labels always use the latched RGB-D target rather
than privileged simulator object state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

from isaac_lab_data_engine.benchmark.config import BenchmarkLevel, load_benchmark_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/benchmark.yaml")
parser.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
parser.add_argument("--train-frames-per-level", type=int, default=200)
parser.add_argument("--validation-frames-per-level", type=int, default=50)
parser.add_argument("--rollouts-per-scene", type=int, default=4)
parser.add_argument("--frames-per-rollout", type=int, default=4)
parser.add_argument("--maximum-rollout-attempts", type=int, default=1000)
parser.add_argument("--maximum-scene-attempts", type=int, default=1000)
parser.add_argument(
    "--domain-seed-offset",
    type=int,
    default=0,
    help="Add to each benchmark level seed so supplemental rounds do not reuse old scenes.",
)
parser.add_argument("--approach-height-m", type=float, default=0.12)
parser.add_argument("--band-min-m", type=float, default=0.0)
parser.add_argument("--band-target-m", type=float, default=0.015)
parser.add_argument("--band-max-m", type=float, default=0.030)
parser.add_argument(
    "--require-learner-visited-in-band",
    action="store_true",
    help="Reject a rollout unless the post-learner state remains inside the target band.",
)
parser.add_argument(
    "--maximum-record-distance-m",
    type=float,
    default=0.045,
    help="Keep a continuous post-policy recovery even if it briefly exits the 3 cm target band.",
)
parser.add_argument("--stage-steps", type=int, default=35)
parser.add_argument(
    "--stage-tolerance-m",
    type=float,
    default=0.05,
    help="Safety bound on shell-target tracking error; acceptance is also gated by actual pre-grasp distance.",
)
parser.add_argument("--action-repeat", type=int, default=5)
parser.add_argument("--recovery-step-m", type=float, default=0.006)
parser.add_argument("--maximum-label-step-m", type=float, default=0.04)
parser.add_argument("--recovery-loss-weight", type=float, default=2.0)
parser.add_argument("--hard-loss-multiplier", type=float, default=1.25)
parser.add_argument(
    "--learner-visited-loss-multiplier",
    type=float,
    default=1.0,
    help="Extra loss weight for the state reached immediately after the learner action.",
)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--camera-warmup-min-steps", type=int, default=10)
parser.add_argument("--camera-warmup-max-steps", type=int, default=30)
parser.add_argument(
    "--camera-ready-timeout-s",
    type=float,
    default=180.0,
    help="Wall-clock deadline for RGB/Depth and visual-pose readiness after a reset.",
)
parser.add_argument(
    "--camera-render-yield-s",
    type=float,
    default=0.02,
    help="Small scheduler yield between RTX render checks.",
)
parser.add_argument("--visual-init-window", type=int, default=5)
parser.add_argument("--visual-init-attempts", type=int, default=3)
parser.add_argument("--visual-init-max-spread-m", type=float, default=0.025)
parser.add_argument(
    "--visual-invalid-pose-timeout-s",
    type=float,
    default=30.0,
    help="Reject persistent invalid pose estimates after RGB-D becomes valid; 0 disables early rejection.",
)
parser.add_argument("--camera-refresh-renders", type=int, default=3)
parser.add_argument(
    "--camera-capture-attempts",
    type=int,
    default=5,
    help="Retry render/annotator refresh before rejecting an otherwise valid trajectory frame.",
)
parser.add_argument(
    "--camera-capture-timeout-s",
    type=float,
    default=5.0,
    help="Wall-clock deadline for an RGB-D frame that temporarily drops mid-rollout.",
)
parser.add_argument("--camera-min-rgb-std", type=float, default=30.0)
parser.add_argument("--camera-min-rgb-range", type=float, default=128.0)
parser.add_argument("--camera-min-depth-fraction", type=float, default=0.40)
parser.add_argument(
    "--require-complete-rollout",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Keep only trajectories containing the requested number of continuous frames.",
)
parser.add_argument(
    "--vision-checkpoint",
    type=Path,
    default=PROJECT_ROOT / "results/vision/rgbd_keypoint_supplement_160x120_v3/best_model.pt",
)
parser.add_argument("--vla-host", default="127.0.0.1")
parser.add_argument("--vla-port", type=int, default=8765)
parser.add_argument("--vla-timeout", type=float, default=30.0)
parser.add_argument(
    "--learner-checkpoint-role",
    default="unspecified_vla_learner",
    help="Checkpoint identifier persisted in every episode and the collection summary.",
)
parser.add_argument("--task", default="Pick up the red cube and lift it 10 centimeters.")
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "dataset/vla_dagger_recovery_pregrasp15mm_v6",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

benchmark_cfg = load_benchmark_config(args_cli.config)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.randomization import create_domain_randomizer
    from isaac_lab_data_engine.vision import (
        RGBDPoseEstimator,
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
    from isaac_lab_data_engine.vla.recovery import (
        APPROACH_PHASE,
        RECOVERY_FORMAT_VERSION,
        RecoveryBand,
        build_pregrasp_recovery_actions,
        distance_to_pregrasp,
        pregrasp_waypoint,
        validate_recovery_episode,
    )
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


DIFFICULTY_IDS = {"easy": 0.0, "medium": 1.0, "hard": 2.0}
TOP_DOWN_QUAT = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32)


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
    position_camera = torch.stack(
        (
            (x[None] - intrinsics[:, 0, 2, None, None])
            / intrinsics[:, 0, 0, None, None]
            * depth,
            (y[None] - intrinsics[:, 1, 2, None, None])
            / intrinsics[:, 1, 1, None, None]
            * depth,
            depth,
        ),
        dim=-1,
    )
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
    raw = world_to_root_position(
        position_w, robot.data.root_pos_w, robot.data.root_quat_w
    )
    valid = (
        torch.isfinite(raw).all(dim=-1)
        & (raw[:, 0] >= 0.365)
        & (raw[:, 0] <= 0.775)
        & (raw[:, 1].abs() <= 0.295)
        & (raw[:, 2] >= -0.03)
        & (raw[:, 2] <= 0.15)
    )
    finite = torch.empty_like(raw)
    finite[:, 0] = torch.nan_to_num(raw[:, 0], nan=0.55).clamp(0.36, 0.78)
    finite[:, 1] = torch.nan_to_num(raw[:, 1], nan=0.0).clamp(-0.30, 0.30)
    finite[:, 2] = 0.03
    return finite, valid


def _camera_quality(env: ManagerBasedRLEnv) -> dict[str, float | bool | str]:
    camera = env.scene["table_camera"]
    rgb = camera.data.output["rgb"][0, ..., :3].to(torch.float32)
    depth = camera.data.output["distance_to_image_plane"][0].to(torch.float32)
    rgb_std = float(rgb.std(unbiased=False).cpu())
    rgb_range = float((rgb.max() - rgb.min()).cpu())
    depth_fraction = float(
        (torch.isfinite(depth) & (depth > 0.0)).float().mean().cpu()
    )
    valid = (
        rgb_std >= args_cli.camera_min_rgb_std
        and rgb_range >= args_cli.camera_min_rgb_range
        and depth_fraction >= args_cli.camera_min_depth_fraction
    )
    return {
        "valid": valid,
        "rgb_std": rgb_std,
        "rgb_range": rgb_range,
        "depth_valid_fraction": depth_fraction,
        "rgb_sha256": hashlib.sha256(
            rgb.to(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest(),
    }


@torch.inference_mode()
def _initialize_visual_position(
    env: ManagerBasedRLEnv, estimator: RGBDPoseEstimator
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    camera = env.scene["table_camera"]
    env.sim.forward()
    camera.reset()
    ee_pose, _ = _poses_in_robot_frame(env)
    hold = torch.cat((ee_pose, torch.ones((1, 1), device=env.device)), dim=-1)
    recent: deque[torch.Tensor] = deque(maxlen=args_cli.visual_init_window)
    diagnostics: dict[str, Any] = {"stable": False}
    started = time.perf_counter()
    deadline = started + args_cli.camera_ready_timeout_s
    render_check = 0
    control_steps = 0
    invalid_pose_since: float | None = None
    while time.perf_counter() < deadline:
        render_check += 1
        if control_steps < args_cli.camera_warmup_max_steps:
            _, _, terminated, truncated, _ = env.step(hold)
            control_steps += 1
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError("Environment reset during visual initialization")
        for _ in range(args_cli.camera_refresh_renders):
            env.sim.render()
            camera.update(0.0, force_recompute=True)
        if args_cli.camera_render_yield_s:
            time.sleep(args_cli.camera_render_yield_s)
        if control_steps < args_cli.camera_warmup_min_steps:
            continue
        quality = _camera_quality(env)
        diagnostics = {
            **quality,
            "steps": control_steps,
            "render_checks": render_check,
            "elapsed_s": time.perf_counter() - started,
            "stable": False,
        }
        if not bool(quality["valid"]):
            recent.clear()
            invalid_pose_since = None
            continue
        estimate, valid = _visual_position_b(env, estimator)
        diagnostics.update(
            {
                "pose_valid": bool(valid[0]),
                "pose_xyz_b": estimate[0].detach().cpu().tolist(),
            }
        )
        if not bool(valid[0]):
            recent.clear()
            now = time.perf_counter()
            if invalid_pose_since is None:
                invalid_pose_since = now
            if (
                args_cli.visual_invalid_pose_timeout_s > 0.0
                and now - invalid_pose_since >= args_cli.visual_invalid_pose_timeout_s
            ):
                diagnostics["early_rejection"] = "persistent_invalid_visual_pose"
                break
            continue
        invalid_pose_since = None
        recent.append(estimate[0].clone())
        if len(recent) < args_cli.visual_init_window:
            continue
        samples = torch.stack(tuple(recent))
        median = samples.median(dim=0).values
        spread = float(torch.linalg.vector_norm(samples - median, dim=-1).max().cpu())
        diagnostics.update({"stable": spread <= args_cli.visual_init_max_spread_m, "spread_m": spread})
        if bool(diagnostics["stable"]):
            env.episode_length_buf.zero_()
            return median.unsqueeze(0), torch.ones(1, dtype=torch.bool, device=env.device), diagnostics
    env.episode_length_buf.zero_()
    fallback = torch.tensor([[0.55, 0.0, 0.03]], dtype=ee_pose.dtype, device=env.device)
    return fallback, torch.zeros(1, dtype=torch.bool, device=env.device), diagnostics


def _pack_request(
    env: ManagerBasedRLEnv,
    ee_pose: torch.Tensor,
    visual_position: torch.Tensor,
) -> VLARequest:
    robot = env.scene["robot"]
    state = pack_proprioceptive_state(
        robot.data.joint_pos[0].detach().cpu().numpy(),
        robot.data.joint_vel[0].detach().cpu().numpy(),
        ee_pose[0].detach().cpu().numpy(),
        visual_position[0].detach().cpu().numpy(),
        True,
        controller_phase=float(APPROACH_PHASE),
    )
    rgb = env.scene["table_camera"].data.output["rgb"][0, ..., :3]
    return VLARequest(
        rgb=rgb.detach().cpu().numpy().astype(np.uint8, copy=False),
        state=state,
        task=args_cli.task,
    )


def _absolute_action(eef_pose: np.ndarray, delta_xyz: np.ndarray) -> np.ndarray:
    action = np.empty((8,), dtype=np.float32)
    action[:3] = np.asarray(eef_pose[:3], dtype=np.float32) + np.asarray(
        delta_xyz, dtype=np.float32
    )
    action[3:7] = TOP_DOWN_QUAT
    action[7] = 1.0
    return action


def _step_action(
    env: ManagerBasedRLEnv, action: np.ndarray, repeats: int
) -> bool:
    tensor = torch.as_tensor(action, dtype=torch.float32, device=env.device).unsqueeze(0)
    for _ in range(repeats):
        _, _, terminated, truncated, _ = env.step(tensor)
        if bool(terminated[0]) or bool(truncated[0]):
            return False
    return True


def _stage_shell(
    env: ManagerBasedRLEnv,
    visual_position: torch.Tensor,
    *,
    radius_m: float,
    angle_rad: float,
) -> tuple[bool, torch.Tensor, float, float]:
    waypoint = pregrasp_waypoint(
        visual_position[0].detach().cpu().numpy(),
        approach_height_m=args_cli.approach_height_m,
    )
    # Mostly lateral offsets keep the wrist safely above the object while
    # covering both sides of the 1.5 cm phase threshold.
    direction = np.asarray(
        [np.cos(angle_rad), np.sin(angle_rad), 0.20 * np.sin(2.0 * angle_rad)],
        dtype=np.float32,
    )
    direction /= np.linalg.norm(direction)
    target = waypoint + float(radius_m) * direction
    stage_action = np.concatenate((target, TOP_DOWN_QUAT, [1.0])).astype(np.float32)
    alive = _step_action(env, stage_action, args_cli.stage_steps)
    ee_pose, _ = _poses_in_robot_frame(env)
    error = float(
        np.linalg.norm(ee_pose[0, :3].detach().cpu().numpy() - target)
    )
    pregrasp_distance = float(
        distance_to_pregrasp(
            ee_pose[0].detach().cpu().numpy(),
            visual_position[0].detach().cpu().numpy(),
            approach_height_m=args_cli.approach_height_m,
        )
    )
    staged = (
        alive
        and error <= args_cli.stage_tolerance_m
        and args_cli.band_min_m <= pregrasp_distance <= args_cli.band_max_m
    )
    return staged, ee_pose, error, pregrasp_distance


def _capture_frame(
    env: ManagerBasedRLEnv,
    visual_position: torch.Tensor,
    *,
    level_name: str,
    learner_visited: bool,
    executed_action: np.ndarray,
) -> dict[str, np.ndarray] | None:
    camera = env.scene["table_camera"]
    quality: dict[str, float | bool | str] = {"valid": False}
    capture_started = time.perf_counter()
    capture_attempt = 0
    while (
        capture_attempt < args_cli.camera_capture_attempts
        or time.perf_counter() - capture_started < args_cli.camera_capture_timeout_s
    ):
        capture_attempt += 1
        for _ in range(args_cli.camera_refresh_renders):
            env.sim.render()
            camera.update(0.0, force_recompute=True)
        quality = _camera_quality(env)
        if bool(quality["valid"]):
            break
        if args_cli.camera_render_yield_s:
            time.sleep(args_cli.camera_render_yield_s)
    if not bool(quality["valid"]):
        return None
    ee_pose, _ = _poses_in_robot_frame(env)
    ee_np = ee_pose[0].detach().cpu().numpy().astype(np.float32, copy=False)
    visual_np = visual_position[0].detach().cpu().numpy().astype(np.float32, copy=False)
    distance = float(
        distance_to_pregrasp(
            ee_np, visual_np, approach_height_m=args_cli.approach_height_m
        )
    )
    if distance > args_cli.maximum_record_distance_m:
        return None
    robot = env.scene["robot"]
    state = pack_proprioceptive_state(
        robot.data.joint_pos[0].detach().cpu().numpy(),
        robot.data.joint_vel[0].detach().cpu().numpy(),
        ee_np,
        visual_np,
        True,
        controller_phase=float(APPROACH_PHASE),
    )
    label = build_pregrasp_recovery_actions(
        ee_np,
        visual_np,
        maximum_position_step_m=args_cli.maximum_label_step_m,
        approach_height_m=args_cli.approach_height_m,
    )
    loss_weight = args_cli.recovery_loss_weight
    if level_name == "hard":
        loss_weight *= args_cli.hard_loss_multiplier
    if learner_visited:
        loss_weight *= args_cli.learner_visited_loss_multiplier
    return {
        "rgb": camera.data.output["rgb"][0, ..., :3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8, copy=False),
        "state": state.astype(np.float32, copy=False),
        "action": label.astype(np.float32, copy=False),
        "controller_phase": np.asarray([APPROACH_PHASE], dtype=np.float32),
        "difficulty_id": np.asarray([DIFFICULTY_IDS[level_name]], dtype=np.float32),
        "loss_weight": np.asarray([loss_weight], dtype=np.float32),
        "distance_to_pregrasp_m": np.asarray([distance], dtype=np.float32),
        "learner_visited": np.asarray([learner_visited], dtype=np.bool_),
        "executed_action": np.asarray(executed_action, dtype=np.float32),
        "visual_position_b": visual_np.copy(),
        "eef_pose_b": ee_np.copy(),
        "rgb_std": np.asarray([quality["rgb_std"]], dtype=np.float32),
        "rgb_range": np.asarray([quality["rgb_range"]], dtype=np.float32),
        "depth_valid_fraction": np.asarray(
            [quality["depth_valid_fraction"]], dtype=np.float32
        ),
    }


def _stack_frames(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}


def _write_episode(
    root: Path,
    *,
    split: str,
    level: BenchmarkLevel,
    episode_index: int,
    frames: list[dict[str, np.ndarray]],
    domain_params: dict[str, Any],
    visual_init: dict[str, Any],
    policy_inference_ms: float,
    policy_reason: str,
) -> dict[str, Any]:
    arrays = _stack_frames(frames)
    validate_recovery_episode(arrays)
    level_dir = root / split / level.name
    level_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_index:06d}"
    npz_path = level_dir / f"{stem}.npz"
    json_path = level_dir / f"{stem}.json"
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite {npz_path}")
    temporary = npz_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(npz_path)
    distances = arrays["distance_to_pregrasp_m"].reshape(-1)
    metadata = {
        "format_version": RECOVERY_FORMAT_VERSION,
        "split": split,
        "level": level.name,
        "episode_index": episode_index,
        "frames": len(frames),
        "task": args_cli.task,
        "policy_checkpoint_role": args_cli.learner_checkpoint_role,
        "policy_position_target_mode": "delta",
        "recovery_band_m": {
            "minimum": args_cli.band_min_m,
            "target": args_cli.band_target_m,
            "maximum": args_cli.band_max_m,
        },
        "controller_phase": APPROACH_PHASE,
        "label_source": "visual_scripted_ik_delta_xyz",
        "privileged_object_state_used_for_selection_or_label": False,
        "continuous_trajectory": True,
        "learner_visited_frames": int(arrays["learner_visited"].sum()),
        "distance_min_m": float(distances.min()),
        "distance_mean_m": float(distances.mean()),
        "distance_max_m": float(distances.max()),
        "policy_inference_ms": policy_inference_ms,
        "policy_decision_reason": policy_reason,
        "domain_parameters": domain_params,
        "visual_initialization": visual_init,
        "npz": str(npz_path.resolve()),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


@torch.inference_mode()
def _collect_rollout(
    env: ManagerBasedRLEnv,
    client: VLAClient,
    visual_position: torch.Tensor,
    *,
    level: BenchmarkLevel,
    radius_m: float,
    angle_rad: float,
) -> tuple[list[dict[str, np.ndarray]], float, str]:
    staged, ee_pose, stage_error, stage_distance = _stage_shell(
        env, visual_position, radius_m=radius_m, angle_rad=angle_rad
    )
    if not staged:
        return (
            [],
            0.0,
            "stage_failed:"
            f"target_error_m={stage_error:.5f},"
            f"pregrasp_distance_m={stage_distance:.5f}",
        )
    env.episode_length_buf.zero_()
    expert_delta = build_pregrasp_recovery_actions(
        ee_pose[0].detach().cpu().numpy(),
        visual_position[0].detach().cpu().numpy(),
        maximum_position_step_m=args_cli.maximum_label_step_m,
        approach_height_m=args_cli.approach_height_m,
    )[:3]
    fallback = _absolute_action(ee_pose[0].detach().cpu().numpy(), expert_delta)
    first = _capture_frame(
        env,
        visual_position,
        level_name=level.name,
        learner_visited=False,
        executed_action=fallback,
    )
    if first is None:
        return [], 0.0, "invalid_camera_or_distance_at_seed"

    started = time.perf_counter()
    response = client.predict(_pack_request(env, ee_pose, visual_position))
    rpc_ms = (time.perf_counter() - started) * 1000.0
    raw = response.actions[0] if len(response.actions) else None
    decoded = decode_vla_position_target(
        raw,
        ee_pose[0].detach().cpu().numpy(),
        position_target_mode="delta",
    )
    guard = HybridActionGuard(maximum_position_step_m=args_cli.maximum_label_step_m)
    decision = guard.filter(
        decoded,
        fallback,
        current_eef_pose=ee_pose[0].detach().cpu().numpy(),
        visual_object_position=visual_position[0].detach().cpu().numpy(),
        perception_valid=True,
    )
    if decision.source != "vla":
        return [], rpc_ms, decision.reason

    first["executed_action"] = decision.action.astype(np.float32, copy=True)
    frames = [first]
    if not _step_action(env, decision.action, args_cli.action_repeat):
        return [], rpc_ms, "environment_ended_after_vla"
    for frame_index in range(1, args_cli.frames_per_rollout):
        ee_pose, _ = _poses_in_robot_frame(env)
        expert = build_pregrasp_recovery_actions(
            ee_pose[0].detach().cpu().numpy(),
            visual_position[0].detach().cpu().numpy(),
            maximum_position_step_m=args_cli.maximum_label_step_m,
            approach_height_m=args_cli.approach_height_m,
        )
        step_delta = expert[:3].copy()
        norm = float(np.linalg.norm(step_delta))
        if norm > args_cli.recovery_step_m:
            step_delta *= args_cli.recovery_step_m / norm
        recovery_action = _absolute_action(
            ee_pose[0].detach().cpu().numpy(), step_delta
        )
        frame = _capture_frame(
            env,
            visual_position,
            level_name=level.name,
            learner_visited=frame_index == 1,
            executed_action=recovery_action,
        )
        if frame is None:
            break
        if frame_index == 1 and args_cli.require_learner_visited_in_band:
            learner_distance = float(frame["distance_to_pregrasp_m"].reshape(-1)[0])
            if not args_cli.band_min_m <= learner_distance <= args_cli.band_max_m:
                return (
                    [],
                    rpc_ms,
                    f"learner_visited_outside_band:distance_m={learner_distance:.5f}",
                )
        frames.append(frame)
        if not _step_action(env, recovery_action, args_cli.action_repeat):
            break
    if not any(bool(frame["learner_visited"].reshape(-1)[0]) for frame in frames):
        return [], rpc_ms, "no_learner_visited_frame"
    if args_cli.require_complete_rollout and len(frames) != args_cli.frames_per_rollout:
        return (
            [],
            rpc_ms,
            f"incomplete_rollout:frames={len(frames)}/{args_cli.frames_per_rollout}",
        )
    return frames, rpc_ms, decision.reason


def _selected_levels() -> list[BenchmarkLevel]:
    configured = {level.name: level for level in benchmark_cfg.levels}
    unknown = set(args_cli.levels) - set(configured)
    if unknown:
        raise ValueError(f"Unknown benchmark levels: {sorted(unknown)}")
    return [configured[name] for name in args_cli.levels]


def _validate_args() -> RecoveryBand:
    positive = (
        args_cli.train_frames_per_level,
        args_cli.validation_frames_per_level,
        args_cli.rollouts_per_scene,
        args_cli.frames_per_rollout,
        args_cli.maximum_rollout_attempts,
        args_cli.maximum_scene_attempts,
        args_cli.stage_steps,
        args_cli.action_repeat,
        args_cli.camera_refresh_renders,
        args_cli.camera_capture_attempts,
    )
    if min(positive) < 1:
        raise ValueError("Frame counts, rollout counts, and step counts must be positive")
    if args_cli.camera_ready_timeout_s <= 0.0 or args_cli.camera_capture_timeout_s < 0.0:
        raise ValueError("Camera readiness timeouts must be positive/non-negative")
    if args_cli.camera_render_yield_s < 0.0:
        raise ValueError("camera-render-yield-s must be non-negative")
    if not np.isfinite(args_cli.visual_invalid_pose_timeout_s) or args_cli.visual_invalid_pose_timeout_s < 0.0:
        raise ValueError("visual-invalid-pose-timeout-s must be finite and non-negative")
    if args_cli.maximum_record_distance_m < args_cli.band_max_m:
        raise ValueError("maximum-record-distance-m must cover band-max-m")
    if args_cli.recovery_step_m <= 0.0 or args_cli.maximum_label_step_m <= 0.0:
        raise ValueError("Recovery and label step sizes must be positive")
    if (
        args_cli.recovery_loss_weight <= 0.0
        or args_cli.hard_loss_multiplier <= 0.0
        or args_cli.learner_visited_loss_multiplier <= 0.0
    ):
        raise ValueError("Loss multipliers must be positive")
    if not args_cli.learner_checkpoint_role.strip():
        raise ValueError("learner-checkpoint-role must be non-empty")
    effective_seeds = [level.seed + args_cli.domain_seed_offset for level in _selected_levels()]
    if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in effective_seeds):
        raise ValueError("Effective domain seeds must fit in unsigned 32-bit integers")
    if not args_cli.vision_checkpoint.is_file():
        raise FileNotFoundError(args_cli.vision_checkpoint)
    if args_cli.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args_cli.output_root}")
    return RecoveryBand(
        args_cli.band_min_m, args_cli.band_target_m, args_cli.band_max_m
    )


@torch.inference_mode()
def main() -> None:
    band = _validate_args()
    levels = _selected_levels()
    args_cli.output_root.mkdir(parents=True)
    print(
        "DAGGER_RECOVERY_STAGE="
        + json.dumps({"stage": "output_created", "output": str(args_cli.output_root)}),
        flush=True,
    )
    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.lazy_sensor_update = True
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 30.0
    configure_parallel_physx_capacity(env_cfg, 1)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("DAGGER_RECOVERY_STAGE={\"stage\":\"environment_ready\"}", flush=True)
    estimator = RGBDPoseEstimator.load(args_cli.vision_checkpoint, env.device)
    client = VLAClient(args_cli.vla_host, args_cli.vla_port, args_cli.vla_timeout)
    print("DAGGER_RECOVERY_STAGE={\"stage\":\"vision_and_rpc_ready\"}", flush=True)
    records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    distance_bins = np.asarray(
        [
            band.minimum_m,
            0.5 * (band.minimum_m + band.target_m),
            band.target_m,
            0.5 * (band.target_m + band.maximum_m),
            band.maximum_m,
        ],
        dtype=np.float32,
    )
    # Use bin centers so the 1.5 cm transition is approached from both sides.
    shell_radii = (distance_bins[:-1] + distance_bins[1:]) * 0.5

    try:
        env.reset()
        ee_pose, _ = _poses_in_robot_frame(env)
        warmup = torch.cat((ee_pose, torch.ones((1, 1), device=env.device)), dim=-1)
        for _ in range(args_cli.warmup_steps):
            env.step(warmup)

        for level in levels:
            effective_seed = level.seed + args_cli.domain_seed_offset
            np.random.seed(effective_seed)
            torch.manual_seed(effective_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(effective_seed)
            randomizer = create_domain_randomizer(level.randomization_config)
            randomizer.reseed(effective_seed)
            for split, target_frames in (
                ("train", args_cli.train_frames_per_level),
                ("validation", args_cli.validation_frames_per_level),
            ):
                collected = 0
                episode_index = 0
                rollout_attempt = 0
                scene_attempt = 0
                while collected < target_frames:
                    scene_attempt += 1
                    if scene_attempt > args_cli.maximum_scene_attempts:
                        raise RuntimeError(
                            f"Exceeded {args_cli.maximum_scene_attempts} scene attempts "
                            f"for {split}/{level.name} with {collected}/{target_frames} frames; "
                            f"rejections={dict(rejection_counts)}"
                        )
                    params = randomizer.sample_batch(1)[0]
                    visual_position = None
                    visual_valid = None
                    visual_init: dict[str, Any] = {}
                    applied: dict[str, Any] | None = None
                    for attempt in range(1, args_cli.visual_init_attempts + 1):
                        env.reset()
                        applied = randomizer.apply_batch(env, [params])[0]
                        visual_position, visual_valid, visual_init = _initialize_visual_position(
                            env, estimator
                        )
                        visual_init["attempts"] = attempt
                        if bool(visual_valid[0]):
                            break
                    if visual_position is None or visual_valid is None or not bool(visual_valid[0]):
                        rejection_counts["visual_initialization"] += 1
                        count = rejection_counts["visual_initialization"]
                        if count <= 10 or count % 25 == 0:
                            print(
                                "DAGGER_RECOVERY_REJECT="
                                + json.dumps(
                                    {
                                        "split": split,
                                        "level": level.name,
                                        "reason": "visual_initialization",
                                        "count": count,
                                        "details": visual_init,
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        continue
                    assert applied is not None
                    for scene_rollout in range(args_cli.rollouts_per_scene):
                        if collected >= target_frames:
                            break
                        rollout_attempt += 1
                        if rollout_attempt > args_cli.maximum_rollout_attempts:
                            raise RuntimeError(
                                f"Exceeded {args_cli.maximum_rollout_attempts} rollout attempts "
                                f"for {split}/{level.name} with {collected}/{target_frames} frames; "
                                f"rejections={dict(rejection_counts)}"
                            )
                        radius = float(shell_radii[scene_rollout % len(shell_radii)])
                        angle = float(
                            2.0
                            * np.pi
                            * ((episode_index * 0.61803398875 + scene_rollout / 4.0) % 1.0)
                        )
                        try:
                            frames, inference_ms, reason = _collect_rollout(
                                env,
                                client,
                                visual_position,
                                level=level,
                                radius_m=radius,
                                angle_rad=angle,
                            )
                        except (ConnectionError, OSError, TimeoutError, RuntimeError, ValueError) as error:
                            rejection_counts[type(error).__name__] += 1
                            print(
                                "DAGGER_RECOVERY_REJECT="
                                + json.dumps(
                                    {
                                        "split": split,
                                        "level": level.name,
                                        "reason": f"{type(error).__name__}: {error}",
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                            continue
                        if not frames:
                            rejection_counts[reason] += 1
                            print(
                                "DAGGER_RECOVERY_REJECT="
                                + json.dumps(
                                    {
                                        "split": split,
                                        "level": level.name,
                                        "reason": reason,
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                            continue
                        episode_index += 1
                        metadata = _write_episode(
                            args_cli.output_root,
                            split=split,
                            level=level,
                            episode_index=episode_index,
                            frames=frames,
                            domain_params=applied,
                            visual_init=visual_init,
                            policy_inference_ms=inference_ms,
                            policy_reason=reason,
                        )
                        records.append(metadata)
                        collected += len(frames)
                        print(
                            "DAGGER_RECOVERY_EPISODE="
                            + json.dumps(
                                {
                                    "split": split,
                                    "level": level.name,
                                    "episode": episode_index,
                                    "frames": len(frames),
                                    "collected": collected,
                                    "target": target_frames,
                                    "learner_visited": metadata["learner_visited_frames"],
                                    "distance_mean_cm": metadata["distance_mean_m"] * 100.0,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
    finally:
        env.close()

    summary = {
        "status": "VLA_DAGGER_RECOVERY_COLLECTION_OK",
        "format_version": RECOVERY_FORMAT_VERSION,
        "output_root": str(args_cli.output_root.resolve()),
        "task": args_cli.task,
        "policy_checkpoint_role": args_cli.learner_checkpoint_role,
        "domain_seed_offset": args_cli.domain_seed_offset,
        "effective_domain_seeds": {
            level.name: level.seed + args_cli.domain_seed_offset for level in levels
        },
        "distance_definition": "norm(eef_xyz - (latched_visual_xyz + [0,0,0.12]))",
        "privileged_object_state_used_for_selection_or_label": False,
        "band_m": {
            "minimum": band.minimum_m,
            "target": band.target_m,
            "maximum": band.maximum_m,
        },
        "require_learner_visited_in_band": args_cli.require_learner_visited_in_band,
        "distance_bin_edges_m": distance_bins.tolist(),
        "action_repeat": args_cli.action_repeat,
        "frames_per_rollout": args_cli.frames_per_rollout,
        "require_complete_rollout": args_cli.require_complete_rollout,
        "camera_capture_attempts": args_cli.camera_capture_attempts,
        "camera_ready_timeout_s": args_cli.camera_ready_timeout_s,
        "visual_invalid_pose_timeout_s": args_cli.visual_invalid_pose_timeout_s,
        "camera_capture_timeout_s": args_cli.camera_capture_timeout_s,
        "camera_render_yield_s": args_cli.camera_render_yield_s,
        "loss_weighting": {
            "recovery": args_cli.recovery_loss_weight,
            "hard_multiplier": args_cli.hard_loss_multiplier,
            "learner_visited_multiplier": args_cli.learner_visited_loss_multiplier,
        },
        "records": len(records),
        "frames": int(sum(int(record["frames"]) for record in records)),
        "learner_visited_frames": int(
            sum(int(record["learner_visited_frames"]) for record in records)
        ),
        "counts": {
            split: {
                level.name: {
                    "episodes": sum(
                        record["split"] == split and record["level"] == level.name
                        for record in records
                    ),
                    "frames": sum(
                        int(record["frames"])
                        for record in records
                        if record["split"] == split and record["level"] == level.name
                    ),
                }
                for level in levels
            }
            for split in ("train", "validation")
        },
        "rejections": dict(rejection_counts),
    }
    (args_cli.output_root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DAGGER_RECOVERY_COLLECTION=" + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
