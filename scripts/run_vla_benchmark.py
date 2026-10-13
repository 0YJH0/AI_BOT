"""Run a fixed-seed Easy/Medium/Hard closed-loop VLA benchmark in one Isaac process.

The runner keeps one Isaac environment alive for the complete benchmark and
advances each level's seeded DomainRandomizer exactly once per episode.  It
supports both hybrid VLA + Scripted IK fallback and a strict pure-VLA mode in
which a rejected/unavailable VLA action fails the episode without executing
the fallback action.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
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
parser.add_argument(
    "--config",
    type=Path,
    default=PROJECT_ROOT / "configs/benchmark.yaml",
)
parser.add_argument(
    "--levels",
    nargs="+",
    default=None,
    help="Configured levels to run, in order (default: Easy Medium Hard).",
)
parser.add_argument(
    "--episodes-per-level",
    "--episodes_per_level",
    dest="episodes_per_level",
    type=int,
    default=None,
    help="Common episode count; per-level overrides below take precedence.",
)
for _level_name in ("easy", "medium", "hard"):
    parser.add_argument(f"--{_level_name}-episodes", type=int, default=None)
parser.add_argument("--max-steps", "--max_steps", dest="max_steps", type=int, default=None)
parser.add_argument("--minimum-true-lift-m", type=float, default=0.0,
                    help="Optional actual lift relative to the initial object height; evaluation metric only.")
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--progress-every", type=int, default=0,
                    help="Print per-episode diagnostic progress every N control steps; 0 disables it.")
parser.add_argument("--phase-mechanism", choices=("legacy", "feedback_hysteresis"), default="legacy")
parser.add_argument("--student-phase-mode", choices=("legacy", "grip_feedback"), default="legacy")
parser.add_argument("--proprio-only", action="store_true",
                    help="25-D robot-only state; no pose estimator, target-distance gate or expert FSM.")
parser.add_argument("--orientation-mode", choices=("model", "fixed_top_down"), default="model")
parser.add_argument("--joint-target-actions", action="store_true",
                    help="Independent v23 interface: absolute seven arm joints plus gripper; direct PD hold.")
parser.add_argument("--record-phase-diagnostics", action="store_true")
parser.add_argument("--grasp-enable-distance-m", type=float, default=0.075,
                    help="Visual-target distance permitting the first gripper closure; latch behavior unchanged.")
parser.add_argument("--vla-input-interface", choices=("legacy", "relative4d"), default="legacy")
parser.add_argument(
    "--camera-warmup-min-steps",
    type=int,
    default=10,
    help="Minimum real Isaac control steps after each randomized reset before using RGB-D.",
)
parser.add_argument(
    "--camera-warmup-max-steps",
    type=int,
    default=30,
    help="Maximum real Isaac control steps allowed for a stable visual initialization.",
)
parser.add_argument(
    "--visual-init-window",
    type=int,
    default=5,
    help="Number of consecutive valid RGB-D estimates used by the median visual latch.",
)
parser.add_argument(
    "--visual-init-attempts",
    type=int,
    default=3,
    help="Maximum same-sample reset/retry attempts for visual initialization.",
)
parser.add_argument(
    "--visual-init-max-spread-m",
    type=float,
    default=0.025,
    help="Maximum distance from the window median for accepting the visual latch.",
)
parser.add_argument(
    "--camera-refresh-renders",
    type=int,
    default=3,
    help="Forced RTX render/update passes after each camera warmup control step.",
)
parser.add_argument("--camera-min-rgb-std", type=float, default=30.0)
parser.add_argument("--camera-ready-timeout-s", type=float, default=0.0,
                    help="Optional wall-clock readiness wait after the physics warmup; 0 keeps the step limit.")
parser.add_argument("--visual-invalid-pose-timeout-s", type=float, default=0.0,
                    help="Reject persistently invalid pose estimates after RGB-D becomes usable; 0 disables.")
parser.add_argument("--camera-render-yield-s", type=float, default=0.02)
parser.add_argument("--camera-min-rgb-range", type=float, default=128.0)
parser.add_argument(
    "--camera-min-depth-fraction",
    type=float,
    default=0.40,
    help="Minimum fraction of finite positive depth pixels required for a usable RGB-D frame.",
)
parser.add_argument(
    "--vision-checkpoint",
    type=Path,
    default=PROJECT_ROOT / "results/vision/rgbd_keypoint_supplement_160x120_v3/best_model.pt",
)
parser.add_argument("--vla-host", default="127.0.0.1")
parser.add_argument("--vla-port", type=int, default=8765)
parser.add_argument("--vla-timeout", type=float, default=30.0)
parser.add_argument("--task", default="Pick up the red cube and lift it 10 centimeters.")
parser.add_argument("--training-label-bundle", type=Path, default=None,
                    help="Audited exported-label bundle; no VLA model or RPC is used.")
parser.add_argument("--training-label-mode", choices=("stored", "online"), default=None,
                    help="Replay saved delta actions or evaluate their exact rule on the live state.")
parser.add_argument("--collect-full-trajectories", type=Path, default=None,
                    help="New output root for successful, causally aligned live-label rollouts.")
parser.add_argument("--collection-target-per-level", type=int, default=200)
parser.add_argument("--collection-max-attempts-per-level", type=int, default=800)
parser.add_argument("--collection-seed-offset", type=int, default=20261011)
parser.add_argument("--collection-initial-joint-std", type=float, default=0.025,
                    help="Small 7-joint reset variation, including otherwise-fixed Easy scenes.")
parser.add_argument("--collection-teacher", choices=("visual", "truth"), default="visual",
                    help="Truth is privileged expert-only supervision; student state stays visual.")
parser.add_argument("--execution-horizon", type=int, default=8)
parser.add_argument(
    "--vla-position-target-mode",
    choices=("absolute", "delta"),
    default="absolute",
    help=(
        "Interpret VLA xyz as an absolute target or current-EEF-relative "
        "delta. Delta chunks are decoded one action at a time at execution."
    ),
)
parser.add_argument(
    "--action-repeat",
    type=int,
    default=5,
    help="Hold each 10 Hz VLA action for this many 50 Hz Isaac control steps.",
)
parser.add_argument(
    "--condition-vla-on-visual-pose",
    action="store_true",
    help="Append RGB-D target xyz + validity, producing the 29-D policy state.",
)
parser.add_argument(
    "--condition-vla-on-controller-phase",
    action="store_true",
    help=(
        "Append the pre-action Scripted IK FSM phase to the visual state, "
        "producing the 30-D phase-conditioned policy state."
    ),
)
parser.add_argument(
    "--require-vla",
    action="store_true",
    help="Abort the benchmark immediately on a VLA RPC/server error.",
)
parser.add_argument(
    "--disable-scripted-fallback",
    "--pure-vla",
    dest="disable_scripted_fallback",
    action="store_true",
    help=(
        "Strict pure-VLA evaluation: if perception/action validation fails or VLA is unavailable, "
        "fail that episode without executing Scripted IK."
    ),
)
parser.add_argument(
    "--scripted-only",
    action="store_true",
    help=(
        "Visual-control baseline: execute only the RGB-D-conditioned Scripted IK "
        "action and do not contact the VLA server."
    ),
)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--realtime-speed", type=float, default=1.0)
parser.add_argument("--keep-open", action="store_true",
                    help="GUI only: write reports, then retain the final scene until the window is closed.")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=PROJECT_ROOT / "results/vla/closed_loop_benchmark",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.joint_target_actions:
    if args_cli.collect_full_trajectories:
        if args_cli.collection_teacher != "truth" or args_cli.vla_input_interface != "legacy":
            parser.error("Joint teacher collection requires truth supervision and legacy raw recorder")
    elif not args_cli.proprio_only:
        parser.error("Joint student evaluation requires --proprio-only; no target/phase inputs")
    from isaac_lab_data_engine.vla.joint_interface import validate_execution_settings
    try:
        validate_execution_settings(args_cli.execution_horizon, args_cli.action_repeat,
            teacher_or_replay=bool(args_cli.collect_full_trajectories or args_cli.training_label_mode))
    except ValueError as error:
        parser.error(str(error))
    if args_cli.training_label_mode not in (None, "stored"):
        parser.error("Joint target mode does not support old Cartesian label replay")
if args_cli.proprio_only:
    if (args_cli.vla_input_interface != "legacy" or args_cli.condition_vla_on_visual_pose
            or args_cli.condition_vla_on_controller_phase or args_cli.collect_full_trajectories
            or (args_cli.training_label_mode and not args_cli.joint_target_actions) or args_cli.scripted_only
            or args_cli.phase_mechanism != "legacy" or args_cli.student_phase_mode != "legacy"
            or args_cli.orientation_mode != "model" or not args_cli.disable_scripted_fallback):
        parser.error("--proprio-only requires pure VLA, model orientation and no target/phase inputs or expert")
if not 0 < args_cli.grasp_enable_distance_m <= 0.075:
    parser.error("--grasp-enable-distance-m must be in (0, 0.075]; cannot weaken the original guard")
if args_cli.vla_input_interface == "relative4d":
    args_cli.condition_vla_on_visual_pose = True
    args_cli.condition_vla_on_controller_phase = False
if not 0 <= args_cli.minimum_true_lift_m <= 0.5:
    parser.error("--minimum-true-lift-m must be in [0, 0.5]")
if (args_cli.training_label_bundle is None) != (args_cli.training_label_mode is None):
    parser.error("--training-label-bundle and --training-label-mode must be provided together")
if args_cli.collect_full_trajectories and args_cli.training_label_mode:
    parser.error("Collection samples new scenes; do not provide a replay bundle")
if args_cli.collection_teacher == "truth" and not args_cli.collect_full_trajectories:
    parser.error("Truth supervision is collection-only; VLA evaluation must remain unprivileged")
if not args_cli.proprio_only and (args_cli.record_phase_diagnostics or args_cli.phase_mechanism != "legacy" or args_cli.orientation_mode != "model"):
    if (args_cli.collect_full_trajectories or args_cli.training_label_mode or args_cli.scripted_only
            or not args_cli.disable_scripted_fallback
            or (not args_cli.condition_vla_on_controller_phase and args_cli.vla_input_interface != "relative4d")):
        parser.error("Phase/orientation ablations require phase-conditioned pure VLA evaluation, never collection or teacher replay")
if args_cli.training_label_mode or args_cli.collect_full_trajectories:
    if args_cli.scripted_only or args_cli.require_vla:
        parser.error("Training-label verification cannot use --scripted-only or --require-vla")
    if args_cli.execution_horizon != 1 or args_cli.vla_position_target_mode != "delta":
        parser.error("Training-label verification requires --execution-horizon 1 and delta mode")
    args_cli.disable_scripted_fallback = True
if args_cli.keep_open and args_cli.headless:
    parser.error("--keep-open requires a GUI; remove --headless")
if (
    args_cli.condition_vla_on_controller_phase
    and not args_cli.condition_vla_on_visual_pose
):
    parser.error(
        "--condition-vla-on-controller-phase requires --condition-vla-on-visual-pose"
    )
if args_cli.scripted_only and (
    args_cli.disable_scripted_fallback or args_cli.require_vla
):
    parser.error("--scripted-only cannot be combined with --pure-vla or --require-vla")
args_cli.enable_cameras = True

benchmark_cfg = load_benchmark_config(args_cli.config)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import numpy as np
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.benchmark.reporting import classify_failure
    from isaac_lab_data_engine.benchmark.vla_execution import select_benchmark_action
    from isaac_lab_data_engine.benchmark.vla_reporting import write_vla_benchmark_reports
    from isaac_lab_data_engine.controllers import PickAndLiftState, PickAndLiftStateMachine
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.randomization import DomainRandomizer, create_domain_randomizer
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
    from isaac_lab_data_engine.vla.label_replay import TrainingLabelReplay
    from isaac_lab_data_engine.vla.dataset import build_visual_fsm_bounded_actions
    from isaac_lab_data_engine.vla.full_trajectory import FORMAT, FullTrajectoryRecorder
    from isaac_lab_data_engine.vla.teacher import teacher_target, student_observation
    from isaac_lab_data_engine.vla.phase_diagnostics import FeedbackPhase, GripPhaseInput, override_orientation
    from isaac_lab_data_engine.vla.joint_interface import JointCommandBridge, JointTrajectoryRecorder
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _synchronize(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _camera_snapshot(env):
    data = env.scene["table_camera"].data
    return {"rgb": data.output["rgb"].clone(),
            "depth": data.output["distance_to_image_plane"].clone(),
            "intrinsics": data.intrinsic_matrices.clone(),
            "pos_w": data.pos_w.clone(), "quat_w_ros": data.quat_w_ros.clone(),
            "frame": int(env.scene["table_camera"].frame[0].cpu())}


def _workspace_pixel_mask(env: ManagerBasedRLEnv, snapshot=None) -> torch.Tensor:
    camera = env.scene["table_camera"]
    snapshot = snapshot or _camera_snapshot(env)
    depth = snapshot["depth"]
    if depth.ndim == 4:
        depth = depth[..., 0]
    intrinsics = snapshot["intrinsics"]
    _, height, width = depth.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=env.device, dtype=depth.dtype),
        torch.arange(width, device=env.device, dtype=depth.dtype),
        indexing="ij",
    )
    x_camera = (
        (x[None] - intrinsics[:, 0, 2, None, None])
        / intrinsics[:, 0, 0, None, None]
        * depth
    )
    y_camera = (
        (y[None] - intrinsics[:, 1, 2, None, None])
        / intrinsics[:, 1, 1, None, None]
        * depth
    )
    position_camera = torch.stack((x_camera, y_camera, depth), dim=-1)
    position_w = camera_to_world_position(
        position_camera,
        snapshot["pos_w"][:, None, None, :],
        snapshot["quat_w_ros"][:, None, None, :],
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
    env: ManagerBasedRLEnv,
    estimator: RGBDPoseEstimator,
    snapshot=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = env.scene["table_camera"]
    snapshot = snapshot or _camera_snapshot(env)
    position_camera = estimator.predict(
        snapshot["rgb"], snapshot["depth"], snapshot["intrinsics"],
        valid_pixel_mask=_workspace_pixel_mask(env, snapshot),
    )
    position_w = camera_to_world_position(
        position_camera,
        snapshot["pos_w"], snapshot["quat_w_ros"],
    )
    robot = env.scene["robot"]
    raw_position_b = world_to_root_position(
        position_w,
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )
    valid = (
        torch.isfinite(raw_position_b).all(dim=-1)
        # Use the same deployment workspace as the returned position.  The
        # previous wider validity bounds allowed gross predictions to be
        # silently clamped onto x/y boundaries and accepted as stable.
        & (raw_position_b[:, 0] >= 0.365)
        & (raw_position_b[:, 0] <= 0.775)
        & (raw_position_b[:, 1].abs() <= 0.295)
        & (raw_position_b[:, 2] >= -0.03)
        & (raw_position_b[:, 2] <= 0.15)
    )
    # Keep all policy/RPC tensors finite.  The validity bit still forces the
    # guard to reject this placeholder, so it cannot be mistaken for a valid
    # perception result.
    position_b = torch.empty_like(raw_position_b)
    position_b[:, 0] = torch.nan_to_num(raw_position_b[:, 0], nan=0.55).clamp(0.36, 0.78)
    position_b[:, 1] = torch.nan_to_num(raw_position_b[:, 1], nan=0.0).clamp(-0.30, 0.30)
    position_b[:, 2] = 0.03
    return position_b, valid


@torch.inference_mode()
def _initialize_visual_position(
    env: ManagerBasedRLEnv,
    estimator: RGBDPoseEstimator,
    cached_training_position: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Advance the randomized scene and latch one stable, static target estimate.

    A render-only refresh can leave a camera sensor on the previous episode's
    frame.  Advancing the environment makes the randomized camera/object state
    observable.  The policy then receives the median of a consecutive stable
    window, matching the cached visual-target semantics used for training.

    Label-only verification uses the actual cached training position, not new
    perception or simulator truth. In that mode RGB-D quality is diagnostic
    only: a fresh renderer/estimator must not gate a policy that never uses it.
    """

    camera = env.scene["table_camera"]
    env.sim.forward()
    camera.reset()
    ee_pose_b, _ = _poses_in_robot_frame(env)
    hold_action = torch.cat(
        (ee_pose_b, torch.ones((1, 1), device=env.device)), dim=-1
    )
    from isaac_lab_data_engine.vla.visual_initialization import ConsecutiveVisualWindow
    window = ConsecutiveVisualWindow(args_cli.visual_init_window, args_cli.visual_init_max_spread_m)
    snapshot = _camera_snapshot(env)
    last_finite = torch.tensor(
        [[0.55, 0.0, 0.03]], device=env.device, dtype=ee_pose_b.dtype
    )
    spread_m: float | None = None
    last_rgb_std: float | None = None
    last_rgb_range: float | None = None
    last_depth_fraction: float | None = None
    started = time.perf_counter()
    render_checks = 0
    warmup_step = 0
    invalid_pose_started = None

    def diagnostics(*, steps: int, stable: bool) -> dict[str, Any]:
        rgb = snapshot["rgb"][0, ..., :3]
        rgb_sha256 = hashlib.sha256(
            rgb.detach().cpu().numpy().tobytes()
        ).hexdigest()
        return {
            "steps": steps,
            "valid_frames": window.valid_frames,
            "max_consecutive_valid_frames": window.maximum_consecutive,
            "invalid_frames": window.invalid_frames,
            "duplicate_frames": window.duplicate_frames,
            "sampling_protocol": "one_coherent_snapshot_per_real_control_step_v2",
            "spread_m": spread_m,
            "stable": stable,
            "rgb_std": last_rgb_std,
            "rgb_range": last_rgb_range,
            "depth_valid_fraction": last_depth_fraction,
            "rgb_sha256": rgb_sha256,
            "camera_frame": snapshot["frame"],
            "render_checks": render_checks,
            "elapsed_s": time.perf_counter() - started,
            "position_source": (
                "actual_exported_training_visual_cache"
                if cached_training_position is not None else
                "camera_quality_only_no_target" if args_cli.proprio_only else "live_rgbd_estimator"
            ),
            "camera_quality_valid": (
                last_rgb_std is not None
                and last_rgb_std >= args_cli.camera_min_rgb_std
                and last_rgb_range >= args_cli.camera_min_rgb_range
                and (args_cli.proprio_only or last_depth_fraction >= args_cli.camera_min_depth_fraction)
            ),
        }

    while (warmup_step < args_cli.camera_warmup_max_steps
           or (cached_training_position is None and args_cli.camera_ready_timeout_s > 0.0
               and time.perf_counter() - started < args_cli.camera_ready_timeout_s)):
        render_checks += 1
        # Keep physics advancing even after the nominal warmup budget when a
        # wall-clock readiness wait was requested. Render-only, zero-dt forced
        # camera updates can alternate usable/empty annotator buffers forever.
        # This is initialization, not policy time: prevent a time-limit reset
        # from silently drawing a different physical scene during readiness.
        env.episode_length_buf.zero_()
        _, _, terminated, truncated, _ = env.step(hold_action)
        warmup_step += 1
        if bool(terminated[0]) or bool(truncated[0]):
            raise RuntimeError("Environment reset during camera warmup; sample provenance changed")
        for _ in range(args_cli.camera_refresh_renders):
            env.sim.render()
        if args_cli.camera_ready_timeout_s > 0.0 and args_cli.camera_render_yield_s:
            time.sleep(args_cli.camera_render_yield_s)
        # Read the annotators once after completed renders, never between
        # render passes. Quality, pose and checksum use the same immutable data.
        camera.update(env.step_dt, force_recompute=True)
        snapshot = _camera_snapshot(env)
        if warmup_step < args_cli.camera_warmup_min_steps:
            continue

        rgb = snapshot["rgb"][0, ..., :3].to(dtype=torch.float32)
        depth = snapshot["depth"][0].to(
            dtype=torch.float32
        )
        last_rgb_std = float(rgb.std(unbiased=False).cpu())
        last_rgb_range = float((rgb.max() - rgb.min()).cpu())
        last_depth_fraction = float(
            (torch.isfinite(depth) & (depth > 0.0)).float().mean().cpu()
        )
        quality_valid = (
            last_rgb_std >= args_cli.camera_min_rgb_std
            and last_rgb_range >= args_cli.camera_min_rgb_range
            and (args_cli.proprio_only or last_depth_fraction >= args_cli.camera_min_depth_fraction)
        )
        if not quality_valid and cached_training_position is None:
            invalid_pose_started = None
            window.observe(snapshot["frame"], [0, 0, 0], quality_valid=False, pose_valid=False)
            continue

        if args_cli.proprio_only:
            # Zero is a readiness-window placeholder, never a target or policy feature.
            estimate = torch.zeros((1, 3), device=env.device)
            valid = torch.ones((1,), dtype=torch.bool, device=env.device)
        elif cached_training_position is None:
            estimate, valid = _visual_position_b(env, estimator, snapshot)
        else:
            estimate = cached_training_position
            valid = torch.isfinite(estimate).all(dim=-1)
        last_finite = estimate
        if not bool(valid[0]):
            if invalid_pose_started is None:
                invalid_pose_started = time.perf_counter()
            if (args_cli.visual_invalid_pose_timeout_s > 0
                and time.perf_counter() - invalid_pose_started >= args_cli.visual_invalid_pose_timeout_s):
                break
            window.observe(snapshot["frame"], estimate[0].cpu().numpy(), quality_valid=quality_valid, pose_valid=False)
            continue
        invalid_pose_started = None

        ready = window.observe(snapshot["frame"], estimate[0].cpu().numpy(),
                               quality_valid=quality_valid or cached_training_position is not None,
                               pose_valid=True)
        spread_m = window.spread_m
        if ready:
            env.episode_length_buf.zero_()
            return (
                torch.as_tensor(window.median, device=env.device).unsqueeze(0),
                torch.ones((1,), dtype=torch.bool, device=env.device),
                diagnostics(steps=warmup_step, stable=True),
            )

    if window.median is not None:
        spread_m = window.spread_m
        visual_position = torch.as_tensor(window.median, device=env.device).unsqueeze(0)
    else:
        visual_position = last_finite
    env.episode_length_buf.zero_()
    return (
        visual_position,
        torch.zeros((1,), dtype=torch.bool, device=env.device),
        diagnostics(steps=warmup_step, stable=False),
    )


def _vla_request(
    env: ManagerBasedRLEnv,
    ee_pose_b: torch.Tensor,
    visual_position_b: torch.Tensor,
    perception_valid: torch.Tensor,
    controller_phase: int | None = None,
) -> VLARequest:
    robot = env.scene["robot"]
    camera_rgb = env.scene["table_camera"].data.output["rgb"][0, ..., :3]
    visual_args = (
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
        *visual_args,
        **state_kwargs,
    )
    if args_cli.vla_input_interface == "relative4d":
        from isaac_lab_data_engine.vla.relative_interface import relative_student_state
        state = relative_student_state(state)
    return VLARequest(
        rgb=camera_rgb.detach().cpu().numpy(),
        state=state,
        task=args_cli.task,
        input_interface="joint8" if args_cli.joint_target_actions else args_cli.vla_input_interface,
    )


def _latency(values: list[float], quantile: float) -> float | None:
    return float(np.quantile(values, quantile)) if values else None


@torch.inference_mode()
def _run_episode(
    env: ManagerBasedRLEnv,
    *,
    level: BenchmarkLevel,
    episode_index: int,
    randomizer: DomainRandomizer,
    estimator: RGBDPoseEstimator,
    client: VLAClient,
    max_steps: int,
) -> dict[str, Any]:
    label_replay = (
        TrainingLabelReplay(args_cli.training_label_bundle, level.name, args_cli.training_label_mode)
        if args_cli.training_label_mode else None
    )
    if label_replay is not None:
        if args_cli.joint_target_actions != (label_replay.semantics.get("interface") == "joint8"):
            raise ValueError("Saved labels and action execution interface disagree")
    collecting = args_cli.collect_full_trajectories is not None
    truth_teacher = collecting and args_cli.collection_teacher == "truth"
    label_control = collecting or label_replay is not None
    recorder = (JointTrajectoryRecorder() if args_cli.joint_target_actions else FullTrajectoryRecorder()) if collecting else None
    joint_bridge = getattr(env, "vla_joint_bridge", None)
    sampled_domain_params = (
        label_replay.domain if label_replay is not None else randomizer.sample_batch(1)[0]
    )
    domain_params: dict[str, Any] | None = None
    visual_position: torch.Tensor | None = None
    perception_valid: torch.Tensor | None = None
    visual_init: dict[str, Any] | None = None
    for visual_attempt in range(1, args_cli.visual_init_attempts + 1):
        if joint_bridge is not None:
            joint_bridge.reset()
        env.reset()
        # Reapply the exact same sampled parameters on retries so the fixed
        # benchmark sample key and its physical scene never change.
        domain_params = randomizer.apply_batch(env, [sampled_domain_params])[0]
        if collecting:
            robot = env.scene["robot"]
            q = robot.data.default_joint_pos.clone()
            reset_rng = np.random.default_rng(randomizer.seed + 31 * int(domain_params["sample_index"]))
            q[:, :7] += torch.as_tensor(
                reset_rng.normal(0, args_cli.collection_initial_joint_std, (1, 7)),
                device=env.device, dtype=q.dtype,
            )
            q = torch.clamp(q, robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            env.sim.forward()
            env.scene.update(env.physics_dt)
            domain_params["collection_initial_joint_position"] = q[0].cpu().tolist()
        if label_replay is not None:
            # Start at the actual first exported proprioceptive state, not a
            # newly randomized robot pose unrelated to the saved demonstration.
            robot = env.scene["robot"]
            q = torch.as_tensor(label_replay.states[0, :9], device=env.device).unsqueeze(0)
            dq = torch.as_tensor(label_replay.states[0, 9:18], device=env.device).unsqueeze(0)
            robot.write_joint_state_to_sim(q, dq)
            robot.set_joint_position_target(q)
            env.sim.forward()
            env.scene.update(env.physics_dt)
        cached_position = None
        if label_replay is not None:
            if not bool(label_replay.states[0, 28]):
                raise ValueError("The selected training demonstration has an invalid visual cache")
            cached_position = torch.as_tensor(
                label_replay.states[0, 25:28], device=env.device
            ).unsqueeze(0)
        visual_position, perception_valid, visual_init = _initialize_visual_position(
            env, estimator, cached_training_position=cached_position
        )
        visual_init["attempts"] = visual_attempt
        if bool(perception_valid[0]):
            break
    assert domain_params is not None
    assert visual_position is not None
    assert perception_valid is not None
    assert visual_init is not None
    if label_replay is not None:
        # This is the estimated RGB-D target actually stored in training state,
        # not simulator truth. Keep the exact original label target for replay.
        visual_position = torch.as_tensor(
            label_replay.states[0, 25:28], device=env.device
        ).unsqueeze(0)
    ee_pose_b, initial_true_object_pose_b = _poses_in_robot_frame(env)
    lift_anchor = (
        initial_true_object_pose_b[:, :3].clone() if truth_teacher else visual_position.clone()
    )
    initial_visual_error_m = None if args_cli.proprio_only else float(
        torch.linalg.vector_norm(visual_position[0] - initial_true_object_pose_b[0, :3]).cpu()
    )

    scripted = PickAndLiftStateMachine(env.step_dt, 1, env.device)
    feedback_phase = FeedbackPhase(env.step_dt) if args_cli.phase_mechanism == "feedback_hysteresis" else None
    student_phase_feedback = GripPhaseInput(env.step_dt) if args_cli.student_phase_mode == "grip_feedback" else None
    phase_trace = []
    phase_transitions = []
    gripper_blocked_steps = 0
    orientation_override_actions = 0
    raw_policy_action = None
    decoded_eef_anchor = None
    action_origin_phase = None
    guard = HybridActionGuard(maximum_position_step_m=0.04,
                              grasp_enable_distance_m=args_cli.grasp_enable_distance_m,
                              use_visual_grasp_gate=not args_cli.proprio_only)
    actions = torch.cat((ee_pose_b, torch.ones((1, 1), device=env.device)), dim=-1)
    action_queue: deque[np.ndarray] = deque()
    repeated_action: np.ndarray | None = None
    repeat_remaining = 0

    criteria = benchmark_cfg.success_criteria
    initial_height = env.scene["object"].data.root_pos_w[0, 2].clone()
    maximum_height = initial_height.clone()
    final_height = initial_height.clone()
    minimum_distance = torch.tensor(float("inf"), device=env.device)
    success_streak = 0
    success = False
    terminated_result = False
    truncated_result = False
    ever_grasp_closed = False
    ever_object_held = False
    ever_lifted = False
    total_reward = 0.0
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    vla_errors: list[str] = []
    inference_times_ms: list[float] = []
    rpc_roundtrip_times_ms: list[float] = []
    request_count = 0
    request_phase_counts: Counter[str] = Counter()
    pure_vla_failure: str | None = None
    step_count = 0
    label_trace: list[dict[str, Any]] = []
    stored_sequence_exhausted = False
    captured_phases: set[int] = set()
    labels_emitted = 0
    pending_record = None

    _synchronize(env.device)
    episode_started = time.perf_counter()
    for step_count in range(1, max_steps + 1):
        control_started = time.perf_counter()
        if recorder is not None:
            robot = env.scene["robot"]
            joint_kwargs = {}
            if joint_bridge is not None:
                joint_kwargs["joint_action"] = joint_bridge.held_action()
                if joint_kwargs["joint_action"] is None:
                    joint_kwargs["joint_action"] = np.r_[robot.data.joint_pos[0, :7].cpu().numpy(), 1.]
            recorder.physics_step(
                eef=ee_pose_b[0].cpu().numpy(), q=robot.data.joint_pos[0].cpu().numpy(),
                dq=robot.data.joint_vel[0].cpu().numpy(), action=actions[0].cpu().numpy(),
                object_position_w=env.scene["object"].data.root_pos_w[0].cpu().numpy(), step=step_count,
                **joint_kwargs,
            )
        _, reward, terminated, truncated, _ = env.step(actions)
        total_reward += float(reward[0].detach().cpu())
        ee_pose_b, true_object_pose_b = _poses_in_robot_frame(env)
        if args_cli.joint_target_actions and not collecting:
            actual_tcp = ee_pose_b[0, :3].cpu().numpy()
            if np.any(actual_tcp < np.array([.32, -.35, .015])) or np.any(actual_tcp > np.array([.82, .35, .60])):
                pure_vla_failure = "joint_workspace_violation"
                break

        object_position_w = env.scene["object"].data.root_pos_w
        ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        finger_position = env.scene["robot"].data.joint_pos[:, -2:].mean(dim=-1)
        current_height = object_position_w[0, 2]
        distance = torch.linalg.vector_norm(ee_position_w[0] - object_position_w[0])
        maximum_height = torch.maximum(maximum_height, current_height)
        final_height = current_height.clone()
        minimum_distance = torch.minimum(minimum_distance, distance)
        lifted = bool(
            current_height >= TABLE_TOP_Z + float(criteria["minimum_height_above_table_m"])
        )
        if label_control or args_cli.minimum_true_lift_m > 0:
            # Stronger than the legacy height-above-table check: require a
            # genuine 10 cm lift relative to this episode's initial height.
            required_lift = max(args_cli.minimum_true_lift_m, 0.10 if label_control else 0.0)
            lifted = lifted and bool(current_height - initial_height >= required_lift)
        gripper_closed = bool(
            finger_position[0] < float(criteria["maximum_mean_finger_position"])
        )
        object_held = bool(distance < float(criteria["maximum_ee_object_distance_m"]))
        ever_grasp_closed |= gripper_closed
        ever_object_held |= gripper_closed and object_held
        ever_lifted |= lifted
        if args_cli.progress_every > 0 and step_count % args_cli.progress_every == 0:
            print("VLA_BENCHMARK_PROGRESS=" + json.dumps({
                "difficulty": level.name, "episode": episode_index, "step": step_count,
                "max_steps": max_steps, "controller_phase": int(scripted.state[0].item()),
                "requests": request_count, "minimum_ee_object_distance_cm": float(minimum_distance.cpu() * 100.0),
                "maximum_lift_cm": float((maximum_height - initial_height).cpu() * 100.0),
                "wall_time_s": time.perf_counter() - episode_started}), flush=True)
        current_success = lifted and gripper_closed and object_held
        success_streak = success_streak + 1 if current_success else 0
        success = success_streak >= int(criteria["sustained_steps"])
        terminated_result = bool(terminated[0]) and not success
        truncated_result = bool(truncated[0]) and not success
        if success or terminated_result or truncated_result or step_count == max_steps:
            break

        if not bool(perception_valid[0]):
            pure_vla_failure = "invalid_visual_initialization"
            break

        predicted_object_pose = torch.zeros((1, 7), device=env.device)
        predicted_object_pose[:, :3] = (
            true_object_pose_b[:, :3] if truth_teacher else visual_position
        )
        predicted_object_pose[:, 3] = 1.0
        lift_pose = predicted_object_pose.clone()
        lift_pose[:, :3] = lift_anchor
        lift_pose[:, 2] += 0.18
        # This pre-compute phase is aligned with the action generated by the
        # current FSM state.  ``compute`` is allowed to transition afterwards.
        controller_phase = int(scripted.state[0].item())
        control_target = None if args_cli.proprio_only else teacher_target(
            mode="truth" if truth_teacher else "visual",
            visual_position=visual_position[0].cpu().numpy(),
            true_position=true_object_pose_b[0, :3].cpu().numpy(),
            lift_anchor=lift_anchor[0].cpu().numpy(), phase=controller_phase,
        )
        if args_cli.proprio_only:
            controller_phase = -1  # No phase is computed or supplied to the policy.
            fallback_action = torch.cat((ee_pose_b, torch.ones((1, 1), device=env.device)), dim=-1)
            phase_event = None
        elif feedback_phase is None:
            fallback_action = scripted.compute(ee_pose_b, predicted_object_pose, lift_pose)
            phase_event = None
        else:
            phase_event = feedback_phase.observe(
                ee_pose_b[0, :3].cpu().numpy(), visual_position[0].cpu().numpy(),
                env.scene["robot"].data.joint_pos[0, -2:].cpu().numpy(),
                env.scene["robot"].data.joint_vel[0, -2:].cpu().numpy(),
                guard.grasp_latched,
            )
            scripted.state.fill_(feedback_phase.phase)
            # Dummy hold is only the guard's rejection payload. Pure mode fails
            # rejection without ever executing it; no expert action is inserted.
            fallback_action = torch.cat((ee_pose_b, torch.ones((1, 1), device=env.device)), dim=-1)
        phase_after = -1 if args_cli.proprio_only else int(scripted.state[0].item())
        student_phase = controller_phase
        student_phase_event = None
        if student_phase_feedback is not None:
            student_phase, student_phase_event = student_phase_feedback.observe(
                controller_phase, env.scene["robot"].data.joint_pos[0, -2:].cpu().numpy(),
                env.scene["robot"].data.joint_vel[0, -2:].cpu().numpy(), guard.grasp_latched,
                float(torch.linalg.vector_norm(ee_pose_b[0, :3] - visual_position[0]).cpu()))
        if phase_after != controller_phase:
            phase_transitions.append({"control_step": step_count, "from": controller_phase, "to": phase_after,
                                      "reason": phase_event["reason"] if phase_event else "legacy_geometry_or_timer"})
        diagnostic_row = None
        if args_cli.record_phase_diagnostics:
            eef_xyz = ee_pose_b[0, :3].cpu().numpy()
            visual_xyz = visual_position[0].cpu().numpy()
            diagnostic_row = {
                "control_step": step_count, "phase_input": student_phase, "phase_after_update": phase_after,
                "legacy_phase_input": controller_phase, "student_phase_feedback": student_phase_event,
                "object_lift_m": float((current_height - initial_height).cpu()),
                "true_ee_object_distance_m": float(distance.cpu()),
                "object_held_metric": bool(object_held), "gripper_closed_metric": bool(gripper_closed),
                "eef_xyz_b": eef_xyz.tolist(), "visual_xyz_b": None if args_cli.proprio_only else visual_xyz.tolist(),
                "true_object_xyz_b": true_object_pose_b[0, :3].cpu().tolist(),
                "true_grasp_xy_distance_m": float(torch.linalg.vector_norm(ee_pose_b[0, :2] - true_object_pose_b[0, :2]).cpu()),
                "true_grasp_z_offset_m": float((ee_pose_b[0, 2] - true_object_pose_b[0, 2]).cpu()),
                "pregrasp_distance_m": None if args_cli.proprio_only else float(np.linalg.norm(eef_xyz - visual_xyz - np.array([0., 0., .12]))),
                "grasp_distance_m": None if args_cli.proprio_only else float(np.linalg.norm(eef_xyz - visual_xyz)),
                "finger_positions_m": env.scene["robot"].data.joint_pos[0, -2:].cpu().tolist(),
                "finger_velocities_m_s": env.scene["robot"].data.joint_vel[0, -2:].cpu().tolist(),
                "grasp_latched_before_guard": guard.grasp_latched, "feedback_event": phase_event,
            }
            phase_trace.append(diagnostic_row)
        if label_replay is not None and controller_phase not in captured_phases:
            _refresh_camera(env)
            from PIL import Image
            args_cli.output_dir.mkdir(parents=True, exist_ok=True)
            pixels = env.scene["table_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
            Image.fromarray(pixels.astype(np.uint8)).save(
                args_cli.output_dir / f"label_{level.name}_phase{controller_phase}.png"
            )
            captured_phases.add(controller_phase)

        if stored_sequence_exhausted:
            # Never repeatedly decode the last delta (which would introduce a
            # new cumulative action). Hold its last absolute IK target only.
            source_counts["training_label"] += 1
            reason_counts["stored_sequence_end_hold_absolute"] += 1
            continue

        if args_cli.scripted_only:
            reason_counts["scripted_only"] += 1
            source_counts["scripted_ik"] += 1
            actions = fallback_action
            if args_cli.realtime:
                remaining = env.step_dt / args_cli.realtime_speed - (
                    time.perf_counter() - control_started
                )
                if remaining > 0.0:
                    time.sleep(remaining)
            continue

        if not action_queue and repeat_remaining == 0:
            request_count += 1
            request_phase_counts[str(controller_phase)] += 1
            rpc_started = time.perf_counter()
            try:
                if label_control:
                    if label_replay is not None:
                        label_action = label_replay.next_action(
                            ee_pose_b[0].cpu().numpy(), visual_position[0].cpu().numpy(), controller_phase
                        )
                    else:
                        label_action = build_visual_fsm_bounded_actions(
                            ee_pose_b.cpu().numpy(), control_target[None],
                            np.asarray([controller_phase]), maximum_position_step_m=0.04,
                            approach_height_m=0.12, lift_height_m=0.18, position_target_mode="delta",
                        )[0]
                    if label_action is None:
                        stored_sequence_exhausted = True
                        source_counts["training_label"] += 1
                        reason_counts["stored_sequence_end_hold_absolute"] += 1
                        continue
                    labels_emitted += 1
                    index = labels_emitted - 1
                    trace = {
                        "control_step": step_count, "label_index": index,
                        "live_phase": controller_phase, "live_eef": ee_pose_b[0].cpu().tolist(),
                        "label_action": label_action.tolist(),
                    }
                    if label_replay is not None and label_replay.mode == "stored":
                        trace["recorded_phase"] = int(label_replay.states[index, 29])
                        trace["recorded_eef"] = label_replay.states[index, 18:25].tolist()
                        trace["eef_position_drift_cm"] = float(np.linalg.norm(
                            ee_pose_b[0].cpu().numpy()[:3] - label_replay.states[index, 18:21]
                        ) * 100)
                    if label_replay is not None:
                        label_trace.append(trace)
                    if recorder is not None:
                        # Render-only capture: no physics advance between this
                        # observation and the action about to be executed.
                        _refresh_camera(env)
                        camera = env.scene["table_camera"]
                        robot = env.scene["robot"]
                        state = student_observation(
                            q=robot.data.joint_pos[0].cpu().numpy(), dq=robot.data.joint_vel[0].cpu().numpy(),
                            eef=ee_pose_b[0].cpu().numpy(),
                            visual_position=visual_position[0].cpu().numpy(),
                            visual_validity=float(perception_valid[0]), phase=controller_phase,
                        )
                        pending_record = {
                            "state": state, "action": label_action.copy(), "step": step_count,
                            "teacher_control_position_b": control_target.copy(),
                            "true_object_position_b": true_object_pose_b[0, :3].cpu().numpy().copy(),
                            "rgb": camera.data.output["rgb"][0, ..., :3].cpu().numpy().copy(),
                            "depth": camera.data.output["distance_to_image_plane"][0].cpu().numpy().copy(),
                        }
                    action_queue.append(label_action)
                else:
                    response = client.predict(
                    _vla_request(
                        env,
                        ee_pose_b,
                        visual_position,
                        perception_valid,
                        student_phase,
                    )
                    )
                    inference_times_ms.append(float(response.inference_ms))
                    for item in response.actions[: args_cli.execution_horizon]:
                        action_queue.append(item.copy())
            except (ConnectionError, OSError, TimeoutError, RuntimeError, ValueError) as error:
                vla_errors.append(f"{type(error).__name__}: {error}")
                if args_cli.require_vla:
                    raise
            finally:
                if not label_control:
                    rpc_roundtrip_times_ms.append((time.perf_counter() - rpc_started) * 1000.0)

        if repeat_remaining == 0:
            queued_action = action_queue.popleft() if action_queue else None
            raw_policy_action = queued_action.copy() if queued_action is not None else None
            decoded_eef_anchor = ee_pose_b[0, :3].cpu().tolist() if queued_action is not None else None
            action_origin_phase = controller_phase if queued_action is not None else None
            if queued_action is not None and args_cli.vla_input_interface == "relative4d":
                # Also test the exact new execution interface with a truth
                # teacher pilot; fixed orientation applies to REST as well.
                queued_action[3:7] = (0., 1., 0., 0.)
            if queued_action is not None and args_cli.orientation_mode != "model":
                queued_action = override_orientation(queued_action, args_cli.orientation_mode, controller_phase)
                assert np.array_equal(queued_action[:3], raw_policy_action[:3])
                assert queued_action[7] == raw_policy_action[7]
                orientation_override_actions += int(controller_phase != 0)
            if args_cli.joint_target_actions and not collecting:
                joint_decision = joint_bridge.set_target(queued_action)
                repeated_action = joint_decision.action if joint_decision.source == "vla" else None
            else:
                repeated_action = decode_vla_position_target(
                    queued_action,
                    ee_pose_b[0].detach().cpu().numpy(),
                    position_target_mode=args_cli.vla_position_target_mode,
                )
            repeat_remaining = args_cli.action_repeat if repeated_action is not None else 1
        repeat_remaining -= 1

        if args_cli.joint_target_actions and not collecting:
            from isaac_lab_data_engine.vla.hybrid_controller import HybridActionDecision
            # Reuse the exact bounded setpoint, never reanchor it during its hold.
            decision = (joint_decision if repeated_action is not None else
                        HybridActionDecision(np.zeros(8, np.float32), "scripted_ik", "invalid_joint_action"))
        else:
            decision = guard.filter(
            repeated_action,
            fallback_action[0].detach().cpu().numpy(),
            current_eef_pose=ee_pose_b[0].detach().cpu().numpy(),
            visual_object_position=(
                true_object_pose_b[0, :3].detach().cpu().numpy()
                if truth_teacher else visual_position[0].detach().cpu().numpy()
            ),
            perception_valid=bool(perception_valid[0]),
            )
        reason_counts[decision.reason] += 1
        gripper_blocked_steps += int(decision.gripper_was_blocked)
        if diagnostic_row is not None:
            diagnostic_row.update({
                "raw_policy_action": raw_policy_action.tolist() if raw_policy_action is not None else None,
                "policy_request_count": request_count,
                "new_action_this_step": (step_count - 1) % args_cli.action_repeat == 0,
                "queued_actions_remaining": len(action_queue),
                "decoded_eef_anchor_xyz_b": decoded_eef_anchor, "action_origin_phase": action_origin_phase,
                "held_absolute_target": repeated_action.tolist() if repeated_action is not None else None,
                "filtered_action": decision.action.tolist(), "action_source": decision.source,
                "guard_reason": decision.reason, "gripper_was_blocked": decision.gripper_was_blocked,
                "position_was_clipped": decision.position_was_clipped,
                "grasp_latched_after_guard": joint_bridge.guard.grasp_latched if args_cli.joint_target_actions else guard.grasp_latched,
            })
        selection = select_benchmark_action(
            decision,
            scripted_fallback_enabled=not args_cli.disable_scripted_fallback,
        )
        source_counts["training_label" if label_control and selection.action is not None
                      else selection.source] += 1
        if selection.action is None:
            # No Scripted IK tensor is assigned to ``actions`` in pure mode.
            pure_vla_failure = selection.failure_reason
            break
        if args_cli.joint_target_actions and not collecting:
            actions = torch.cat((ee_pose_b, torch.full((1, 1), float(selection.action[7]), device=env.device)), dim=-1)
        else:
            actions = torch.as_tensor(selection.action, device=env.device).unsqueeze(0)
        if recorder is not None and pending_record is not None:
            if joint_bridge is not None:
                target_decision = joint_bridge.teacher_target(selection.action)
                if target_decision.source != "vla":
                    raise ValueError("Invalid teacher IK joint target")
                pending_record["joint_action"] = target_decision.action.copy()
                actions[:, 7] = float(target_decision.action[7])
            if truth_teacher:
                # Record the command actually sent to IK, including any guard
                # changes. Convert at execution time using THIS observation,
                # never by relabeling old trajectories afterwards.
                pending_record["teacher_intended_action"] = pending_record["action"].copy()
                executed_delta = np.asarray(selection.action, dtype=np.float32).copy()
                executed_delta[:3] -= pending_record["state"][18:21]
                pending_record["action"] = executed_delta
            recorder.decision(**pending_record, applied_absolute_action=selection.action)
            pending_record = None

        if args_cli.realtime:
            remaining = env.step_dt / args_cli.realtime_speed - (
                time.perf_counter() - control_started
            )
            if remaining > 0.0:
                time.sleep(remaining)

    _synchronize(env.device)
    wall_time_s = time.perf_counter() - episode_started
    physical_failure = classify_failure(
        success=success,
        terminated=terminated_result,
        truncated=truncated_result,
        ever_grasp_closed=ever_grasp_closed,
        ever_object_held=ever_object_held,
        ever_lifted=ever_lifted,
    )
    failure_reason = "none" if success else (pure_vla_failure or physical_failure)
    obj = domain_params["object"]
    vla_action_count = int(source_counts["vla"])
    scripted_action_count = int(source_counts["scripted_ik"])
    executed_decisions = vla_action_count + scripted_action_count
    sample_index = int(domain_params["sample_index"])
    if label_replay is not None:
        args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        _refresh_camera(env)
        from PIL import Image
        pixels = env.scene["table_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
        Image.fromarray(pixels.astype(np.uint8)).save(args_cli.output_dir / f"label_{level.name}_final.png")
        (args_cli.output_dir / f"label_{level.name}_trace.json").write_text(
            json.dumps({
                "mode": label_replay.mode, "source": label_replay.source,
                "exported_episode_index": label_replay.episode_id,
                "frames_available": len(label_replay.actions), "labels_executed": label_replay.cursor,
                "sequence_end_behavior": (
                    "hold_last_absolute_target_without_redecoding_delta"
                    if label_replay.mode == "stored" else "not_applicable_online_feedback_rule"
                ),
                "source_counts": dict(source_counts), "success": success,
                "actual_maximum_lift_cm": float((maximum_height - initial_height).cpu() * 100),
                "trace": label_trace,
            }, indent=2), encoding="utf-8")
    training_label_count = int(source_counts["training_label"])
    executed_decisions += training_label_count
    if args_cli.record_phase_diagnostics:
        trace_root = args_cli.output_dir / "phase_diagnostics"
        trace_root.mkdir(parents=True, exist_ok=True)
        (trace_root / f"{level.name}_{sample_index:06d}.json").write_text(json.dumps({
            "sample_key": f"{level.name}:{level.seed}:{sample_index:06d}",
            "phase_mechanism": args_cli.phase_mechanism, "orientation_mode": args_cli.orientation_mode,
            "feedback_protocol": feedback_phase.protocol() if feedback_phase else None,
            "gripper_blocked_control_steps": gripper_blocked_steps,
            "orientation_override_actions": orientation_override_actions,
            "phase_transitions": phase_transitions, "trace": phase_trace,
            "object_truth_in_feedback": False, "raw_xyz_and_gripper_preserved_by_orientation_override": True,
        }, indent=2), encoding="utf-8")
    return {
        "difficulty": level.name,
        "level_seed": level.seed,
        "episode_index": episode_index,
        "sample_index": sample_index,
        "sample_key": f"{level.name}:{level.seed}:{sample_index:06d}",
        "success": success,
        "_collection_recorder": recorder,
        "_collection_domain": domain_params if collecting else None,
        "sustained_success_steps": success_streak,
        "collection_teacher": args_cli.collection_teacher if collecting else None,
        "teacher_lift_anchor_b": lift_anchor[0].cpu().tolist() if collecting else None,
        "failure_reason": failure_reason,
        "failure_type": physical_failure,
        "steps": step_count,
        "completion_time_s": step_count * env.step_dt,
        "wall_time_s": wall_time_s,
        "total_reward": total_reward,
        "initial_object_height_m": float(initial_height.cpu()),
        "maximum_object_height_m": float(maximum_height.cpu()),
        "final_object_height_m": float(final_height.cpu()),
        "maximum_lift_cm": float((maximum_height - initial_height).cpu() * 100.0),
        "minimum_ee_object_distance_m": float(minimum_distance.cpu()),
        "final_controller_state": None if args_cli.proprio_only else int(scripted.state[0].cpu()),
        "ever_grasp_closed": ever_grasp_closed,
        "ever_object_held": ever_object_held,
        "ever_lifted": ever_lifted,
        "terminated": terminated_result,
        "truncated": truncated_result,
        "source_counts": dict(sorted(source_counts.items())),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "executed_action_decisions": executed_decisions,
        "vla_action_count": vla_action_count,
        "training_label_action_count": training_label_count,
        "training_label_source": label_replay.source if label_replay is not None else None,
        "scripted_ik_action_count": scripted_action_count,
        "scripted_ik_fallback_fraction": (
            scripted_action_count / executed_decisions if executed_decisions else 0.0
        ),
        "vla_request_count": request_count if not label_control else 0,
        "training_label_request_count": labels_emitted,
        "vla_request_phase_counts": (
            dict(sorted(request_phase_counts.items())) if not label_control else {}
        ),
        "training_label_request_phase_counts": (
            dict(sorted(request_phase_counts.items())) if label_control else {}
        ),
        "vla_error_count": len(vla_errors),
        "last_vla_error": vla_errors[-1] if vla_errors else None,
        "mean_vla_inference_ms": (
            float(np.mean(inference_times_ms)) if inference_times_ms else None
        ),
        "p95_vla_inference_ms": _latency(inference_times_ms, 0.95),
        "mean_rpc_roundtrip_ms": (
            float(np.mean(rpc_roundtrip_times_ms)) if rpc_roundtrip_times_ms else None
        ),
        "p95_rpc_roundtrip_ms": _latency(rpc_roundtrip_times_ms, 0.95),
        "_vla_inference_ms_samples": inference_times_ms,
        "_rpc_roundtrip_ms_samples": rpc_roundtrip_times_ms,
        "visual_object_x_m": None if args_cli.proprio_only else float(visual_position[0, 0].cpu()),
        "visual_object_y_m": None if args_cli.proprio_only else float(visual_position[0, 1].cpu()),
        "visual_object_z_m": None if args_cli.proprio_only else float(visual_position[0, 2].cpu()),
        "initial_visual_position_error_m": initial_visual_error_m,
        "visual_initialization_steps": int(visual_init["steps"]),
        "visual_initialization_attempts": int(visual_init["attempts"]),
        "visual_initialization_valid_frames": int(visual_init["valid_frames"]),
        "visual_initialization_max_consecutive_valid_frames": int(
            visual_init["max_consecutive_valid_frames"]
        ),
        "visual_initialization_spread_m": visual_init["spread_m"],
        "visual_initialization_stable": bool(visual_init["stable"]),
        "visual_initialization_position_source": visual_init["position_source"],
        "visual_initialization_camera_quality_valid": visual_init["camera_quality_valid"],
        "visual_initialization_rgb_std": visual_init["rgb_std"],
        "visual_initialization_rgb_range": visual_init["rgb_range"],
        "visual_initialization_depth_valid_fraction": visual_init[
            "depth_valid_fraction"
        ],
        "visual_initialization_rgb_sha256": visual_init["rgb_sha256"],
        "visual_initialization_camera_frame": int(visual_init["camera_frame"]),
        "visual_initialization_camera_pose_error_m": float(
            torch.linalg.vector_norm(
                env.scene["table_camera"].data.pos_w[0]
                - env.scene.env_origins[0]
                - torch.as_tensor(
                    domain_params["camera"]["position_m"], device=env.device
                )
            ).cpu()
        ),
        "object_x_m": float(obj["position_m"][0]),
        "object_y_m": float(obj["position_m"][1]),
        "object_yaw_deg": float(obj["yaw_deg"]),
        "object_mass_kg": float(obj["mass_kg"]),
        "static_friction": float(obj["static_friction"]),
        "dynamic_friction": float(obj["dynamic_friction"]),
        "restitution": float(obj["restitution"]),
        "camera_position_m": domain_params["camera"]["position_m"],
        "camera_target_m": domain_params["camera"]["target_m"],
        "lighting_intensity": float(domain_params["lighting"]["intensity"]),
    }


def _selected_levels() -> tuple[list[BenchmarkLevel], dict[str, int]]:
    configured = {level.name: level for level in benchmark_cfg.levels}
    names = args_cli.levels or list(configured)
    unknown = set(names) - set(configured)
    if unknown:
        raise ValueError(f"Unknown benchmark levels: {sorted(unknown)}")
    common_count = args_cli.episodes_per_level or benchmark_cfg.episodes_per_level
    counts: dict[str, int] = {}
    for name in names:
        override = getattr(args_cli, f"{name}_episodes", None)
        counts[name] = override if override is not None else common_count
        if counts[name] < 1:
            raise ValueError(f"Episode count for {name} must be positive")
    return [configured[name] for name in names], counts


def main() -> None:
    max_steps = args_cli.max_steps or benchmark_cfg.max_steps
    positive = (
        max_steps,
        args_cli.execution_horizon,
        args_cli.action_repeat,
    )
    if min(positive) < 1:
        raise ValueError("max-steps, execution-horizon and action-repeat must be positive")
    if args_cli.warmup_steps < 0:
        raise ValueError("warmup-steps cannot be negative")
    if args_cli.progress_every < 0:
        raise ValueError("progress-every cannot be negative")
    if (not np.isfinite(args_cli.camera_ready_timeout_s) or args_cli.camera_ready_timeout_s < 0.0
        or not np.isfinite(args_cli.camera_render_yield_s) or args_cli.camera_render_yield_s < 0.0):
        raise ValueError("Camera wall-clock wait and render yield must be finite and non-negative")
    if args_cli.camera_warmup_min_steps < 1:
        raise ValueError("camera-warmup-min-steps must be positive")
    if args_cli.camera_warmup_max_steps < args_cli.camera_warmup_min_steps:
        raise ValueError("camera-warmup-max-steps cannot be smaller than the minimum")
    if args_cli.visual_init_window < 1:
        raise ValueError("visual-init-window must be positive")
    if args_cli.visual_init_attempts < 1:
        raise ValueError("visual-init-attempts must be positive")
    available_visual_steps = (
        args_cli.camera_warmup_max_steps - args_cli.camera_warmup_min_steps + 1
    )
    if args_cli.visual_init_window > available_visual_steps:
        raise ValueError("visual-init-window exceeds the available camera warmup steps")
    if args_cli.visual_init_max_spread_m <= 0.0:
        raise ValueError("visual-init-max-spread-m must be positive")
    if args_cli.camera_refresh_renders < 1:
        raise ValueError("camera-refresh-renders must be positive")
    if args_cli.camera_min_rgb_std < 0.0 or args_cli.camera_min_rgb_range < 0.0:
        raise ValueError("camera RGB quality thresholds cannot be negative")
    if not 0.0 < args_cli.camera_min_depth_fraction <= 1.0:
        raise ValueError("camera-min-depth-fraction must be within (0, 1]")
    if args_cli.realtime_speed <= 0.0 or args_cli.vla_timeout <= 0.0:
        raise ValueError("realtime-speed and vla-timeout must be positive")
    if not args_cli.proprio_only and not args_cli.vision_checkpoint.is_file():
        raise FileNotFoundError(args_cli.vision_checkpoint)
    levels, episode_counts = _selected_levels()
    collection_counts = {}
    collection_summary = None
    if args_cli.collect_full_trajectories:
        root = args_cli.collect_full_trajectories
        if root.exists():
            raise FileExistsError(f"Refusing to overwrite collection: {root}")
        if (args_cli.action_repeat != 5 or args_cli.collection_target_per_level < 1
            or args_cli.collection_max_attempts_per_level < args_cli.collection_target_per_level
            or not 0 < args_cli.collection_initial_joint_std < 0.1):
            raise ValueError("Invalid full-trajectory collection settings")
        root.mkdir(parents=True)
        episode_counts = {level.name: args_cli.collection_max_attempts_per_level for level in levels}
        collection_counts = {level.name: 0 for level in levels}
        collection_summary = {
            "format": FORMAT, "status": "COLLECTING", "target_per_level": args_cli.collection_target_per_level,
            "successes": collection_counts, "attempts": {level.name: 0 for level in levels},
            "reject_reasons": {}, "control_hz": 50, "policy_hz": 10, "action_repeat": 5,
            "execution_horizon": 1, "maximum_position_step_m": 0.04,
            "minimum_true_lift_cm": 10, "seed_offset": args_cli.collection_seed_offset,
            "initial_joint_std": args_cli.collection_initial_joint_std,
            "vision_checkpoint_sha256": _sha256(args_cli.vision_checkpoint),
            "label_source": (
                "live simulator truth plus teacher feedback FSM; student visual input is separate"
                if args_cli.collection_teacher == "truth" else
                "live RGB-D estimate plus feedback FSM; never posthoc relabeled"
            ),
            "collection_teacher": args_cli.collection_teacher,
            "teacher_target_source": (
                "simulator_truth_root_frame" if args_cli.collection_teacher == "truth"
                else "latched_visual_estimate_root_frame"
            ),
            "student_input_target_source": "latched_visual_estimate_root_frame",
            "intended_student_interface": args_cli.vla_input_interface,
            "raw_recorder_state_dim": 30,
            "phase_free_projection_required": args_cli.vla_input_interface == "relative4d",
            "truth_coordinates_in_student_input": False,
        }
        if args_cli.joint_target_actions:
            collection_summary.update(format="vla_joint_execution_aligned_v23",
                label_source="live bounded seven-joint IK setpoint and gripper, directly executed and recorded",
                maximum_joint_step_rad=.2, intended_student_interface="joint8",
                phase_free_projection_required=True,
                student_input_target_source="not_in_student_input; raw visual diagnostics only",
                joint_setpoint_hold_control_steps=5)
    if args_cli.training_label_mode and any(count != 1 for count in episode_counts.values()):
        raise ValueError("Training-label replay requires exactly one exported example per level")

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.lazy_sensor_update = True
    env_cfg.sim.device = args_cli.device
    episode_capacity_steps = max(max_steps, args_cli.camera_warmup_max_steps) + 10
    env_cfg.episode_length_s = (
        episode_capacity_steps * env_cfg.sim.dt * env_cfg.decimation
    )
    physx_capacity = configure_parallel_physx_capacity(env_cfg, 1)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    if args_cli.joint_target_actions:
        env.vla_joint_bridge = JointCommandBridge(env)
    estimator = None if args_cli.proprio_only else RGBDPoseEstimator.load(args_cli.vision_checkpoint, env.device)
    client = VLAClient(args_cli.vla_host, args_cli.vla_port, args_cli.vla_timeout)
    all_rows: list[dict[str, Any]] = []
    seen_sample_keys: set[str] = set()

    try:
        env.reset()
        ee_pose_b, _ = _poses_in_robot_frame(env)
        warmup_action = torch.cat((ee_pose_b, torch.ones((1, 1), device=env.device)), dim=-1)
        for _ in range(args_cli.warmup_steps):
            env.step(warmup_action)
        _refresh_camera(env)

        for level in levels:
            # The DomainRandomizer owns an independent NumPy Generator seeded
            # by the immutable level config.  Global seeds cover any auxiliary
            # Torch/NumPy operations without restarting the Isaac process.
            np.random.seed(level.seed)
            torch.manual_seed(level.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(level.seed)
            randomizer = create_domain_randomizer(level.randomization_config)
            if collection_summary is not None:
                randomizer.reseed(level.seed + args_cli.collection_seed_offset)
            for episode_index in range(1, episode_counts[level.name] + 1):
                row = _run_episode(
                    env,
                    level=level,
                    episode_index=episode_index,
                    randomizer=randomizer,
                    estimator=estimator,
                    client=client,
                    max_steps=max_steps,
                )
                recorder = row.pop("_collection_recorder", None)
                collection_domain = row.pop("_collection_domain", None)
                if collection_summary is not None:
                    row["sample_key"] = f"{level.name}:{randomizer.seed}:{row['sample_index']:06d}"
                    row["level_seed"] = randomizer.seed
                    collection_summary["attempts"][level.name] = episode_index
                    rejection = row["failure_reason"]
                    if row["success"]:
                        metadata = {
                            "format": FORMAT, "level": level.name, "success": True,
                            "sample_key": row["sample_key"], "domain": collection_domain,
                            "maximum_lift_cm": row["maximum_lift_cm"],
                            "sustained_success_steps": row["sustained_success_steps"],
                            "model_requests": row["vla_request_count"],
                            "scripted_fallback_actions": row["scripted_ik_action_count"],
                            "metrics": row, "task": args_cli.task,
                            "teacher_mode": args_cli.collection_teacher,
                            "teacher_lift_anchor_b": row["teacher_lift_anchor_b"],
                            "student_visual_cache_b": [
                                row["visual_object_x_m"], row["visual_object_y_m"], row["visual_object_z_m"]
                            ],
                            "student_phase_source": "teacher_feedback_fsm_phase_id_not_target_coordinates",
                            "deployment_phase_source": "visual_feedback_fsm_no_ground_truth",
                        }
                        path = args_cli.collect_full_trajectories / level.name / f"episode_{collection_counts[level.name]:06d}.npz"
                        try:
                            recorder.save(path, metadata)
                        except ValueError as error:
                            rejection = f"trajectory_audit: {error}"
                        else:
                            collection_counts[level.name] += 1
                            rejection = "none"
                    if rejection != "none":
                        counts = collection_summary["reject_reasons"]
                        counts[rejection] = counts.get(rejection, 0) + 1
                    (args_cli.collect_full_trajectories / "collection_summary.json").write_text(
                        json.dumps(collection_summary, indent=2), encoding="utf-8",
                    )
                    print("VLA_FULL_COLLECTION_PROGRESS=" + json.dumps(collection_summary), flush=True)
                if row["sample_key"] in seen_sample_keys:
                    raise RuntimeError(f"Duplicate benchmark sample: {row['sample_key']}")
                seen_sample_keys.add(str(row["sample_key"]))
                all_rows.append(row)
                print(
                    "VLA_BENCHMARK_EPISODE="
                    + json.dumps(
                        {
                            "difficulty": row["difficulty"],
                            "episode": row["episode_index"],
                            "sample_index": row["sample_index"],
                            "success": row["success"],
                            "failure_reason": row["failure_reason"],
                            "fallback_fraction": row["scripted_ik_fallback_fraction"],
                            "mean_vla_inference_ms": row["mean_vla_inference_ms"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if (collection_summary is not None
                    and collection_counts[level.name] >= args_cli.collection_target_per_level):
                    break
    finally:
        # A completed GUI demonstration retains its final scene for inspection.
        # Failed/partial runs and the default benchmark still close immediately.
        if not args_cli.keep_open or len(all_rows) != sum(episode_counts.values()):
            env.close()

    if collection_summary is not None:
        complete = all(n >= args_cli.collection_target_per_level for n in collection_counts.values())
        collection_summary["status"] = "FULL_TRAJECTORY_COLLECTION_OK" if complete else "INCOMPLETE"
        (args_cli.collect_full_trajectories / "collection_summary.json").write_text(
            json.dumps(collection_summary, indent=2), encoding="utf-8",
        )
        if not complete:
            raise RuntimeError("Collection quota not reached; successful files preserved, training must not start")

    device = torch.device(args_cli.device)
    level_seeds = {
        level.name: level.seed + (args_cli.collection_seed_offset if collection_summary is not None else 0)
        for level in levels
    }
    config_snapshot = {
        "benchmark_version": benchmark_cfg.version,
        "config_file": str(benchmark_cfg.path),
        "config_sha256": _sha256(benchmark_cfg.path),
        "runner": "single_persistent_isaac_process_sequential_episodes",
        "controller_mode": (
            "execution_aligned_full_trajectory_label_oracle"
            if args_cli.collect_full_trajectories else
            f"training_labels_{args_cli.training_label_mode}_no_fallback"
            if args_cli.training_label_mode else
            "rgbd_visual_scripted_ik_only"
            if args_cli.scripted_only
            else (
                "pure_vla_no_scripted_fallback"
                if args_cli.disable_scripted_fallback
                else "vla_with_visual_scripted_ik_fallback"
            )
        ),
        "scripted_only": args_cli.scripted_only,
        "scripted_ik_fallback_enabled": (
            not args_cli.scripted_only and not args_cli.disable_scripted_fallback
        ),
        "require_vla": args_cli.require_vla,
        "task_instruction": args_cli.task,
        "training_label_verification": {
            "mode": args_cli.training_label_mode,
            "bundle": str(args_cli.training_label_bundle.resolve()),
            "bundle_sha256": _sha256(args_cli.training_label_bundle),
            "visual_target_source": "exported_estimated_rgbd_xyz_not_object_truth",
            "robot_initial_state": "first_exported_q_and_dq",
            "minimum_actual_lift_m": 0.10,
            "phase_source": "live_feedback_fsm",
            "stored_sequence_end_behavior": "hold_last_absolute_target",
        } if args_cli.training_label_mode else None,
        "vla_input_interface": args_cli.vla_input_interface,
        "proprio_only": args_cli.proprio_only,
        "external_target_used_for_control": not args_cli.proprio_only,
        "model_action_dimension": 4 if args_cli.vla_input_interface == "relative4d" else 8,
        "joint_target_actions": args_cli.joint_target_actions,
        "action_semantics": "absolute_arm_joint_radians_plus_gripper" if args_cli.joint_target_actions else "Cartesian_pose_plus_gripper",
        "student_arm_execution": "direct_PD_no_IK" if args_cli.joint_target_actions else "differential_IK",
        "student_phase_in_model_input": args_cli.condition_vla_on_controller_phase,
        "state_dimension": (
            None
            if args_cli.scripted_only
            else (
                30
                if args_cli.condition_vla_on_controller_phase
                else 29 if args_cli.condition_vla_on_visual_pose else 25
            )
        ),
        "condition_vla_on_visual_pose": args_cli.condition_vla_on_visual_pose,
        "condition_vla_on_controller_phase": args_cli.condition_vla_on_controller_phase,
        "vla_position_target_mode": args_cli.vla_position_target_mode,
        "vla_position_decode": (
            "current_eef_xyz_at_execution + predicted_delta_xyz"
            if args_cli.vla_position_target_mode == "delta"
            else None
        ),
        "controller_phase_source": (
            ("truth_teacher_fsm_training_only" if args_cli.collect_full_trajectories
             and args_cli.collection_teacher == "truth" else "visual_feedback_fsm")
            if args_cli.condition_vla_on_controller_phase
            else None
        ),
        "phase_mechanism": args_cli.phase_mechanism,
        "student_phase_mode": args_cli.student_phase_mode,
        "orientation_mode": args_cli.orientation_mode,
        "phase_diagnostics_recorded": args_cli.record_phase_diagnostics,
        "feedback_phase_protocol": FeedbackPhase().protocol() if args_cli.phase_mechanism != "legacy" else None,
        "object_truth_usage": (
            "teacher action/FSM/guard and diagnostic metrics; never student observation"
            if args_cli.collect_full_trajectories and args_cli.collection_teacher == "truth"
            else "success/failure metrics only"
        ),
        "collection_teacher": args_cli.collection_teacher if args_cli.collect_full_trajectories else None,
        "vla_server": (
            None if args_cli.scripted_only or args_cli.training_label_mode or args_cli.collect_full_trajectories
            else f"{args_cli.vla_host}:{args_cli.vla_port}"
        ),
        "execution_horizon": args_cli.execution_horizon,
        "grasp_enable_distance_m": None if args_cli.proprio_only else args_cli.grasp_enable_distance_m,
        "gripper_latch_mode": "hold_closed_until_episode_reset",
        "action_repeat": args_cli.action_repeat,
        "max_steps": max_steps,
        "minimum_actual_lift_m": max(
            args_cli.minimum_true_lift_m,
            0.10 if args_cli.collect_full_trajectories or args_cli.training_label_mode else 0.0,
        ),
        "warmup_steps": args_cli.warmup_steps,
        "camera_warmup_min_steps": args_cli.camera_warmup_min_steps,
        "camera_warmup_max_steps": args_cli.camera_warmup_max_steps,
        "camera_refresh_renders": args_cli.camera_refresh_renders,
        "camera_ready_timeout_s": args_cli.camera_ready_timeout_s,
        "camera_render_yield_s": args_cli.camera_render_yield_s,
        "visual_initialization": {
            "mode": (
                "actual_exported_visual_cache_after_fixed_physics_warmup"
                if args_cli.training_label_mode else
                "consecutive_rgb_quality_only" if args_cli.proprio_only else "static_median_of_consecutive_valid_frames"
            ),
            "fresh_camera_quality_is_a_gate": not bool(args_cli.training_label_mode),
            "window": args_cli.visual_init_window,
            "same_sample_attempts": args_cli.visual_init_attempts,
            "max_spread_m": args_cli.visual_init_max_spread_m,
            "minimum_rgb_std": args_cli.camera_min_rgb_std,
            "minimum_rgb_range": args_cli.camera_min_rgb_range,
            "minimum_depth_fraction": args_cli.camera_min_depth_fraction,
            "updated_during_control": False,
        },
        "episode_counts": episode_counts,
        "success_criteria": benchmark_cfg.success_criteria,
        "vision_checkpoint": None if args_cli.proprio_only else str(args_cli.vision_checkpoint.resolve()),
        "vision_checkpoint_sha256": None if args_cli.proprio_only else _sha256(args_cli.vision_checkpoint),
        "physx_gpu_total_aggregate_pairs_capacity": physx_capacity,
        "levels": [
            {
                "name": level.name,
                "seed": level.seed,
                "randomization_config": str(level.randomization_config),
                "randomization_config_sha256": _sha256(level.randomization_config),
                "episodes": episode_counts[level.name],
                "description": level.description,
            }
            for level in levels
        ],
        "hardware": {
            "device": args_cli.device,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "isaaclab": _distribution_version("isaaclab"),
            "isaacsim": _distribution_version("isaacsim"),
        },
    }
    files = write_vla_benchmark_reports(
        args_cli.output_dir,
        config_snapshot=config_snapshot,
        episode_rows=all_rows,
        level_seeds=level_seeds,
    )
    print(
        "VLA_BENCHMARK_SUMMARY="
        + json.dumps(
            {
                "status": "VLA_BENCHMARK_OK",
                "episodes": len(all_rows),
                "unique_samples": len(seen_sample_keys),
                **files,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args_cli.keep_open:
        print("VLA_DEMO_KEEP_OPEN: Reports saved. Close the Isaac window to exit.", flush=True)
        try:
            while simulation_app.is_running():
                # Render only: no further policy actions, physics steps, or resets.
                env.sim.render()
                time.sleep(0.02)
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
