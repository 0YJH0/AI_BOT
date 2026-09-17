"""Evaluate a trained Phase 11 PPO checkpoint on the Phase 7 benchmark levels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# See train_ppo.py: tensordict must be loaded before Kit on this Windows host.
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=(
        ROOT
        / "results"
        / "ppo"
        / "jointpos_shaped_height_1024env_1000iter"
        / "model_500.pt"
    ),
)
parser.add_argument("--benchmark-config", type=Path, default=ROOT / "configs" / "benchmark.yaml")
parser.add_argument("--agent-config", type=Path, default=None)
parser.add_argument("--action-space", choices=("joint", "ik_abs"), default="joint")
parser.add_argument("--levels", nargs="+", default=None)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--episodes-per-level", type=int, default=256)
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "ppo_evaluation_shaped")
parser.add_argument(
    "--visual-checkpoint",
    type=Path,
    default=None,
    help="Replace privileged object terms with RGB-D predictions (IK 57-D policies only).",
)
parser.add_argument("--visual-camera-interval", type=int, default=5)
parser.add_argument("--visual-ema-alpha", type=float, default=0.45)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.visual_checkpoint is not None

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaac_lab_data_engine.benchmark.config import load_benchmark_config
    from isaac_lab_data_engine.benchmark.reporting import (
        classify_failure,
        summarize_level,
        write_benchmark_reports,
    )
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        PickLiftIKPPOEnvCfg,
        PickLiftPPOEnvCfg,
        configure_parallel_physx_capacity,
    )
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


def _refresh_camera(env: ManagerBasedRLEnv) -> None:
    camera = env.scene["table_camera"]
    for _ in range(3):
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)


def _workspace_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
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


@torch.no_grad()
def _visual_position_b(env: ManagerBasedRLEnv, estimator: RGBDPoseEstimator) -> torch.Tensor:
    camera = env.scene["table_camera"]
    position_camera = estimator.predict(
        camera.data.output["rgb"],
        camera.data.output["distance_to_image_plane"],
        camera.data.intrinsic_matrices,
        valid_pixel_mask=_workspace_mask(env),
    )
    position_w = camera_to_world_position(
        position_camera, camera.data.pos_w, camera.data.quat_w_ros
    )
    robot = env.scene["robot"]
    position_b = world_to_root_position(
        position_w, robot.data.root_pos_w, robot.data.root_quat_w
    )
    position_b[:, 0].clamp_(0.36, 0.78)
    position_b[:, 1].clamp_(-0.30, 0.30)
    position_b[:, 2] = 0.03
    return position_b


def _patch_visual_observations(
    observations: Any,
    adapter: VisualPolicyObservationAdapter,
    visual_position: torch.Tensor,
    elapsed: torch.Tensor,
) -> Any:
    """Patch either a legacy tensor or RSL-RL 3.x TensorDict policy group."""

    if isinstance(observations, torch.Tensor):
        return adapter.patch(
            observations, visual_position, elapsed_s=elapsed, apply_filter=False
        )
    try:
        policy_observation = observations["policy"]
    except (KeyError, TypeError) as error:
        raise TypeError(
            f"Unsupported RSL observation container: {type(observations).__name__}"
        ) from error
    observations["policy"] = adapter.patch(
        policy_observation, visual_position, elapsed_s=elapsed, apply_filter=False
    )
    return observations


def _run_batch(
    env: ManagerBasedRLEnv,
    wrapped: RslRlVecEnvWrapper,
    policy: Any,
    randomizer: Any,
    criteria: dict[str, float | int],
    max_steps: int,
    visual_estimator: RGBDPoseEstimator | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    wrapped.reset()
    domain_params = randomizer.apply_batch(env, randomizer.sample_batch(env.num_envs))
    observations = wrapped.get_observations()
    visual_adapter = None
    visual_position = None
    if visual_estimator is not None:
        _refresh_camera(env)
        visual_adapter = VisualPolicyObservationAdapter(ema_alpha=args.visual_ema_alpha)
        visual_position = visual_adapter.filter_position(_visual_position_b(env, visual_estimator))
        elapsed = env.episode_length_buf.float() * env.step_dt
        observations = _patch_visual_observations(
            observations, visual_adapter, visual_position, elapsed
        )

    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(done)
    terminated_result = torch.zeros_like(done)
    truncated_result = torch.zeros_like(done)
    success_streak = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    completion_steps = torch.full((env.num_envs,), max_steps, dtype=torch.int64, device=env.device)
    cumulative_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    initial_height = env.scene["object"].data.root_pos_w[:, 2].clone()
    maximum_height = initial_height.clone()
    minimum_distance = torch.full((env.num_envs,), float("inf"), device=env.device)
    ever_grasp_closed = torch.zeros_like(done)
    ever_object_held = torch.zeros_like(done)
    ever_lifted = torch.zeros_like(done)

    torch.cuda.synchronize(env.device)
    started = time.perf_counter()
    vector_steps = 0
    # Isaac Lab keeps state tensors across reset calls.  inference_mode would
    # permanently tag tensors created during a step and make the next reset's
    # in-place updates illegal, so no_grad is the correct boundary here.
    with torch.no_grad():
        for vector_steps in range(1, max_steps + 1):
            active = ~done
            actions = policy(observations)
            observations, rewards, dones, _ = wrapped.step(actions)
            if visual_estimator is not None:
                assert visual_adapter is not None and visual_position is not None
                # Fuse only unobstructed initial frames, then latch the static
                # tabletop target before the arm occludes it.
                if (
                    vector_steps <= 10
                    and vector_steps % args.visual_camera_interval == 0
                ):
                    visual_position = visual_adapter.filter_position(
                        _visual_position_b(env, visual_estimator)
                    )
                elapsed = env.episode_length_buf.float() * env.step_dt
                observations = _patch_visual_observations(
                    observations, visual_adapter, visual_position, elapsed
                )
            cumulative_reward += rewards * active

            object_position_w = env.scene["object"].data.root_pos_w
            ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            finger_mean = env.scene["robot"].data.joint_pos[:, -2:].mean(dim=-1)
            distance = torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1)
            lifted = object_position_w[:, 2] >= (
                TABLE_TOP_Z + float(criteria["minimum_height_above_table_m"])
            )
            gripper_closed = finger_mean < float(criteria["maximum_mean_finger_position"])
            object_held = distance < float(criteria["maximum_ee_object_distance_m"])

            maximum_height = torch.where(
                active, torch.maximum(maximum_height, object_position_w[:, 2]), maximum_height
            )
            minimum_distance = torch.where(active, torch.minimum(minimum_distance, distance), minimum_distance)
            ever_grasp_closed |= active & gripper_closed
            ever_object_held |= active & gripper_closed & object_held
            ever_lifted |= active & lifted

            current_success = active & lifted & gripper_closed & object_held
            success_streak = torch.where(
                current_success, success_streak + 1, torch.where(active, 0, success_streak)
            )
            newly_successful = active & (success_streak >= int(criteria["sustained_steps"]))
            newly_terminated = active & dones.bool() & ~newly_successful
            newly_timed_out = active & (vector_steps == max_steps) & ~newly_successful & ~newly_terminated
            newly_done = newly_successful | newly_terminated | newly_timed_out
            completion_steps[newly_done] = vector_steps
            success |= newly_successful
            terminated_result |= newly_terminated
            truncated_result |= newly_timed_out
            done |= newly_done
            if bool(done.all()):
                break

    torch.cuda.synchronize(env.device)
    wall_time_s = time.perf_counter() - started
    cpu = {
        "success": success.cpu().tolist(),
        "terminated": terminated_result.cpu().tolist(),
        "truncated": truncated_result.cpu().tolist(),
        "steps": completion_steps.cpu().tolist(),
        "reward": cumulative_reward.cpu().tolist(),
        "initial_height": initial_height.cpu().tolist(),
        "maximum_height": maximum_height.cpu().tolist(),
        "minimum_distance": minimum_distance.cpu().tolist(),
        "ever_grasp_closed": ever_grasp_closed.cpu().tolist(),
        "ever_object_held": ever_object_held.cpu().tolist(),
        "ever_lifted": ever_lifted.cpu().tolist(),
    }
    rows: list[dict[str, Any]] = []
    for index, params in enumerate(domain_params):
        failure_type = classify_failure(
            success=cpu["success"][index],
            terminated=cpu["terminated"][index],
            truncated=cpu["truncated"][index],
            ever_grasp_closed=cpu["ever_grasp_closed"][index],
            ever_object_held=cpu["ever_object_held"][index],
            ever_lifted=cpu["ever_lifted"][index],
        )
        obj = params["object"]
        rows.append(
            {
                "env_index": index,
                "sample_index": params["sample_index"],
                "success": cpu["success"][index],
                "failure_type": failure_type,
                "steps": cpu["steps"][index],
                "completion_time_s": cpu["steps"][index] * env.step_dt,
                "total_reward": cpu["reward"][index],
                "initial_object_height_m": cpu["initial_height"][index],
                "maximum_object_height_m": cpu["maximum_height"][index],
                "final_object_height_m": float(env.scene["object"].data.root_pos_w[index, 2].item()),
                "minimum_ee_object_distance_m": cpu["minimum_distance"][index],
                "final_controller_state": -1,
                "ever_object_held": cpu["ever_object_held"][index],
                "ever_lifted": cpu["ever_lifted"][index],
                "object_x_m": obj["position_m"][0],
                "object_y_m": obj["position_m"][1],
                "object_yaw_deg": obj["yaw_deg"],
                "object_mass_kg": obj["mass_kg"],
                "static_friction": obj["static_friction"],
                "dynamic_friction": obj["dynamic_friction"],
                "restitution": obj["restitution"],
                "camera_applied": params["camera"]["applied"],
                "camera_position_m": params["camera"]["position_m"],
                "camera_target_m": params["camera"]["target_m"],
                "lighting_intensity": params["lighting"]["intensity"],
            }
        )
    return rows, {
        "vector_steps": vector_steps,
        "simulated_environment_steps": env.num_envs * vector_steps,
        "wall_time_s": wall_time_s,
        "environment_steps_per_second": env.num_envs * vector_steps / wall_time_s,
    }


def _write_comparison(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    ik_path = ROOT / "results" / "benchmark" / "benchmark_results.csv"
    if not ik_path.is_file():
        return
    with ik_path.open("r", encoding="utf-8", newline="") as stream:
        ik = {row["difficulty"]: row for row in csv.DictReader(stream)}
    rows = []
    for summary in summaries:
        level = summary["difficulty"]
        ik_rate = float(ik[level]["success_rate"])
        ppo_rate = float(summary["success_rate"])
        rows.append(
            {
                "difficulty": level,
                "scripted_ik_success_rate": ik_rate,
                "ppo_success_rate": ppo_rate,
                "ppo_minus_scripted_ik_percentage_points": 100.0 * (ppo_rate - ik_rate),
            }
        )
    with (output_dir / "ppo_vs_scripted_ik.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if min(args.num_envs, args.episodes_per_level, args.max_steps) < 1:
        raise ValueError("num-envs, episodes-per-level, and max-steps must be positive")
    if args.visual_checkpoint is not None:
        if args.action_space != "ik_abs":
            raise ValueError("Visual observation replacement currently requires --action-space ik_abs")
        if args.visual_camera_interval < 1:
            raise ValueError("visual-camera-interval must be positive")
        if not args.visual_checkpoint.is_file():
            raise FileNotFoundError(args.visual_checkpoint)
    benchmark_cfg = load_benchmark_config(args.benchmark_config)
    levels_by_name = {level.name: level for level in benchmark_cfg.levels}
    selected_names = args.levels or list(levels_by_name)
    if unknown := set(selected_names) - set(levels_by_name):
        raise ValueError(f"Unknown benchmark levels: {sorted(unknown)}")

    agent_path = args.agent_config or checkpoint.parent / "agent_config.yaml"
    agent_cfg = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    env_cfg = PickLiftPPOEnvCfg() if args.action_space == "joint" else PickLiftIKPPOEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if args.visual_checkpoint is None:
        env_cfg.scene.table_camera = None
    env_cfg.scene.robot.spawn.semantic_tags = None
    env_cfg.scene.object.spawn.semantic_tags = None
    env_cfg.scene.table.spawn.semantic_tags = None
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = (args.max_steps + 10) * env_cfg.sim.dt * env_cfg.decimation
    capacity = configure_parallel_physx_capacity(env_cfg, args.num_envs)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions"))
    runner = OnPolicyRunner(wrapped, agent_cfg, log_dir=None, device=args.device)
    runner.load(str(checkpoint), load_optimizer=False, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    visual_estimator = (
        RGBDPoseEstimator.load(args.visual_checkpoint, args.device)
        if args.visual_checkpoint is not None
        else None
    )
    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    try:
        for level_name in selected_names:
            level = levels_by_name[level_name]
            randomizer = create_domain_randomizer(level.randomization_config)
            level_rows: list[dict[str, Any]] = []
            metrics: list[dict[str, float | int]] = []
            for batch_index in range(math.ceil(args.episodes_per_level / args.num_envs)):
                rows, batch_metrics = _run_batch(
                    env,
                    wrapped,
                    policy,
                    randomizer,
                    benchmark_cfg.success_criteria,
                    args.max_steps,
                    visual_estimator,
                )
                for row in rows[: args.episodes_per_level - len(level_rows)]:
                    row.update(
                        {
                            "difficulty": level.name,
                            "level_seed": level.seed,
                            "episode_index": len(level_rows) + 1,
                            "batch_index": batch_index,
                        }
                    )
                    level_rows.append(row)
                metrics.append(batch_metrics)
            summary = summarize_level(level.name, level.seed, level_rows, metrics)
            summary["description"] = level.description
            summaries.append(summary)
            all_rows.extend(level_rows)
            print(
                "PHASE11_LEVEL="
                + json.dumps(
                    {
                        "difficulty": level.name,
                        "episodes": summary["num_episodes"],
                        "success_rate": summary["success_rate"],
                        "failure_counts": summary["failure_counts"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        wrapped.close()

    output_dir = args.output_dir.resolve()
    files = write_benchmark_reports(
        output_dir,
        config_snapshot={
            "phase": 11,
            "policy": "rsl_rl_ppo",
            "action_space": args.action_space,
            "checkpoint": str(checkpoint),
            "benchmark_config": str(args.benchmark_config.resolve()),
            "agent_config": str(agent_path.resolve()),
            "device": args.device,
            "num_envs": args.num_envs,
            "episodes_per_level": args.episodes_per_level,
            "max_steps": args.max_steps,
            "physx_gpu_total_aggregate_pairs_capacity": capacity,
            "visual_checkpoint": (
                str(args.visual_checkpoint.resolve()) if args.visual_checkpoint else None
            ),
            "object_observation_source": "rgbd" if args.visual_checkpoint else "simulator_state",
            "success_criteria": benchmark_cfg.success_criteria,
        },
        summaries=summaries,
        episode_rows=all_rows,
    )
    _write_comparison(output_dir, summaries)
    report_path = output_dir / "benchmark_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "PHASE11_PPO_EVALUATION_OK"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "PHASE11_EVALUATION_SUMMARY="
        + json.dumps({"status": "PHASE11_PPO_EVALUATION_OK", "summaries": summaries, **files}),
        flush=True,
    )


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
