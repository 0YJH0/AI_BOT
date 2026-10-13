"""Run the reproducible Phase 7 Easy/Medium/Hard online benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

from isaac_lab_data_engine.benchmark.config import load_benchmark_config


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--config",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml",
)
parser.add_argument("--levels", nargs="+", default=None, help="Subset of configured levels to run.")
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=None)
parser.add_argument("--episodes_per_level", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--output_dir", type=Path, default=None)
parser.add_argument(
    "--visual-checkpoint",
    type=Path,
    default=None,
    help="Use RGB-D XYZ estimates instead of simulator object state for Scripted IK.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = args_cli.visual_checkpoint is not None

benchmark_cfg = load_benchmark_config(args_cli.config)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv

    from isaac_lab_data_engine.benchmark.reporting import summarize_level, write_benchmark_reports
    from isaac_lab_data_engine.benchmark.runner import VectorizedBenchmarkRunner
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
def _visual_position_b(
    env: ManagerBasedRLEnv, estimator: RGBDPoseEstimator
) -> torch.Tensor:
    _refresh_camera(env)
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


def main() -> None:
    num_envs = args_cli.num_envs or benchmark_cfg.num_envs
    episodes_per_level = args_cli.episodes_per_level or benchmark_cfg.episodes_per_level
    max_steps = args_cli.max_steps or benchmark_cfg.max_steps
    output_dir = (args_cli.output_dir or benchmark_cfg.output_dir).resolve()
    if min(num_envs, episodes_per_level, max_steps) < 1:
        raise ValueError("num_envs, episodes_per_level, and max_steps must be positive")
    if args_cli.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if args_cli.visual_checkpoint is not None and not args_cli.visual_checkpoint.is_file():
        raise FileNotFoundError(args_cli.visual_checkpoint)

    configured_levels = {level.name: level for level in benchmark_cfg.levels}
    selected_names = args_cli.levels or list(configured_levels)
    unknown = set(selected_names) - set(configured_levels)
    if unknown:
        raise ValueError(f"Unknown benchmark levels: {sorted(unknown)}")
    levels = [configured_levels[name] for name in selected_names]

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = num_envs
    physx_capacity = configure_parallel_physx_capacity(env_cfg, num_envs)
    if args_cli.visual_checkpoint is None:
        env_cfg.scene.table_camera = None
    env_cfg.scene.robot.spawn.semantic_tags = None
    env_cfg.scene.object.spawn.semantic_tags = None
    env_cfg.scene.table.spawn.semantic_tags = None
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = (max_steps + 10) * env_cfg.sim.dt * env_cfg.decimation
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = torch.device(env.device)
    hardware = {
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
        ),
    }
    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "isaaclab": _distribution_version("isaaclab"),
        "isaacsim": _distribution_version("isaacsim"),
    }
    visual_estimator = (
        RGBDPoseEstimator.load(args_cli.visual_checkpoint, env.device)
        if args_cli.visual_checkpoint is not None
        else None
    )
    position_provider = (
        (lambda current_env: _visual_position_b(current_env, visual_estimator))
        if visual_estimator is not None
        else None
    )
    runner = VectorizedBenchmarkRunner(
        env,
        benchmark_cfg.success_criteria,
        object_position_provider=position_provider,
    )
    summaries: list[dict[str, object]] = []
    all_episode_rows: list[dict[str, object]] = []

    try:
        # Compile kernels and fill runtime caches before the first measured
        # difficulty so Easy does not absorb one-time initialization cost.
        env.reset()
        with torch.no_grad():
            ee_pose_b, _ = runner._task_poses()
            warmup_actions = torch.cat(
                (ee_pose_b, torch.ones((env.num_envs, 1), device=env.device)), dim=-1
            )
            for _ in range(args_cli.warmup_steps):
                env.step(warmup_actions)

        for level in levels:
            randomizer = create_domain_randomizer(level.randomization_config)
            level_rows: list[dict[str, object]] = []
            level_metrics: list[dict[str, float | int]] = []
            batches = math.ceil(episodes_per_level / num_envs)
            for batch_index in range(batches):
                batch_rows, metrics = runner.run_batch(randomizer, max_steps)
                remaining = episodes_per_level - len(level_rows)
                selected_rows = batch_rows[:remaining]
                for row in selected_rows:
                    row.update(
                        {
                            "difficulty": level.name,
                            "level_seed": level.seed,
                            "episode_index": len(level_rows) + 1,
                            "batch_index": batch_index,
                        }
                    )
                    level_rows.append(row)
                level_metrics.append(metrics)

            summary = summarize_level(level.name, level.seed, level_rows, level_metrics)
            summary["description"] = level.description
            summaries.append(summary)
            all_episode_rows.extend(level_rows)
            print(
                "PHASE7_LEVEL="
                + json.dumps(
                    {
                        "difficulty": level.name,
                        "episodes": summary["num_episodes"],
                        "success_rate": summary["success_rate"],
                        "average_completion_time_s": summary["average_completion_time_s"],
                        "average_episode_reward": summary["average_episode_reward"],
                        "failure_counts": summary["failure_counts"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        env.close()

    config_snapshot = {
        "benchmark_version": benchmark_cfg.version,
        "config_file": str(benchmark_cfg.path),
        "config_sha256": _sha256(benchmark_cfg.path),
        "task": benchmark_cfg.task,
        "controller": (
            "rgbd_xyz_plus_scripted_absolute_differential_ik"
            if visual_estimator is not None
            else benchmark_cfg.controller
        ),
        "device": args_cli.device,
        "num_envs": num_envs,
        "episodes_per_level": episodes_per_level,
        "max_steps": max_steps,
        "warmup_steps": args_cli.warmup_steps,
        "physx_gpu_total_aggregate_pairs_capacity": physx_capacity,
        "success_criteria": benchmark_cfg.success_criteria,
        "levels": [
            {
                "name": level.name,
                "seed": level.seed,
                "randomization_config": str(level.randomization_config),
                "randomization_config_sha256": _sha256(level.randomization_config),
                "description": level.description,
            }
            for level in levels
        ],
        "camera_enabled": visual_estimator is not None,
        "visual_checkpoint": (
            str(args_cli.visual_checkpoint.resolve()) if args_cli.visual_checkpoint else None
        ),
        "object_observation_source": (
            "rgbd_latched_xyz" if visual_estimator is not None else "simulator_state"
        ),
        "camera_note": (
            "RGB-D is rendered once per episode batch before arm motion; images are not saved."
            if visual_estimator is not None
            else "Camera parameters are sampled but not instantiated for the state-based baseline."
        ),
        "hardware": hardware,
        "software": software,
    }
    files = write_benchmark_reports(
        output_dir,
        config_snapshot=config_snapshot,
        summaries=summaries,
        episode_rows=all_episode_rows,
    )
    final_summary = {
        "status": "PHASE7_BENCHMARK_OK",
        "levels": [
            {
                "difficulty": summary["difficulty"],
                "success_rate": summary["success_rate"],
                "num_episodes": summary["num_episodes"],
            }
            for summary in summaries
        ],
        "num_episode_results": len(all_episode_rows),
        **files,
    }
    print("PHASE7_SUMMARY=" + json.dumps(final_summary, ensure_ascii=False), flush=True)


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
