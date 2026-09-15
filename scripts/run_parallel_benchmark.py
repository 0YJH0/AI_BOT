"""Benchmark vectorized Isaac Lab Pick-and-Lift rollout throughput."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Benchmark one Isaac Lab parallel environment count.")
parser.add_argument(
    "--num_envs", "--num-envs", dest="num_envs", type=int, required=True,
    help="Number of environments simulated in parallel.",
)
parser.add_argument("--max_steps", type=int, default=600)
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument(
    "--results_csv",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "results" / "parallel_rollout.csv",
)
parser.add_argument("--randomization_config", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The Phase 5 scaling experiment measures physics/control throughput. RTX
# camera cost is measured separately when image export is introduced.
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms

    from isaac_lab_data_engine.controllers import PickAndLiftStateMachine
    from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.randomization import DomainRandomizer
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


CSV_FIELDS = [
    "timestamp_utc",
    "device",
    "num_envs",
    "vector_steps",
    "physics_steps_per_env",
    "wall_time_s",
    "simulation_fps",
    "environment_steps_per_second",
    "episodes_per_minute",
    "success_rate",
    "average_completion_time_s",
    "gpu_memory_used_mb",
    "physx_gpu_total_aggregate_pairs_capacity",
    "domain_randomization",
]


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


def _initial_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_pose_b, _ = _task_poses(env)
    return torch.cat((ee_pose_b, torch.ones((env.num_envs, 1), device=env.device)), dim=-1)


def _write_latest_row(path: Path, row: dict[str, object]) -> None:
    """Keep the latest result for each environment count, sorted by scale."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
    existing = [item for item in existing if int(item["num_envs"]) != int(row["num_envs"])]
    existing.append({field: str(row[field]) for field in CSV_FIELDS})
    existing.sort(key=lambda item: int(item["num_envs"]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1")
    if args_cli.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if args_cli.warmup_steps < 0:
        raise ValueError("--warmup_steps cannot be negative")

    env_cfg = PickLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    physx_aggregate_pairs_capacity = configure_parallel_physx_capacity(
        env_cfg, args_cli.num_envs
    )
    env_cfg.scene.table_camera = None
    env_cfg.scene.robot.spawn.semantic_tags = None
    env_cfg.scene.object.spawn.semantic_tags = None
    env_cfg.scene.table.spawn.semantic_tags = None
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = (args_cli.max_steps + 10) * env_cfg.sim.dt * env_cfg.decimation
    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()
        with torch.no_grad():
            warmup_actions = _initial_actions(env)
            for _ in range(args_cli.warmup_steps):
                env.step(warmup_actions)

        env.reset()
        randomizer = (
            DomainRandomizer(args_cli.randomization_config)
            if args_cli.randomization_config is not None
            else None
        )
        if randomizer is not None:
            randomizer.apply_batch(env, randomizer.sample_batch(env.num_envs))

        state_machine = PickAndLiftStateMachine(env.step_dt, env.num_envs, env.device)
        actions = _initial_actions(env)
        success_streak = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        completion_steps = torch.full(
            (env.num_envs,), args_cli.max_steps, dtype=torch.int64, device=env.device
        )
        maximum_object_height = env.scene["object"].data.root_pos_w[:, 2].clone()
        vector_steps = 0

        torch.cuda.synchronize(env.device)
        start = time.perf_counter()
        with torch.no_grad():
            for vector_steps in range(1, args_cli.max_steps + 1):
                _, _, terminated, truncated, _ = env.step(actions)
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

                object_position_w = env.scene["object"].data.root_pos_w
                maximum_object_height = torch.maximum(
                    maximum_object_height, object_position_w[:, 2]
                )
                ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
                finger_position = env.scene["robot"].data.joint_pos[:, -2:]
                current_success = (
                    (object_position_w[:, 2] >= TABLE_TOP_Z + 0.10)
                    & (finger_position.mean(dim=-1) < 0.030)
                    & (torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1) < 0.08)
                )
                active = ~done
                next_streak = torch.where(
                    current_success, success_streak + 1, torch.zeros_like(success_streak)
                )
                success_streak = torch.where(active, next_streak, success_streak)
                newly_successful = active & (success_streak >= 10)
                newly_done = newly_successful | (active & (terminated | truncated))
                completion_steps[newly_done] = vector_steps
                success |= newly_successful
                done |= newly_done
                actions[done, :7] = ee_pose_b[done]
                actions[done, 7] = state_machine.OPEN_GRIPPER
                if bool(done.all()):
                    break
        torch.cuda.synchronize(env.device)
        wall_time_s = time.perf_counter() - start

        free_bytes, total_bytes = torch.cuda.mem_get_info(env.device)
        gpu_memory_used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0)
        success_rate = float(success.float().mean().item())
        successful_completion = completion_steps[success]
        average_completion_time_s = (
            float(successful_completion.float().mean().item() * env.step_dt)
            if successful_completion.numel()
            else float("nan")
        )
        row: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(env.device),
            "num_envs": env.num_envs,
            "vector_steps": vector_steps,
            "physics_steps_per_env": vector_steps * env_cfg.decimation,
            "wall_time_s": wall_time_s,
            "simulation_fps": vector_steps * env_cfg.decimation / wall_time_s,
            "environment_steps_per_second": env.num_envs * vector_steps / wall_time_s,
            "episodes_per_minute": env.num_envs * 60.0 / wall_time_s,
            "success_rate": success_rate,
            "average_completion_time_s": average_completion_time_s,
            "gpu_memory_used_mb": gpu_memory_used_mb,
            "physx_gpu_total_aggregate_pairs_capacity": physx_aggregate_pairs_capacity,
            "domain_randomization": randomizer is not None,
        }
        _write_latest_row(args_cli.results_csv.resolve(), row)
        summary = {
            "status": "PHASE5_BENCHMARK_OK",
            **row,
            "num_success": int(success.sum().item()),
            "num_failure": int((~success).sum().item()),
            "final_controller_state_counts": torch.bincount(
                state_machine.state, minlength=5
            ).detach().cpu().tolist(),
            "maximum_object_height_mean_m": float(maximum_object_height.mean().item()),
            "maximum_object_height_max_m": float(maximum_object_height.max().item()),
            "final_object_height_mean_m": float(
                env.scene["object"].data.root_pos_w[:, 2].mean().item()
            ),
            "results_csv": str(args_cli.results_csv.resolve()),
        }
        print("PHASE5_BENCHMARK=" + json.dumps(summary, ensure_ascii=False), flush=True)
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
