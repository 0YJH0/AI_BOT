"""Warm-start the IK-action PPO actor from online scripted demonstrations.

The script deliberately collects demonstrations inside the same vectorized
Isaac Lab environment used by PPO.  Consequently observations, action units,
domain randomization, and control timing are identical at BC and PPO time.
The resulting checkpoint is a normal RSL-RL checkpoint and can be passed to
``train_ppo.py --resume-checkpoint`` for on-policy fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import yaml

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Load tensordict/RSL-RL before Kit on Windows (see train_ppo.py).
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--benchmark-config", type=Path, default=ROOT / "configs" / "benchmark.yaml")
parser.add_argument("--ppo-config", type=Path, default=ROOT / "configs" / "ppo.yaml")
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--horizon", type=int, default=150)
parser.add_argument(
    "--lift-delta-m",
    type=float,
    default=0.15,
    help="Teacher end-effector lift target above each object's initial center height.",
)
parser.add_argument("--levels", nargs="+", default=("easy", "medium", "hard"))
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--validation-fraction", type=float, default=0.10)
parser.add_argument(
    "--dagger-rounds",
    type=int,
    default=0,
    help="Collect corrective labels on student-visited states after initial BC.",
)
parser.add_argument("--dagger-epochs", type=int, default=10)
parser.add_argument(
    "--dagger-initial-teacher-probability",
    type=float,
    default=0.5,
    help="Per-environment teacher execution probability in DAgger round one; linearly decays to zero.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output-dir", type=Path, default=ROOT / "results" / "ppo" / "bc_warmstart_ik_abs"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
        LiftCubePPORunnerCfg,
    )

    from isaac_lab_data_engine.benchmark.config import load_benchmark_config
    from isaac_lab_data_engine.envs import observations as task_observations
    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftIKPPOEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.randomization import create_domain_randomizer
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


def _adaptive_teacher_action(
    env: ManagerBasedRLEnv,
    ee_pose: torch.Tensor,
    object_pose: torch.Tensor,
    lift_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return adaptive teacher actions and the corresponding phase indices.

    The lift command uses the object's pre-grasp x/y anchor. It therefore
    raises the object in place instead of transporting it toward the fixed
    command position used by the upstream Isaac Lab lift task.
    """

    phase = task_observations.adaptive_manipulation_phase(env).argmax(dim=-1)
    top_down_quat = torch.tensor(
        (0.0, 1.0, 0.0, 0.0), dtype=torch.float32, device=env.device
    ).repeat(env.num_envs, 1)
    grasp_pose = torch.cat((object_pose[:, :3], top_down_quat), dim=-1)
    approach_pose = grasp_pose.clone()
    approach_pose[:, 2] += 0.12
    lift_pose = grasp_pose.clone()
    lift_pose[:, :3] = lift_position

    desired_pose = ee_pose.clone()
    desired_pose[phase == 1] = approach_pose[phase == 1]
    desired_pose[phase == 2] = grasp_pose[phase == 2]
    desired_pose[phase == 3] = grasp_pose[phase == 3]
    desired_pose[phase == 4] = lift_pose[phase == 4]
    gripper = torch.where(
        phase >= 3,
        -torch.ones(env.num_envs, device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )
    return torch.cat((desired_pose, gripper.unsqueeze(-1)), dim=-1), phase


def _collect_level(
    env: ManagerBasedRLEnv,
    wrapped: RslRlVecEnvWrapper,
    randomization_path: Path,
    horizon: int,
    lift_delta_m: float,
    actor: torch.nn.Module | None = None,
    teacher_probability: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, float | int]]:
    wrapped.reset()
    randomizer = create_domain_randomizer(randomization_path)
    randomizer.apply_batch(env, randomizer.sample_batch(env.num_envs))
    observations = wrapped.get_observations()
    ee_pose, object_pose = _task_poses(env)
    initial_object_position = object_pose[:, :3].clone()
    maximum_object_height = initial_object_position[:, 2].clone()
    maximum_object_xy_drift = torch.zeros(env.num_envs, device=env.device)
    xy_drift_at_maximum_height = torch.zeros(env.num_envs, device=env.device)
    lift_position = initial_object_position.clone()
    lift_position[:, 2] += lift_delta_m
    actions, phase = _adaptive_teacher_action(env, ee_pose, object_pose, lift_position)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    observation_batches: list[torch.Tensor] = []
    action_batches: list[torch.Tensor] = []
    grasp_state_batches: list[torch.Tensor] = []
    state_counts: Counter[int] = Counter()

    with torch.no_grad():
        for _ in range(horizon):
            observation_batches.append(observations["policy"][active].clone())
            action_batches.append(actions[active].clone())
            grasp_state_batches.append(
                task_observations.grasp_state_features(env)[active].clone()
            )
            state_counts.update(phase[active].cpu().tolist())

            executed_actions = actions
            if actor is not None:
                actor.eval()
                student_actions = actor(observations["policy"])
                teacher_mask = (
                    torch.rand((env.num_envs, 1), device=env.device) < teacher_probability
                )
                executed_actions = torch.where(teacher_mask, actions, student_actions)
            active_before_step = active.clone()
            observations, _, dones, _ = wrapped.step(executed_actions)
            ee_pose, object_pose = _task_poses(env)
            xy_drift = torch.linalg.vector_norm(
                object_pose[:, :2] - initial_object_position[:, :2], dim=-1
            )
            new_maximum = active_before_step & (
                object_pose[:, 2] > maximum_object_height
            )
            maximum_object_height = torch.where(
                new_maximum, object_pose[:, 2], maximum_object_height
            )
            xy_drift_at_maximum_height = torch.where(
                new_maximum, xy_drift, xy_drift_at_maximum_height
            )
            maximum_object_xy_drift = torch.where(
                active_before_step,
                torch.maximum(maximum_object_xy_drift, xy_drift),
                maximum_object_xy_drift,
            )
            active &= ~dones.bool()
            if not bool(active.any()):
                break

            actions, phase = _adaptive_teacher_action(env, ee_pose, object_pose, lift_position)

    grasp_states = torch.cat(grasp_state_batches, dim=0)
    diagnostics = {
        "minimum_xy_distance_m": float(grasp_states[:, 0].min().item()),
        "minimum_ee_object_distance_m": float(grasp_states[:, 2].min().item()),
        "minimum_mean_finger_position_m": float(grasp_states[:, 3].min().item()),
        "maximum_mean_finger_position_m": float(grasp_states[:, 3].max().item()),
        "maximum_mean_finger_speed_m_s": float(grasp_states[:, 4].max().item()),
        "held_sample_count": int(grasp_states[:, 5].sum().item()),
        "envs_reaching_formal_10cm": int(
            (
                maximum_object_height - initial_object_position[:, 2]
                >= 0.10
            ).sum().item()
        ),
        "mean_maximum_lift_m": float(
            (maximum_object_height - initial_object_position[:, 2]).mean().item()
        ),
        "maximum_xy_drift_m": float(maximum_object_xy_drift.max().item()),
        "mean_xy_drift_at_maximum_height_m": float(
            xy_drift_at_maximum_height.mean().item()
        ),
    }
    return (
        torch.cat(observation_batches, dim=0),
        torch.cat(action_batches, dim=0),
        {str(key): value for key, value in sorted(state_counts.items())},
        diagnostics,
    )


def _fit_actor(
    actor: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=observations.device)
    generator.manual_seed(seed)
    order = torch.randperm(observations.shape[0], generator=generator, device=observations.device)
    validation_size = max(1, int(observations.shape[0] * validation_fraction))
    validation_indices = order[:validation_size]
    training_indices = order[validation_size:]
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    component_weights = torch.tensor(
        (10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 3.0), device=observations.device
    )
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        permutation = training_indices[
            torch.randperm(training_indices.numel(), generator=generator, device=observations.device)
        ]
        actor.train()
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, permutation.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            prediction = actor(observations[indices])
            squared_error = (prediction - actions[indices]).square() * component_weights
            # Closing examples are rarer than open/approach examples; balance
            # them so the actor does not minimize loss by leaving the hand open.
            sample_weights = torch.where(actions[indices, 7] < 0.0, 3.0, 1.0)
            loss = (squared_error.mean(dim=-1) * sample_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * indices.numel()
            sample_count += indices.numel()

        actor.eval()
        with torch.no_grad():
            validation_prediction = actor(observations[validation_indices])
            absolute_error = (validation_prediction - actions[validation_indices]).abs()
            validation_mse = float(
                torch.mean((validation_prediction - actions[validation_indices]).square()).item()
            )
            gripper_accuracy = float(
                (
                    torch.sign(validation_prediction[:, 7])
                    == torch.sign(actions[validation_indices, 7])
                )
                .float()
                .mean()
                .item()
            )
        row = {
            "epoch": epoch,
            "training_weighted_mse": loss_sum / sample_count,
            "validation_mse": validation_mse,
            "validation_position_mae_m": float(absolute_error[:, :3].mean().item()),
            "validation_quaternion_mae": float(absolute_error[:, 3:7].mean().item()),
            "validation_gripper_mae": float(absolute_error[:, 7].mean().item()),
            "validation_gripper_sign_accuracy": gripper_accuracy,
        }
        history.append(row)
        print("BC_EPOCH=" + json.dumps(row), flush=True)
    return {
        "training_samples": int(training_indices.numel()),
        "validation_samples": int(validation_indices.numel()),
        "history": history,
        "final": history[-1],
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if min(args.num_envs, args.horizon, args.epochs, args.batch_size, args.dagger_epochs) < 1:
        raise ValueError("num-envs, horizon, epochs, and batch-size must be positive")
    if args.lift_delta_m <= 0.10:
        raise ValueError("lift-delta-m must exceed the formal 0.10 m lift threshold")
    if args.dagger_rounds < 0:
        raise ValueError("dagger-rounds cannot be negative")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")
    if not 0.0 <= args.dagger_initial_teacher_probability <= 1.0:
        raise ValueError("dagger-initial-teacher-probability must be between zero and one")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"BC output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    benchmark = load_benchmark_config(args.benchmark_config)
    levels_by_name = {level.name: level for level in benchmark.levels}
    if unknown := set(args.levels) - set(levels_by_name):
        raise ValueError(f"Unknown benchmark levels: {sorted(unknown)}")
    project_cfg = yaml.safe_load(args.ppo_config.read_text(encoding="utf-8"))
    torch.manual_seed(args.seed)

    env_cfg = PickLiftIKPPOEnvCfg()
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.scene.table_camera = None
    env_cfg.scene.robot.spawn.semantic_tags = None
    env_cfg.scene.object.spawn.semantic_tags = None
    env_cfg.scene.table.spawn.semantic_tags = None
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = (args.horizon + 10) * env_cfg.sim.dt * env_cfg.decimation
    capacity = configure_parallel_physx_capacity(env_cfg, args.num_envs)

    agent_cfg = LiftCubePPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.num_steps_per_env = int(project_cfg["num_steps_per_env"])
    agent_dict = agent_cfg.to_dict()
    (output_dir / "agent_config.yaml").write_text(
        yaml.safe_dump(agent_dict, sort_keys=False), encoding="utf-8"
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=str(output_dir), device=args.device)
    # ``learn()`` normally initializes this field. BC trains the actor
    # directly, so set the local logger type before using runner.save().
    runner.logger_type = "tensorboard"
    started = time.perf_counter()
    observation_batches: list[torch.Tensor] = []
    action_batches: list[torch.Tensor] = []
    state_counts: dict[str, dict[str, int]] = {}
    collection_diagnostics: dict[str, dict[str, float | int]] = {}
    try:
        for level_name in args.levels:
            level = levels_by_name[level_name]
            observations, actions, counts, diagnostics = _collect_level(
                env, wrapped, level.randomization_config, args.horizon, args.lift_delta_m
            )
            observation_batches.append(observations)
            action_batches.append(actions)
            state_counts[level_name] = counts
            collection_diagnostics[level_name] = diagnostics
            print(
                "BC_COLLECTION="
                + json.dumps(
                    {
                        "level": level_name,
                        "samples": observations.shape[0],
                        "state_counts": counts,
                        "diagnostics": diagnostics,
                    }
                ),
                flush=True,
            )

        observations = torch.cat(observation_batches, dim=0)
        actions = torch.cat(action_batches, dim=0)
        actor = runner.alg.policy.actor
        metrics = _fit_actor(
            actor,
            observations,
            actions,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.validation_fraction,
            args.seed,
        )
        dagger_metrics: list[dict[str, object]] = []
        for round_index in range(1, args.dagger_rounds + 1):
            if args.dagger_rounds == 1:
                teacher_probability = 0.0
            else:
                teacher_probability = args.dagger_initial_teacher_probability * (
                    1.0 - (round_index - 1) / (args.dagger_rounds - 1)
                )
            round_observations: list[torch.Tensor] = []
            round_actions: list[torch.Tensor] = []
            round_state_counts: dict[str, dict[str, int]] = {}
            for level_name in args.levels:
                level = levels_by_name[level_name]
                collected_observations, collected_actions, counts, diagnostics = _collect_level(
                    env,
                    wrapped,
                    level.randomization_config,
                    args.horizon,
                    args.lift_delta_m,
                    actor=actor,
                    teacher_probability=teacher_probability,
                )
                round_observations.append(collected_observations)
                round_actions.append(collected_actions)
                round_state_counts[level_name] = counts
            observations = torch.cat((observations, *round_observations), dim=0)
            actions = torch.cat((actions, *round_actions), dim=0)
            round_optimization = _fit_actor(
                actor,
                observations,
                actions,
                args.dagger_epochs,
                args.batch_size,
                args.learning_rate,
                args.validation_fraction,
                args.seed + round_index,
            )
            round_summary: dict[str, object] = {
                "round": round_index,
                "teacher_execution_probability": teacher_probability,
                "new_samples": int(sum(batch.shape[0] for batch in round_observations)),
                "aggregate_samples": int(observations.shape[0]),
                "state_counts": round_state_counts,
                "optimization": round_optimization,
            }
            dagger_metrics.append(round_summary)
            print("DAGGER_ROUND=" + json.dumps(round_summary, ensure_ascii=False), flush=True)
        checkpoint = output_dir / "model_bc.pt"
        summary: dict[str, object] = {
            "status": "BC_WARMSTART_OK",
            "action_space": "ik_abs",
            "seed": args.seed,
            "num_envs": args.num_envs,
            "levels": list(args.levels),
            "horizon": args.horizon,
            "lift_delta_m": args.lift_delta_m,
            "total_samples": int(observations.shape[0]),
            "observation_dimension": int(observations.shape[1]),
            "action_dimension": int(actions.shape[1]),
            "state_counts": state_counts,
            "collection_diagnostics": collection_diagnostics,
            "optimization": metrics,
            "dagger_rounds": dagger_metrics,
            "physx_gpu_total_aggregate_pairs_capacity": capacity,
            "wall_time_s": time.perf_counter() - started,
            "checkpoint": str(checkpoint),
        }
        runner.save(str(checkpoint), infos=summary)
        _atomic_json(output_dir / "bc_summary.json", summary)
        print("BC_WARMSTART_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        wrapped.close()


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
