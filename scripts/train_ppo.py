"""Train the optional Phase 11 PPO baseline with Isaac Lab's RSL-RL config."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import yaml

# This machine has no git executable on PATH. RSL-RL only uses GitPython to
# snapshot code state, so quiet discovery keeps training functional while the
# runner safely skips unavailable repositories.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Import RSL-RL (and therefore tensordict) before Kit loads its native plugin
# stack.  On Windows, importing tensordict for the first time after Kit starts
# can terminate the process inside a native extension instead of raising a
# catchable Python exception.
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=ROOT / "configs" / "ppo.yaml")
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument(
    "--num-steps-per-env",
    type=int,
    default=None,
    help="Override the rollout horizon from the project config.",
)
parser.add_argument(
    "--save-interval",
    type=int,
    default=None,
    help="Save an intermediate checkpoint every N PPO iterations.",
)
parser.add_argument("--run-name", default=None)
parser.add_argument(
    "--action-space",
    choices=("joint", "ik_abs"),
    default="joint",
    help="Use the official joint-position task or the scripted-controller-compatible IK action space.",
)
parser.add_argument(
    "--action-noise-std",
    type=float,
    default=None,
    help="Override the loaded PPO action standard deviation (IK default: 0.05).",
)
parser.add_argument(
    "--learning-rate",
    type=float,
    default=None,
    help="Override the official PPO learning rate (use a small value for BC fine-tuning).",
)
parser.add_argument(
    "--resume-checkpoint",
    type=Path,
    default=None,
    help="Resume optimizer and policy state; --iterations is the number of additional updates.",
)
parser.add_argument(
    "--load-optimizer",
    action="store_true",
    help="Restore optimizer state when resuming an actual PPO run. Leave disabled for BC/DAgger warm starts.",
)
parser.add_argument(
    "--bc-teacher-checkpoint",
    type=Path,
    default=None,
    help="Frozen BC/DAgger checkpoint used for the on-policy auxiliary behavior-cloning loss.",
)
parser.add_argument("--bc-loss-coef", type=float, default=0.05)
parser.add_argument("--bc-num-mini-batches", type=int, default=4)
parser.add_argument(
    "--learning-rate-schedule",
    choices=("adaptive", "fixed"),
    default=None,
    help="Override PPO's learning-rate schedule. Fixed is recommended for conservative BC fine-tuning.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

project_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    import torch

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import (
        LiftCubePPORunnerCfg,
    )

    from isaac_lab_data_engine.envs.pick_lift_env_cfg import (
        PickLiftEnvCfg,
        PickLiftIKPPOEnvCfg,
        PickLiftPPOEnvCfg,
        configure_parallel_physx_capacity,
    )
    from isaac_lab_data_engine.training import FrozenTeacherBCAuxiliary
except BaseException:
    traceback.print_exc()
    simulation_app.close()
    sys.exit(1)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    iterations = args.iterations or int(project_cfg["max_iterations"])
    num_envs = args.num_envs or int(project_cfg["num_envs"])
    if min(iterations, num_envs) < 1:
        raise ValueError("iterations and num_envs must be positive")
    run_name = args.run_name or f"franka_lift_{num_envs}env_{iterations}iter"
    output_dir = (args.config.parent / project_cfg["output_root"] / run_name).resolve()
    if output_dir.exists():
        raise FileExistsError(f"PPO run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(project_cfg["seed"])
    torch.manual_seed(seed)
    env_cfg = PickLiftPPOEnvCfg() if args.action_space == "joint" else PickLiftIKPPOEnvCfg()
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.table_camera = None
    env_cfg.scene.robot.spawn.semantic_tags = None
    env_cfg.scene.object.spawn.semantic_tags = None
    env_cfg.scene.table.spawn.semantic_tags = None
    env_cfg.sim.device = args.device
    physx_capacity = configure_parallel_physx_capacity(env_cfg, num_envs)

    agent_cfg = LiftCubePPORunnerCfg()
    agent_cfg.seed = seed
    num_steps_per_env = args.num_steps_per_env or int(project_cfg["num_steps_per_env"])
    if num_steps_per_env < 1:
        raise ValueError("num-steps-per-env must be positive")
    agent_cfg.num_steps_per_env = num_steps_per_env
    agent_cfg.max_iterations = iterations
    requested_save_interval = args.save_interval or int(project_cfg["save_interval"])
    if requested_save_interval < 1:
        raise ValueError("save-interval must be positive")
    agent_cfg.save_interval = min(requested_save_interval, max(iterations, 1))
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise ValueError("learning-rate must be positive")
        agent_cfg.algorithm.learning_rate = args.learning_rate
    if args.learning_rate_schedule is not None:
        agent_cfg.algorithm.schedule = args.learning_rate_schedule
    agent_dict = agent_cfg.to_dict()
    (output_dir / "agent_config.yaml").write_text(
        yaml.safe_dump(agent_dict, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "project_config.yaml").write_text(
        yaml.safe_dump(project_cfg, sort_keys=False), encoding="utf-8"
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=str(output_dir), device=args.device)
    start_iteration = 0
    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = args.resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(resume_checkpoint)
        runner.load(
            str(resume_checkpoint),
            load_optimizer=args.load_optimizer,
            map_location=args.device,
        )
        # RSL-RL checkpoints store the last completed zero-based iteration.
        # Advance once so a resumed run does not repeat that optimizer update.
        runner.current_learning_iteration += 1
        start_iteration = runner.current_learning_iteration
    # torch.optim.load_state_dict restores the checkpoint's parameter-group
    # learning rate. Reapply the explicit experiment value after loading so a
    # BC checkpoint cannot silently replace 1e-6 with its saved 1e-4 setting.
    runner.alg.learning_rate = float(agent_cfg.algorithm.learning_rate)
    for parameter_group in runner.alg.optimizer.param_groups:
        parameter_group["lr"] = runner.alg.learning_rate

    bc_teacher_checkpoint = None
    if args.bc_teacher_checkpoint is not None:
        if args.action_space != "ik_abs":
            raise ValueError("BC auxiliary loss currently requires --action-space ik_abs")
        if args.bc_loss_coef <= 0.0 or args.bc_num_mini_batches < 1:
            raise ValueError("BC loss coefficient and mini-batch count must be positive")
        bc_teacher_checkpoint = args.bc_teacher_checkpoint.resolve()
        if not bc_teacher_checkpoint.is_file():
            raise FileNotFoundError(bc_teacher_checkpoint)
        teacher_payload = torch.load(
            bc_teacher_checkpoint, map_location=args.device, weights_only=False
        )
        # deepcopy preserves the resolved observation groups and architecture;
        # loading the requested checkpoint then makes the teacher independent
        # of the PPO warm-start source.
        import copy

        teacher_policy = copy.deepcopy(runner.alg.policy)
        teacher_policy.load_state_dict(teacher_payload["model_state_dict"])
        bc_auxiliary = FrozenTeacherBCAuxiliary(
            teacher_policy,
            coefficient=args.bc_loss_coef,
            num_mini_batches=args.bc_num_mini_batches,
            max_grad_norm=agent_cfg.algorithm.max_grad_norm,
        )
        bc_auxiliary.wrap_update(runner.alg)
    action_noise_std = args.action_noise_std
    if action_noise_std is None and args.action_space == "ik_abs":
        action_noise_std = 0.05
    if action_noise_std is not None:
        if action_noise_std <= 0.0:
            raise ValueError("action-noise-std must be positive")
        with torch.no_grad():
            if hasattr(runner.alg.policy, "std"):
                runner.alg.policy.std.fill_(action_noise_std)
            elif hasattr(runner.alg.policy, "log_std"):
                runner.alg.policy.log_std.fill_(math.log(action_noise_std))
            else:
                raise AttributeError("Policy exposes neither std nor log_std")
    started = time.perf_counter()
    try:
        runner.learn(
            num_learning_iterations=iterations,
            # The IK policy observes an adaptive manipulation phase. Keep
            # episode starts synchronized with the reset robot/object state.
            init_at_random_ep_len=args.action_space == "joint",
        )
    finally:
        wrapped.close()
    wall_time = time.perf_counter() - started

    final_iteration = start_iteration + iterations - 1
    checkpoint = output_dir / f"model_{final_iteration}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RSL-RL did not create the expected checkpoint: {checkpoint}")
    summary = {
        "status": "PHASE11_PPO_TRAINING_OK",
        "run_name": run_name,
        "seed": seed,
        "num_envs": num_envs,
        "iterations": iterations,
        "start_iteration": start_iteration,
        "final_iteration": final_iteration,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "action_space": args.action_space,
        "action_noise_std": action_noise_std,
        "learning_rate": agent_cfg.algorithm.learning_rate,
        "effective_optimizer_learning_rate": runner.alg.optimizer.param_groups[0]["lr"],
        "learning_rate_schedule": agent_cfg.algorithm.schedule,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "save_interval": agent_cfg.save_interval,
        "bc_teacher_checkpoint": (
            str(bc_teacher_checkpoint) if bc_teacher_checkpoint is not None else None
        ),
        "bc_loss_coefficient": args.bc_loss_coef if bc_teacher_checkpoint else 0.0,
        "bc_num_mini_batches": args.bc_num_mini_batches if bc_teacher_checkpoint else 0,
        "training_transitions": num_envs * agent_cfg.num_steps_per_env * iterations,
        "wall_time_s": wall_time,
        "transitions_per_second": (
            num_envs * agent_cfg.num_steps_per_env * iterations / wall_time
        ),
        "physx_gpu_total_aggregate_pairs_capacity": physx_capacity,
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "torch": torch.__version__,
    }
    _atomic_json(output_dir / "training_summary.json", summary)
    print("PHASE11_TRAINING_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)


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
