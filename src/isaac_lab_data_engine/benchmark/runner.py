"""GPU-vectorized online evaluator for the scripted Pick-and-Lift baseline."""

from __future__ import annotations

import time
from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import subtract_frame_transforms

from isaac_lab_data_engine.controllers import PickAndLiftStateMachine
from isaac_lab_data_engine.envs.phase1_scene import TABLE_TOP_Z
from isaac_lab_data_engine.randomization import DomainRandomizer
from isaac_lab_data_engine.benchmark.reporting import classify_failure


class VectorizedBenchmarkRunner:
    def __init__(self, env: ManagerBasedRLEnv, criteria: dict[str, float | int]) -> None:
        self.env = env
        self.criteria = criteria

    def _task_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.env.scene["robot"]
        ee_frame = self.env.scene["ee_frame"]
        target_object = self.env.scene["object"]
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

    def run_batch(
        self,
        randomizer: DomainRandomizer,
        max_steps: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
        env = self.env
        env.reset()
        domain_params = randomizer.apply_batch(env, randomizer.sample_batch(env.num_envs))
        controller = PickAndLiftStateMachine(env.step_dt, env.num_envs, env.device)
        ee_pose_b, _ = self._task_poses()
        actions = torch.cat(
            (ee_pose_b, torch.ones((env.num_envs, 1), device=env.device)), dim=-1
        )

        success_streak = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        terminated_result = torch.zeros_like(done)
        truncated_result = torch.zeros_like(done)
        completion_steps = torch.full(
            (env.num_envs,), max_steps, dtype=torch.int64, device=env.device
        )
        cumulative_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        initial_height = env.scene["object"].data.root_pos_w[:, 2].clone()
        maximum_height = initial_height.clone()
        final_height = initial_height.clone()
        minimum_distance = torch.full(
            (env.num_envs,), float("inf"), dtype=torch.float32, device=env.device
        )
        ever_grasp_closed = torch.zeros_like(done)
        ever_object_held = torch.zeros_like(done)
        ever_lifted = torch.zeros_like(done)
        final_state = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
        vector_steps = 0

        torch.cuda.synchronize(env.device)
        started = time.perf_counter()
        with torch.no_grad():
            for vector_steps in range(1, max_steps + 1):
                active = ~done
                _, reward, terminated, truncated, _ = env.step(actions)
                cumulative_reward += reward * active

                ee_pose_b, object_pose_b = self._task_poses()
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
                actions = controller.compute(ee_pose_b, object_pose_b, lift_pose_b)

                object_position_w = env.scene["object"].data.root_pos_w
                ee_position_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
                finger_mean = env.scene["robot"].data.joint_pos[:, -2:].mean(dim=-1)
                distance = torch.linalg.vector_norm(ee_position_w - object_position_w, dim=-1)
                lifted = object_position_w[:, 2] >= (
                    TABLE_TOP_Z + float(self.criteria["minimum_height_above_table_m"])
                )
                gripper_closed = finger_mean < float(
                    self.criteria["maximum_mean_finger_position"]
                )
                object_held = distance < float(
                    self.criteria["maximum_ee_object_distance_m"]
                )

                maximum_height = torch.where(
                    active, torch.maximum(maximum_height, object_position_w[:, 2]), maximum_height
                )
                minimum_distance = torch.where(
                    active, torch.minimum(minimum_distance, distance), minimum_distance
                )
                ever_grasp_closed |= active & (
                    controller.state >= 3
                )
                ever_object_held |= active & gripper_closed & object_held
                ever_lifted |= active & lifted

                current_success = lifted & gripper_closed & object_held
                next_streak = torch.where(
                    current_success, success_streak + 1, torch.zeros_like(success_streak)
                )
                success_streak = torch.where(active, next_streak, success_streak)
                newly_successful = active & (
                    success_streak >= int(self.criteria["sustained_steps"])
                )
                newly_terminated = active & terminated & ~newly_successful
                newly_truncated = active & truncated & ~newly_successful & ~newly_terminated
                newly_done = newly_successful | newly_terminated | newly_truncated
                if bool(newly_done.any()):
                    completion_steps[newly_done] = vector_steps
                    final_height[newly_done] = object_position_w[newly_done, 2]
                    final_state[newly_done] = controller.state[newly_done]
                    success |= newly_successful
                    terminated_result |= newly_terminated
                    truncated_result |= newly_truncated
                    done |= newly_done

                actions[done, :7] = ee_pose_b[done]
                actions[done, 7] = controller.OPEN_GRIPPER
                if bool(done.all()):
                    break

        remaining = ~done
        final_height[remaining] = env.scene["object"].data.root_pos_w[remaining, 2]
        final_state[remaining] = controller.state[remaining]
        torch.cuda.synchronize(env.device)
        wall_time_s = time.perf_counter() - started

        tensors = {
            "success": success.detach().cpu().tolist(),
            "terminated": terminated_result.detach().cpu().tolist(),
            "truncated": truncated_result.detach().cpu().tolist(),
            "completion_steps": completion_steps.detach().cpu().tolist(),
            "reward": cumulative_reward.detach().cpu().tolist(),
            "initial_height": initial_height.detach().cpu().tolist(),
            "maximum_height": maximum_height.detach().cpu().tolist(),
            "final_height": final_height.detach().cpu().tolist(),
            "minimum_distance": minimum_distance.detach().cpu().tolist(),
            "ever_grasp_closed": ever_grasp_closed.detach().cpu().tolist(),
            "ever_object_held": ever_object_held.detach().cpu().tolist(),
            "ever_lifted": ever_lifted.detach().cpu().tolist(),
            "final_state": final_state.detach().cpu().tolist(),
        }
        rows: list[dict[str, Any]] = []
        for env_index, params in enumerate(domain_params):
            failure_type = classify_failure(
                success=bool(tensors["success"][env_index]),
                terminated=bool(tensors["terminated"][env_index]),
                truncated=bool(tensors["truncated"][env_index]),
                ever_grasp_closed=bool(tensors["ever_grasp_closed"][env_index]),
                ever_object_held=bool(tensors["ever_object_held"][env_index]),
                ever_lifted=bool(tensors["ever_lifted"][env_index]),
            )
            obj = params["object"]
            row = {
                    "env_index": env_index,
                    "sample_index": params["sample_index"],
                    "success": bool(tensors["success"][env_index]),
                    "failure_type": failure_type,
                    "steps": int(tensors["completion_steps"][env_index]),
                    "completion_time_s": float(tensors["completion_steps"][env_index] * env.step_dt),
                    "total_reward": float(tensors["reward"][env_index]),
                    "initial_object_height_m": float(tensors["initial_height"][env_index]),
                    "maximum_object_height_m": float(tensors["maximum_height"][env_index]),
                    "final_object_height_m": float(tensors["final_height"][env_index]),
                    "minimum_ee_object_distance_m": float(tensors["minimum_distance"][env_index]),
                    "final_controller_state": int(tensors["final_state"][env_index]),
                    "ever_object_held": bool(tensors["ever_object_held"][env_index]),
                    "ever_lifted": bool(tensors["ever_lifted"][env_index]),
                    "object_x_m": float(obj["position_m"][0]),
                    "object_y_m": float(obj["position_m"][1]),
                    "object_yaw_deg": float(obj["yaw_deg"]),
                    "object_mass_kg": float(obj["mass_kg"]),
                    "static_friction": float(obj["static_friction"]),
                    "dynamic_friction": float(obj["dynamic_friction"]),
                    "restitution": float(obj["restitution"]),
                    "camera_applied": bool(params["camera"]["applied"]),
                    "camera_position_m": params["camera"]["position_m"],
                    "camera_target_m": params["camera"]["target_m"],
                    "lighting_intensity": float(params["lighting"]["intensity"]),
                }
            if "sampling" in params:
                sampling = params["sampling"]
                row.update(
                    {
                        "sampling_strategy": sampling["strategy"],
                        "hard_case_selected": bool(sampling["hard_case_selected"]),
                        "hard_case_probability": float(sampling["hard_case_probability"]),
                        "source_failure_episode_index": sampling[
                            "source_failure_episode_index"
                        ],
                        "source_failure_type": sampling["source_failure_type"],
                    }
                )
            rows.append(row)
        metrics: dict[str, float | int] = {
            "vector_steps": vector_steps,
            "simulated_environment_steps": env.num_envs * vector_steps,
            "wall_time_s": wall_time_s,
            "environment_steps_per_second": env.num_envs * vector_steps / wall_time_s,
        }
        return rows, metrics
