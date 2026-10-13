"""Behavior-cloning regularization for on-policy fine-tuning."""

from __future__ import annotations

import copy
from typing import Any

import torch


class FrozenTeacherBCAuxiliary:
    """Keep a PPO actor close to a frozen BC/DAgger teacher on visited states.

    The auxiliary updates use the observations collected by the current policy.
    This makes the regularizer DAgger-like: it constrains the actor on the state
    distribution that PPO actually visits instead of replaying only the original
    offline demonstrations.
    """

    def __init__(
        self,
        teacher_policy: torch.nn.Module,
        *,
        coefficient: float = 0.05,
        num_mini_batches: int = 4,
        max_grad_norm: float = 1.0,
        action_weights: tuple[float, ...] = (10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 3.0),
    ) -> None:
        if coefficient <= 0.0:
            raise ValueError("coefficient must be positive")
        if num_mini_batches < 1:
            raise ValueError("num_mini_batches must be positive")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        self.teacher = copy.deepcopy(teacher_policy).eval()
        self.teacher.requires_grad_(False)
        self.coefficient = float(coefficient)
        self.num_mini_batches = int(num_mini_batches)
        self.max_grad_norm = float(max_grad_norm)
        self.action_weights = torch.tensor(action_weights, dtype=torch.float32)

    def update(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout_observations: Any,
    ) -> dict[str, float]:
        """Run one auxiliary epoch over the just-collected rollout."""

        observations = rollout_observations.flatten(0, 1)
        sample_count = int(observations.batch_size[0])
        if sample_count < self.num_mini_batches:
            raise ValueError("rollout contains fewer samples than BC mini-batches")
        indices = torch.randperm(sample_count, device=observations.device)
        chunks = torch.tensor_split(indices, self.num_mini_batches)
        weights = self.action_weights.to(observations.device)
        raw_loss_sum = 0.0
        scaled_loss_sum = 0.0

        policy.train()
        for batch_indices in chunks:
            observation_batch = observations[batch_indices]
            with torch.inference_mode():
                teacher_actions = self.teacher.act_inference(observation_batch)
            student_actions = policy.act_inference(observation_batch)
            if student_actions.shape[-1] != weights.numel():
                raise ValueError(
                    f"BC action weights have {weights.numel()} entries but policy outputs "
                    f"{student_actions.shape[-1]} actions"
                )
            raw_loss = ((student_actions - teacher_actions).square() * weights).mean()
            scaled_loss = self.coefficient * raw_loss
            optimizer.zero_grad(set_to_none=True)
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
            optimizer.step()
            raw_loss_sum += float(raw_loss.detach())
            scaled_loss_sum += float(scaled_loss.detach())

        return {
            "bc_auxiliary": raw_loss_sum / len(chunks),
            "bc_auxiliary_scaled": scaled_loss_sum / len(chunks),
        }

    def wrap_update(self, algorithm: Any) -> None:
        """Attach auxiliary optimization to an RSL-RL PPO instance."""

        original_update = algorithm.update

        def update_with_bc() -> dict[str, float]:
            storage = algorithm.storage
            if storage is None or storage.step != storage.num_transitions_per_env:
                raise RuntimeError("BC auxiliary update requires a full PPO rollout")
            observations = storage.observations
            losses = original_update()
            losses.update(self.update(algorithm.policy, algorithm.optimizer, observations))
            return losses

        algorithm.update = update_with_bc
