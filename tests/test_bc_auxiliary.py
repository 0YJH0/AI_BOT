"""Simulator-free tests for PPO behavior-cloning regularization."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaac_lab_data_engine.training import FrozenTeacherBCAuxiliary


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(3, 8, bias=False)

    def act_inference(self, observations: TensorDict) -> torch.Tensor:
        return self.actor(observations["policy"])


def test_bc_auxiliary_moves_student_toward_frozen_teacher() -> None:
    torch.manual_seed(7)
    teacher = _Policy()
    student = _Policy()
    with torch.no_grad():
        teacher.actor.weight.fill_(0.5)
        student.actor.weight.zero_()
    observations = TensorDict(
        {"policy": torch.ones(4, 6, 3)},
        batch_size=[4, 6],
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    auxiliary = FrozenTeacherBCAuxiliary(
        teacher,
        coefficient=1.0,
        num_mini_batches=2,
    )

    before = torch.nn.functional.mse_loss(
        student.act_inference(observations.flatten(0, 1)),
        teacher.act_inference(observations.flatten(0, 1)),
    )
    losses = auxiliary.update(student, optimizer, observations)
    after = torch.nn.functional.mse_loss(
        student.act_inference(observations.flatten(0, 1)),
        teacher.act_inference(observations.flatten(0, 1)),
    )

    assert losses["bc_auxiliary"] > 0.0
    assert after < before
    assert all(parameter.grad is None for parameter in auxiliary.teacher.parameters())
