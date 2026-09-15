"""Simulator-free checks for the vectorized scripted controller buffers."""

import torch

from isaac_lab_data_engine.controllers import PickAndLiftStateMachine


def test_state_machine_supports_independent_environment_reset():
    controller = PickAndLiftStateMachine(dt=0.02, num_envs=4, device="cpu")
    controller.state[:] = torch.tensor([1, 2, 3, 4])
    controller.elapsed[:] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    controller.reset(torch.tensor([1, 3]))

    assert controller.state.tolist() == [1, 0, 3, 0]
    assert torch.allclose(controller.elapsed, torch.tensor([0.1, 0.0, 0.3, 0.0]))
    assert controller.gripper_action.shape == (4,)


def test_state_machine_returns_one_action_per_environment():
    controller = PickAndLiftStateMachine(dt=0.02, num_envs=4, device="cpu")
    pose = torch.zeros((4, 7))
    pose[:, 3] = 1.0

    actions = controller.compute(pose, pose, pose)

    assert actions.shape == (4, 8)
    assert torch.all(actions[:, 7] == controller.OPEN_GRIPPER)
