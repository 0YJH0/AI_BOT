import torch
from isaac_lab_data_engine.vla.phase_diagnostics import GripPhaseInput
from isaac_lab_data_engine.vla.phase_loss_dataset import PhaseLossDataset
from isaac_lab_data_engine.vla.weighted_smolvla_loss import reduce_weighted_action_loss


def test_phase_requires_sustained_nonempty_grip():
    p = GripPhaseInput()
    for _ in range(20):
        assert p.observe(1, [0., 0.], [0., 0.], True, .01)[0] == 1
    phases = [p.observe(1, [.02, .02], [0., 0.], True, .02)[0] for _ in range(20)]
    assert 3 in phases and phases[-1] == 4
    assert p.observe(1, [0., 0.], [0., 0.], True, .01)[0] == 1


def test_dynamic_z_weight_changes_gradient_at_batch_one_and_masks_padding():
    loss = torch.tensor([[[1., 1., 4.], [100., 100., 100.]]], requires_grad=True)
    weights = torch.tensor([[[1., 1., 5.], [1., 1., 5.]]])
    actual = reduce_weighted_action_loss(loss, actual_action_dim=3,
        action_dim_weight=weights, action_is_pad=torch.tensor([[False, True]]))
    torch.testing.assert_close(actual, torch.tensor(22/7))
    actual.backward()
    assert loss.grad[0, 0, 2] > loss.grad[0, 0, 0]
    assert loss.grad[0, 1].sum() == 0


def test_phase_metadata_is_raw_and_state_is_unchanged():
    state = torch.zeros(30); state[-1] = 4
    wrapped = PhaseLossDataset([{'observation.state': state}])
    assert wrapped[0]['training.loss_phase'].item() == 4
    assert wrapped[0]['observation.state'] is state
