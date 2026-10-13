"""CPU tests for the opt-in SmolVLA elementwise weighted loss."""

from __future__ import annotations

import pytest
import torch


def test_per_sample_weight_remains_effective_with_batch_one():
    loss = torch.ones(1, 1, 4, requires_grad=True)
    base = reduce_weighted_action_loss(loss, actual_action_dim=4, sample_weight=[1.],
                                      sample_weight_reduction="per_sample")
    strong = reduce_weighted_action_loss(loss, actual_action_dim=4, sample_weight=[2.5],
                                        sample_weight_reduction="per_sample")
    assert strong.item() == pytest.approx(base.item()*2.5)
    g0 = torch.autograd.grad(base, loss, retain_graph=True)[0]
    g1 = torch.autograd.grad(strong, loss)[0]
    torch.testing.assert_close(g1, g0*2.5)

from isaac_lab_data_engine.vla.weighted_smolvla_loss import (
    WeightedSmolVLALossConfig,
    action_dimension_weights,
    reduce_weighted_action_loss,
)


def test_z_weight_is_applied_to_error_dimension_not_sampling() -> None:
    losses = torch.zeros((1, 1, 32), dtype=torch.float32)
    losses[0, 0, 0] = 1.0
    losses[0, 0, 2] = 4.0
    dimension_weights = action_dimension_weights(8, z_axis_weight=3.0)

    actual = reduce_weighted_action_loss(
        losses,
        actual_action_dim=8,
        action_dim_weight=dimension_weights,
    )

    # (x_error * 1 + z_error * 3) / (7 * 1 + 3)
    assert actual.item() == pytest.approx(13.0 / 10.0)


def test_model_padding_dimensions_are_excluded() -> None:
    losses = torch.zeros((1, 2, 32), dtype=torch.float32)
    losses[:, :, :8] = 2.0
    losses[:, :, 8:] = 10_000.0

    actual = reduce_weighted_action_loss(losses, actual_action_dim=8)

    assert actual.item() == pytest.approx(2.0)


def test_temporal_padding_is_excluded_from_numerator_and_denominator() -> None:
    losses = torch.tensor([[[2.0] * 8, [1000.0] * 8]])
    action_is_pad = torch.tensor([[False, True]])

    actual = reduce_weighted_action_loss(
        losses,
        actual_action_dim=8,
        action_is_pad=action_is_pad,
    )

    assert actual.item() == pytest.approx(2.0)


def test_sample_weight_changes_loss_and_accepts_column_shape() -> None:
    losses = torch.stack(
        (
            torch.ones((1, 8), dtype=torch.float32),
            torch.full((1, 8), 3.0, dtype=torch.float32),
        )
    )
    sample_weight = torch.tensor([[1.0], [3.0]])

    actual = reduce_weighted_action_loss(
        losses,
        actual_action_dim=8,
        sample_weight=sample_weight,
    )

    assert actual.item() == pytest.approx(2.5)


def test_none_contributions_average_to_weighted_mean() -> None:
    generator = torch.Generator().manual_seed(123)
    losses = torch.rand((3, 4, 32), generator=generator)
    dimension_weights = action_dimension_weights(8, z_axis_weight=4.0)
    sample_weights = torch.tensor([1.0, 2.0, 5.0])
    action_is_pad = torch.tensor(
        [
            [False, False, False, True],
            [False, False, True, True],
            [False, False, False, False],
        ]
    )

    scalar = reduce_weighted_action_loss(
        losses,
        actual_action_dim=8,
        action_dim_weight=dimension_weights,
        sample_weight=sample_weights,
        action_is_pad=action_is_pad,
        reduction="mean",
    )
    contributions = reduce_weighted_action_loss(
        losses,
        actual_action_dim=8,
        action_dim_weight=dimension_weights,
        sample_weight=sample_weights,
        action_is_pad=action_is_pad,
        reduction="none",
    )

    assert contributions.shape == (3,)
    assert contributions.mean().item() == pytest.approx(scalar.item())


@pytest.mark.parametrize(
    ("sample_weight", "match"),
    [
        (torch.tensor([1.0, -1.0]), "non-negative"),
        (torch.tensor([1.0, float("nan")]), "finite"),
        (torch.ones((2, 2)), "shape"),
        (torch.zeros(2), "no positive-weight"),
    ],
)
def test_invalid_sample_weights_fail_closed(sample_weight, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        reduce_weighted_action_loss(
            torch.ones((2, 1, 8)),
            actual_action_dim=8,
            sample_weight=sample_weight,
        )


def test_configuration_validation() -> None:
    WeightedSmolVLALossConfig().validate()
    with pytest.raises(ValueError, match="positive"):
        WeightedSmolVLALossConfig(z_axis_weight=0.0).validate()
    with pytest.raises(ValueError, match="non-negative"):
        WeightedSmolVLALossConfig(z_axis_index=-1).validate()


def test_gripper_gradient_boost_excludes_padding_with_batch_one():
    losses = torch.ones((1, 2, 32), requires_grad=True)
    weights = action_dimension_weights(4, gripper_index=3, gripper_weight=2.0)
    value = reduce_weighted_action_loss(
        losses, actual_action_dim=4, action_dim_weight=weights,
        action_is_pad=torch.tensor([[False, True]]), sample_weight=[2.5],
        sample_weight_reduction="per_sample",
    )
    gradient = torch.autograd.grad(value, losses)[0]
    assert gradient[0, 0, 3] == pytest.approx(2 * gradient[0, 0, 0].item())
    assert gradient[0, 0, 2] == pytest.approx(3 * gradient[0, 0, 0].item())
    assert gradient[:, 1].count_nonzero() == 0
    assert gradient[:, :, 4:].count_nonzero() == 0
    assert value.item() == pytest.approx(2.5)


@pytest.mark.parametrize("index,weight", [(2, 2), (4, 2), (3, 0), (3, float("nan")), (None, 2)])
def test_invalid_gripper_weight_or_index_is_rejected(index, weight):
    with pytest.raises(ValueError, match="gripper"):
        action_dimension_weights(4, gripper_index=index, gripper_weight=weight)
