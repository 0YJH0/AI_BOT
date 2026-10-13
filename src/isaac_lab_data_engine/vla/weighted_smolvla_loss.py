"""Weighted flow-matching loss for LeRobot 0.4.4 SmolVLA.

The upstream policy pads every action vector to ``max_action_dim`` (32 for
SmolVLA), while this project's robot action has only eight real dimensions.
The stock 0.4.4 policy reduces all 32 dimensions, so three quarters of its
training loss can be spent predicting artificial padding.  This module keeps
the installed LeRobot package immutable and installs an opt-in policy forward
method that:

* reduces only the action dimensions present in the input batch;
* applies explicit per-action-dimension weights (Z is weighted by default);
* applies an optional per-sample ``training.loss_weight`` supplied by the
  dataset/sampler (for example, combined LIFT and Hard weights); and
* removes padded action timesteps from both the numerator and denominator.

Weights act on the elementwise flow-matching squared error, not merely on the
probability that a frame is sampled.  The reduction is normalized by the sum
of active weights, keeping its scale comparable to an ordinary mean loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Sequence

import torch
from torch import Tensor


DEFAULT_SAMPLE_WEIGHT_KEY = "training.loss_weight"
_PATCH_MARKER = "_isaac_weighted_action_loss_installed"


@dataclass(frozen=True)
class WeightedSmolVLALossConfig:
    """Configuration for the opt-in SmolVLA loss patch."""

    z_axis_index: int = 2
    z_axis_weight: float = 3.0
    sample_weight_key: str = DEFAULT_SAMPLE_WEIGHT_KEY
    sample_weight_reduction: str = "normalized"
    gripper_index: int | None = None
    gripper_weight: float = 1.0
    lift_z_axis_weight: float | None = None

    def validate(self) -> None:
        if self.z_axis_index < 0:
            raise ValueError("z_axis_index must be non-negative")
        if not torch.isfinite(torch.tensor(self.z_axis_weight)):
            raise ValueError("z_axis_weight must be finite")
        if self.z_axis_weight <= 0.0:
            raise ValueError("z_axis_weight must be positive")
        if not self.sample_weight_key:
            raise ValueError("sample_weight_key must not be empty")
        if self.sample_weight_reduction not in ("normalized", "per_sample"):
            raise ValueError("Invalid sample weight reduction")
        if self.gripper_index is not None and (self.gripper_index < 0 or self.gripper_index == self.z_axis_index):
            raise ValueError("gripper_index must be non-negative and different from Z")
        if not torch.isfinite(torch.tensor(self.gripper_weight)) or self.gripper_weight <= 0:
            raise ValueError("gripper_weight must be finite and positive")
        if self.gripper_index is None and self.gripper_weight != 1.0:
            raise ValueError("gripper_index is required when gripper_weight differs from 1")
        if self.lift_z_axis_weight is not None and (not torch.isfinite(torch.tensor(self.lift_z_axis_weight))
                                                  or self.lift_z_axis_weight <= 0):
            raise ValueError("lift_z_axis_weight must be finite and positive")


def action_dimension_weights(
    action_dim: int,
    *,
    z_axis_index: int = 2,
    z_axis_weight: float = 3.0,
    gripper_index: int | None = None,
    gripper_weight: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return weights for the real action dimensions only."""

    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if z_axis_index < 0 or z_axis_index >= action_dim:
        raise ValueError(
            f"z_axis_index={z_axis_index} is outside action_dim={action_dim}"
        )
    if not torch.isfinite(torch.tensor(z_axis_weight)) or z_axis_weight <= 0.0:
        raise ValueError("z_axis_weight must be finite and positive")

    weights = torch.ones(action_dim, device=device, dtype=dtype)
    weights[z_axis_index] = z_axis_weight
    if not torch.isfinite(torch.tensor(gripper_weight)) or gripper_weight <= 0:
        raise ValueError("gripper_weight must be finite and positive")
    if gripper_index is None:
        if gripper_weight != 1.0:
            raise ValueError("gripper_index is required when gripper_weight differs from 1")
    else:
        if not 0 <= gripper_index < action_dim or gripper_index == z_axis_index:
            raise ValueError("gripper_index must be a real non-Z action dimension")
        weights[gripper_index] = gripper_weight
    return weights


def _sample_weights(
    sample_weight: Tensor | Sequence[float] | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if sample_weight is None:
        return torch.ones(batch_size, device=device, dtype=dtype)

    weights = torch.as_tensor(sample_weight, device=device, dtype=dtype)
    if weights.ndim == 2 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if weights.ndim != 1 or weights.shape[0] != batch_size:
        raise ValueError(
            "sample_weight must have shape [B] or [B,1], got "
            f"{tuple(weights.shape)} for B={batch_size}"
        )
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("sample_weight must be finite")
    if bool((weights < 0).any()):
        raise ValueError("sample_weight must be non-negative")
    return weights


def _valid_timestep_mask(
    action_is_pad: Tensor | None,
    *,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> Tensor:
    if action_is_pad is None:
        return torch.ones((batch_size, horizon), device=device, dtype=torch.bool)

    padding = torch.as_tensor(action_is_pad, device=device, dtype=torch.bool)
    if padding.ndim == 3 and padding.shape[-1] == 1:
        padding = padding[..., 0]
    if padding.shape != (batch_size, horizon):
        raise ValueError(
            "action_is_pad must have shape [B,H] or [B,H,1], got "
            f"{tuple(padding.shape)} for B={batch_size}, H={horizon}"
        )
    return ~padding


def reduce_weighted_action_loss(
    elementwise_loss: Tensor,
    *,
    actual_action_dim: int,
    action_dim_weight: Tensor | Sequence[float] | None = None,
    sample_weight: Tensor | Sequence[float] | None = None,
    action_is_pad: Tensor | None = None,
    reduction: str = "mean",
    sample_weight_reduction: str = "normalized",
) -> Tensor:
    """Reduce ``[B,H,D_padded]`` losses over real, valid action elements.

    ``reduction="none"`` returns normalized per-sample contributions whose
    arithmetic mean equals the scalar returned by ``reduction="mean"``.  This
    preserves LeRobot's per-sample API while retaining exact sample and
    timestep weighting.
    """

    if elementwise_loss.ndim != 3:
        raise ValueError(
            "elementwise_loss must have shape [B,H,D], got "
            f"{tuple(elementwise_loss.shape)}"
        )
    batch_size, horizon, padded_dim = elementwise_loss.shape
    if actual_action_dim < 1 or actual_action_dim > padded_dim:
        raise ValueError(
            f"actual_action_dim={actual_action_dim} must be within [1,{padded_dim}]"
        )
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")

    losses = elementwise_loss[:, :, :actual_action_dim]
    if action_dim_weight is None:
        dimension_weights = torch.ones(
            actual_action_dim, device=losses.device, dtype=losses.dtype
        )
    else:
        dimension_weights = torch.as_tensor(
            action_dim_weight, device=losses.device, dtype=losses.dtype
        )
        if dimension_weights.shape not in ((actual_action_dim,), (batch_size, horizon, actual_action_dim)):
            raise ValueError(
                "action_dim_weight must have shape [D] or [B,H,D], got "
                f"{tuple(dimension_weights.shape)}"
            )
        if not bool(torch.isfinite(dimension_weights).all()):
            raise ValueError("action_dim_weight must be finite")
        if bool((dimension_weights <= 0).any()):
            raise ValueError("action_dim_weight must be positive")

    sample_weights = _sample_weights(
        sample_weight,
        batch_size=batch_size,
        device=losses.device,
        dtype=losses.dtype,
    )
    valid_timesteps = _valid_timestep_mask(
        action_is_pad,
        batch_size=batch_size,
        horizon=horizon,
        device=losses.device,
    ).to(dtype=losses.dtype)

    element_weights = (
        sample_weights[:, None, None]
        * valid_timesteps[:, :, None]
        * (dimension_weights[None, None, :] if dimension_weights.ndim == 1 else dimension_weights)
    )
    sample_numerators = (losses * element_weights).sum(dim=(1, 2))
    sample_denominators = element_weights.sum(dim=(1, 2))
    if sample_weight_reduction == "per_sample":
        # Normalize time/dimensions per example BEFORE applying sample weight.
        # Unlike normalization by the batch's sum of weights, this remains
        # effective for batch_size=1. Old runs retain the historical default.
        unweighted_denom = (valid_timesteps.sum(1) * dimension_weights.sum() if dimension_weights.ndim == 1
            else (valid_timesteps[:, :, None] * dimension_weights).sum((1, 2)))
        if not bool((unweighted_denom > 0).all()):
            raise ValueError("A sample has no non-padded action elements")
        contributions = sample_numerators / unweighted_denom
        return contributions.mean() if reduction == "mean" else contributions
    if sample_weight_reduction != "normalized":
        raise ValueError("Invalid sample weight reduction")
    total_denominator = sample_denominators.sum()
    if not bool(total_denominator > 0):
        raise ValueError("weighted loss has no positive-weight, non-padded elements")

    if reduction == "mean":
        return sample_numerators.sum() / total_denominator

    # This is an exact decomposition of the weighted batch mean: taking the
    # mean of these B contributions returns the same scalar as above.
    return sample_numerators * (batch_size / total_denominator)


def install_weighted_smolvla_loss(
    config: WeightedSmolVLALossConfig | None = None,
) -> None:
    """Install the opt-in weighted loss on LeRobot's ``SmolVLAPolicy`` class.

    The import is intentionally local so CPU-only dataset/tests do not require
    LeRobot. Calling this function twice is safe when the configuration is
    identical; a conflicting second configuration is rejected.
    """

    if config is None:
        config = WeightedSmolVLALossConfig()
    config.validate()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    installed = getattr(SmolVLAPolicy, _PATCH_MARKER, None)
    if installed is not None:
        if installed != config:
            raise RuntimeError(
                "weighted SmolVLA loss is already installed with a different "
                f"configuration: {installed}"
            )
        return

    original_forward = SmolVLAPolicy.forward

    @wraps(original_forward)
    def forward_with_weighted_action_loss(
        self: Any,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        time: Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        if self.config.adapt_to_pi_aloha:
            # Match the upstream 0.4.4 preprocessing exactly.
            from lerobot.utils.constants import OBS_STATE

            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        real_actions = batch[ACTION]
        if real_actions.ndim != 3:
            raise ValueError(
                "SmolVLA training actions must have shape [B,H,D], got "
                f"{tuple(real_actions.shape)}"
            )
        actual_action_dim = real_actions.shape[-1]
        if config.z_axis_index >= actual_action_dim:
            raise ValueError(
                f"z_axis_index={config.z_axis_index} is outside the real "
                f"action width {actual_action_dim}"
            )

        images, image_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        language_tokens = batch[OBS_LANGUAGE_TOKENS]
        language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        padded_actions = self.prepare_action(batch)
        losses = self.model.forward(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state,
            padded_actions,
            noise,
            time,
        )

        if losses.ndim != 3 or losses.shape[:2] != real_actions.shape[:2]:
            raise ValueError(
                "unexpected SmolVLA elementwise loss shape: "
                f"loss={tuple(losses.shape)}, action={tuple(real_actions.shape)}"
            )

        action_padding = batch.get("actions_id_pad", batch.get("action_is_pad"))
        sample_weight = batch.get(config.sample_weight_key)
        dimension_weights = action_dimension_weights(
            actual_action_dim,
            z_axis_index=config.z_axis_index,
            z_axis_weight=config.z_axis_weight,
            gripper_index=config.gripper_index,
            gripper_weight=config.gripper_weight,
            device=losses.device,
            dtype=losses.dtype,
        )
        if config.lift_z_axis_weight is not None:
            phase = batch.get('training.loss_phase')
            if phase is None or phase.reshape(-1).shape[0] != losses.shape[0]:
                raise ValueError('Raw loss-phase metadata missing after preprocessing')
            phase = phase.reshape(-1).to(losses.device)
            if not bool(((phase >= 0) & (phase <= 4) & (phase == phase.round())).all()):
                raise ValueError('Invalid raw loss phase')
            dimension_weights = dimension_weights[None, None, :].expand(
                losses.shape[0], losses.shape[1], actual_action_dim).clone()
            dimension_weights[phase == 4, :, config.z_axis_index] = config.lift_z_axis_weight
        loss = reduce_weighted_action_loss(
            losses,
            actual_action_dim=actual_action_dim,
            action_dim_weight=dimension_weights,
            sample_weight=sample_weight,
            action_is_pad=action_padding,
            reduction=reduction,
            sample_weight_reduction=config.sample_weight_reduction,
        )

        real_losses = losses[:, :, :actual_action_dim]
        valid = _valid_timestep_mask(
            action_padding,
            batch_size=losses.shape[0],
            horizon=losses.shape[1],
            device=losses.device,
        )
        valid_real = real_losses[valid]
        if valid_real.numel() == 0:
            raise ValueError("SmolVLA batch contains no valid action targets")
        resolved_sample_weights = _sample_weights(
            sample_weight,
            batch_size=losses.shape[0],
            device=losses.device,
            dtype=losses.dtype,
        )
        scalar_loss = loss.mean() if loss.ndim else loss
        loss_dict = {
            "loss": float(scalar_loss.detach()),
            "loss_unweighted_real_dims": float(valid_real.mean().detach()),
            "loss_z_axis": float(
                real_losses[:, :, config.z_axis_index][valid].mean().detach()
            ),
            "loss_sample_weight_mean": float(
                resolved_sample_weights.mean().detach()
            ),
            "loss_sample_weight_max": float(
                resolved_sample_weights.max().detach()
            ),
            "loss_real_action_dim": float(actual_action_dim),
        }
        if config.gripper_index is not None:
            loss_dict["loss_gripper"] = float(
                real_losses[:, :, config.gripper_index][valid].mean().detach()
            )
        return loss, loss_dict

    setattr(SmolVLAPolicy, "forward", forward_with_weighted_action_loss)
    setattr(SmolVLAPolicy, _PATCH_MARKER, config)


__all__ = [
    "DEFAULT_SAMPLE_WEIGHT_KEY",
    "WeightedSmolVLALossConfig",
    "action_dimension_weights",
    "install_weighted_smolvla_loss",
    "reduce_weighted_action_loss",
]
