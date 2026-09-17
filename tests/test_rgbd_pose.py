"""Simulator-free tests for Phase 12 RGB-D perception and policy wiring."""

from __future__ import annotations

import math

import torch

from isaac_lab_data_engine.vision import (
    RGBDObjectPositionNet,
    RGBDPoseModelCfg,
    VisualPolicyObservationAdapter,
    camera_to_world_position,
    preprocess_rgbd,
    quaternion_apply,
    world_to_root_position,
)


def test_preprocess_rgbd_shape_range_and_invalid_depth() -> None:
    rgb = torch.full((2, 24, 32, 4), 128, dtype=torch.uint8)
    depth = torch.full((2, 24, 32, 1), 2.5)
    depth[0, 0, 0, 0] = float("inf")
    inputs = preprocess_rgbd(rgb, depth, image_height=12, image_width=16, depth_clip_m=5.0)
    assert inputs.shape == (2, 4, 12, 16)
    assert torch.isfinite(inputs).all()
    assert inputs.min() >= -1.0 and inputs.max() <= 1.0


def test_model_keeps_metric_target_transform() -> None:
    config = RGBDPoseModelCfg(image_height=32, image_width=40, base_channels=8)
    mean = torch.tensor((0.1, 0.2, 2.0))
    std = torch.tensor((0.2, 0.3, 0.4))
    model = RGBDObjectPositionNet(config, mean, std).eval()
    normalized = model(torch.zeros(3, 4, 32, 40))
    metric = model.predict_position(torch.zeros(3, 4, 32, 40))
    assert normalized.shape == (3, 3)
    assert torch.allclose(metric, normalized * std + mean)


def test_camera_world_root_coordinate_chain() -> None:
    half = math.sqrt(0.5)
    camera_quaternion = torch.tensor(((half, 0.0, 0.0, half),))
    camera_point = torch.tensor(((1.0, 0.0, 0.0),))
    rotated = quaternion_apply(camera_quaternion, camera_point)
    assert torch.allclose(rotated, torch.tensor(((0.0, 1.0, 0.0),)), atol=1.0e-6)
    world = camera_to_world_position(
        camera_point, torch.tensor(((2.0, 3.0, 4.0),)), camera_quaternion
    )
    root = world_to_root_position(
        world, torch.tensor(((2.0, 3.0, 4.0),)), camera_quaternion
    )
    assert torch.allclose(root, camera_point, atol=1.0e-6)


def test_visual_adapter_replaces_every_object_dependent_term() -> None:
    observation = torch.zeros(2, 57)
    observation[:, 7:9] = 0.02
    observation[:, 29:32] = torch.tensor((0.5, 0.0, 0.7))
    observation[:, 32] = 1.0
    # Deliberately place impossible privileged values in all object terms.
    observation[:, 18:21] = 99.0
    observation[:, 36:57] = 99.0
    visual_position = torch.tensor(((0.5, 0.0, 0.68), (0.6, 0.1, 0.7)))
    adapter = VisualPolicyObservationAdapter(ema_alpha=1.0)
    patched = adapter.patch(
        observation, visual_position, elapsed_s=torch.tensor((1.0, 0.0))
    )
    assert torch.allclose(patched[:, 18:21], visual_position)
    assert torch.allclose(patched[:, 36:39], visual_position)
    assert torch.allclose(patched[:, 39], torch.ones(2))
    assert torch.allclose(patched[:, 40:43], torch.zeros(2, 3))
    assert not bool((patched[:, 43:57] == 99.0).any())
    assert patched[0, 46].item() == 0.0
    assert patched[0, 51].item() == 1.0
    assert patched[0, 56].item() == 1.0
    assert patched[1, 52].item() == 1.0
