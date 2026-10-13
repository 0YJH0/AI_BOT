"""CPU-only tests for offline VLA visual-state cache geometry and filtering."""

from pathlib import Path

import numpy as np
import torch

from isaac_lab_data_engine.geometry import matrix_to_pose, pose_to_matrix
from isaac_lab_data_engine.vla.visual_cache import (
    apply_pregrasp_ema,
    camera_predictions_to_visual_position_b,
    derive_robot_root_poses_w,
    freeze_reference_after_grasp,
    workspace_pixel_mask,
)


def test_root_pose_derivation_handles_translation_and_rotation() -> None:
    root_pose_w = np.asarray(
        [1.2, -0.4, 0.3, 0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float64
    )
    eef_pose_b = np.asarray(
        [0.45, 0.12, 0.55, 0.9659258, 0.2588190, 0.0, 0.0], dtype=np.float64
    )
    eef_pose_w = matrix_to_pose(
        pose_to_matrix(root_pose_w) @ pose_to_matrix(eef_pose_b)
    )

    derived = derive_robot_root_poses_w(eef_pose_w[None], eef_pose_b[None])[0]

    np.testing.assert_allclose(
        pose_to_matrix(derived), pose_to_matrix(root_pose_w), atol=1.0e-6
    )


def test_workspace_mask_and_prediction_transform_use_robot_root_frame() -> None:
    depth = torch.full((1, 2, 3), 0.03, dtype=torch.float32)
    depth[0, 0, 0] = torch.inf
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 1.0], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]]
    )
    root_pose_w = torch.tensor([[1.0, -0.5, 0.2, 1.0, 0.0, 0.0, 0.0]])
    camera_pose_w = torch.tensor([[1.55, -0.5, 0.2, 1.0, 0.0, 0.0, 0.0]])

    mask = workspace_pixel_mask(depth, intrinsics, camera_pose_w, root_pose_w)

    assert mask.shape == (1, 2, 3)
    assert not bool(mask[0, 0, 0])
    assert bool(mask[0, 1, 1])
    position, valid = camera_predictions_to_visual_position_b(
        torch.tensor([[0.02, -0.01, 0.04]]), camera_pose_w, root_pose_w
    )
    torch.testing.assert_close(position, torch.tensor([[0.57, -0.01, 0.03]]))
    assert bool(valid[0])


def test_rest_ema_skips_invalid_update_and_freezes_at_approach() -> None:
    raw = np.asarray(
        [[0.50, 0.00, 0.03], [0.60, 0.00, 0.03], [0.70, 0.00, 0.03], [0.90, 0.00, 0.03]],
        dtype=np.float32,
    )
    filtered, validity = apply_pregrasp_ema(
        raw,
        np.asarray([True, True, False, True]),
        np.asarray([0, 0, 0, 1]),
        alpha=0.45,
    )

    np.testing.assert_allclose(filtered[:, 0], [0.50, 0.545, 0.545, 0.545])
    np.testing.assert_array_equal(validity[:, 0], [1.0, 1.0, 0.0, 0.0])
    assert filtered.dtype == np.float32
    assert validity.shape == (4, 1)


def test_metrics_reference_freezes_when_approach_starts() -> None:
    true_position = np.asarray(
        [[0.5, 0.0, 0.03], [0.51, 0.0, 0.03], [0.52, 0.0, 0.04], [0.52, 0.0, 0.14]],
        dtype=np.float32,
    )
    reference = freeze_reference_after_grasp(
        true_position, np.asarray([0, 1, 2, 4], dtype=np.int64)
    )

    np.testing.assert_allclose(reference[0], true_position[0])
    np.testing.assert_allclose(reference[1:], [[0.5, 0.0, 0.03]] * 3)


def test_cache_round_trip_has_required_policy_fields(tmp_path: Path) -> None:
    cache_path = tmp_path / "episode_000000.npz"
    np.savez_compressed(
        cache_path,
        source=np.asarray(str(tmp_path / "source_episode"), dtype=np.str_),
        visual_position_b=np.zeros((4, 3), dtype=np.float32),
        validity=np.ones((4, 1), dtype=np.float32),
    )

    with np.load(cache_path, allow_pickle=False) as cache:
        assert {"source", "visual_position_b", "validity"} <= set(cache.files)
        assert cache["visual_position_b"].shape == (4, 3)
        assert cache["validity"].shape == (4, 1)
        assert str(cache["source"].item()).endswith("source_episode")
