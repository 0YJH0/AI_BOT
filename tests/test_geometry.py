"""Tests for centralized SE(3) transforms."""

import numpy as np

from isaac_lab_data_engine.geometry import (
    invert_transform,
    matrix_to_pose,
    pose_to_matrix,
    relative_pose,
)


def test_pose_matrix_round_trip_and_inverse():
    pose = np.array([1.0, -2.0, 0.5, 0.9238795, 0.0, 0.0, 0.3826834])
    transform = pose_to_matrix(pose)
    recovered = matrix_to_pose(transform)
    assert np.allclose(pose_to_matrix(recovered), transform, atol=1.0e-7)
    assert np.allclose(invert_transform(transform) @ transform, np.eye(4), atol=1.0e-7)


def test_relative_pose_composes_back_to_world():
    camera = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    target = np.array([1.5, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0])
    target_camera = relative_pose(camera, target)
    reconstructed = pose_to_matrix(camera) @ pose_to_matrix(target_camera)
    assert np.allclose(reconstructed, pose_to_matrix(target), atol=1.0e-7)
