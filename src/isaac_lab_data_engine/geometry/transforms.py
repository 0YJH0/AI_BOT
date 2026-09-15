"""NumPy SE(3) helpers using xyz + quaternion-wxyz poses."""

from __future__ import annotations

import numpy as np


def _normalized_quaternion(quaternion_wxyz) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Quaternion must be finite and non-zero")
    return quaternion / norm


def pose_to_matrix(pose) -> np.ndarray:
    """Convert xyz + quaternion-wxyz to a homogeneous 4x4 transform."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"Expected finite pose shape (7,), got {pose.shape}")
    w, x, y, z = _normalized_quaternion(pose[3:])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose(transform) -> np.ndarray:
    """Convert a rigid homogeneous transform to xyz + quaternion-wxyz."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"Expected finite transform shape (4, 4), got {transform.shape}")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
        raise ValueError("Transform rotation is not orthonormal")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion = _normalized_quaternion(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return np.concatenate((transform[:3, 3], quaternion))


def invert_transform(transform) -> np.ndarray:
    """Invert a rigid homogeneous transform without a general matrix inverse."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform.shape}")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def relative_pose(reference_pose_w, target_pose_w) -> np.ndarray:
    """Express a world-frame target pose in the reference coordinate frame."""
    transform_w_reference = pose_to_matrix(reference_pose_w)
    transform_w_target = pose_to_matrix(target_pose_w)
    return matrix_to_pose(invert_transform(transform_w_reference) @ transform_w_target)
