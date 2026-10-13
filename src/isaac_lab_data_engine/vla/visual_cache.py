"""Geometry and temporal filtering for offline VLA visual-state caches.

The helpers in this module intentionally do not consume object ground truth.
They reconstruct the robot-root transform from synchronized end-effector poses,
then apply the same RGB-D workspace constraints used by the online hybrid
controller.  Ground-truth object poses belong only in the caller's audit path.
"""

from __future__ import annotations

import numpy as np
import torch

from isaac_lab_data_engine.geometry import (
    invert_transform,
    matrix_to_pose,
    pose_to_matrix,
)
from isaac_lab_data_engine.vision.rgbd_pose import (
    camera_to_world_position,
    world_to_root_position,
)


WORKSPACE_PIXEL_MIN = (0.33, -0.33, -0.015)
WORKSPACE_PIXEL_MAX = (0.81, 0.33, 0.09)
RAW_VALID_MIN = (0.32, -0.35, -0.03)
RAW_VALID_MAX = (0.82, 0.35, 0.15)
OUTPUT_XY_MIN = (0.36, -0.30)
OUTPUT_XY_MAX = (0.78, 0.30)
OUTPUT_TABLE_Z_M = 0.03
DEFAULT_FREEZE_STATE_ID = 1


def derive_robot_root_poses_w(
    eef_poses_w: np.ndarray, eef_poses_b: np.ndarray
) -> np.ndarray:
    """Derive world-frame robot-root poses from synchronized EEF poses.

    Pose convention is ``xyz + quaternion wxyz``.  For every frame,
    ``T_w_root = T_w_eef @ inverse(T_root_eef)``.  This remains correct for a
    translated or rotated robot root and avoids assuming the simulator world
    and robot-root frames coincide.
    """

    eef_poses_w = np.asarray(eef_poses_w, dtype=np.float64)
    eef_poses_b = np.asarray(eef_poses_b, dtype=np.float64)
    if eef_poses_w.shape != eef_poses_b.shape:
        raise ValueError(
            "World/root EEF pose arrays must have identical shapes, got "
            f"{eef_poses_w.shape} and {eef_poses_b.shape}"
        )
    if eef_poses_w.ndim != 2 or eef_poses_w.shape[1] != 7:
        raise ValueError(f"Expected EEF poses [F,7], got {eef_poses_w.shape}")
    if not np.isfinite(eef_poses_w).all() or not np.isfinite(eef_poses_b).all():
        raise ValueError("EEF pose arrays must be finite")

    root_poses = np.empty_like(eef_poses_w)
    for index, (eef_pose_w, eef_pose_b) in enumerate(
        zip(eef_poses_w, eef_poses_b, strict=True)
    ):
        transform_w_root = pose_to_matrix(eef_pose_w) @ invert_transform(
            pose_to_matrix(eef_pose_b)
        )
        root_poses[index] = matrix_to_pose(transform_w_root)
    return root_poses.astype(np.float32)


def workspace_pixel_mask(
    depth: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_pose_w: torch.Tensor,
    root_pose_w: torch.Tensor,
) -> torch.Tensor:
    """Return the online hybrid runner's tabletop workspace mask."""

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth [N,H,W], got {tuple(depth.shape)}")
    batch, height, width = depth.shape
    if camera_intrinsics.shape != (batch, 3, 3):
        raise ValueError("camera_intrinsics must have shape [N,3,3]")
    if camera_pose_w.shape != (batch, 7) or root_pose_w.shape != (batch, 7):
        raise ValueError("camera_pose_w and root_pose_w must have shape [N,7]")

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x_camera = (
        (x[None] - camera_intrinsics[:, 0, 2, None, None])
        / camera_intrinsics[:, 0, 0, None, None]
        * depth
    )
    y_camera = (
        (y[None] - camera_intrinsics[:, 1, 2, None, None])
        / camera_intrinsics[:, 1, 1, None, None]
        * depth
    )
    position_camera = torch.stack((x_camera, y_camera, depth), dim=-1)
    position_w = camera_to_world_position(
        position_camera,
        camera_pose_w[:, None, None, :3],
        camera_pose_w[:, None, None, 3:7],
    )
    position_b = world_to_root_position(
        position_w,
        root_pose_w[:, None, None, :3],
        root_pose_w[:, None, None, 3:7],
    )
    lower = torch.as_tensor(
        WORKSPACE_PIXEL_MIN, dtype=position_b.dtype, device=position_b.device
    )
    upper = torch.as_tensor(
        WORKSPACE_PIXEL_MAX, dtype=position_b.dtype, device=position_b.device
    )
    return torch.isfinite(depth) & torch.all(
        (position_b >= lower) & (position_b <= upper), dim=-1
    )


def camera_predictions_to_visual_position_b(
    position_camera: torch.Tensor,
    camera_pose_w: torch.Tensor,
    root_pose_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform predictions to robot root, clamp them, and report validity.

    Bounds and the known tabletop centre height exactly match
    ``scripts/run_hybrid_vla_pick_lift.py``.
    """

    if position_camera.ndim != 2 or position_camera.shape[1] != 3:
        raise ValueError("position_camera must have shape [N,3]")
    batch = position_camera.shape[0]
    if camera_pose_w.shape != (batch, 7) or root_pose_w.shape != (batch, 7):
        raise ValueError("camera_pose_w and root_pose_w must have shape [N,7]")

    position_w = camera_to_world_position(
        position_camera, camera_pose_w[:, :3], camera_pose_w[:, 3:7]
    )
    raw_position_b = world_to_root_position(
        position_w, root_pose_w[:, :3], root_pose_w[:, 3:7]
    )
    position_b = raw_position_b.clone()
    position_b[:, 0].clamp_(OUTPUT_XY_MIN[0], OUTPUT_XY_MAX[0])
    position_b[:, 1].clamp_(OUTPUT_XY_MIN[1], OUTPUT_XY_MAX[1])
    position_b[:, 2] = OUTPUT_TABLE_Z_M
    lower = torch.as_tensor(
        RAW_VALID_MIN, dtype=raw_position_b.dtype, device=raw_position_b.device
    )
    upper = torch.as_tensor(
        RAW_VALID_MAX, dtype=raw_position_b.dtype, device=raw_position_b.device
    )
    valid = torch.isfinite(raw_position_b).all(dim=-1) & torch.all(
        (raw_position_b >= lower) & (raw_position_b <= upper), dim=-1
    )
    return position_b, valid


def apply_pregrasp_ema(
    raw_position_b: np.ndarray,
    raw_validity: np.ndarray,
    controller_state: np.ndarray,
    *,
    alpha: float = 0.45,
    freeze_state_id: int = DEFAULT_FREEZE_STATE_ID,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply EMA in the unobstructed rest phase, then latch the target.

    The object is static throughout approach, while the robot increasingly
    occludes it and can attract the keypoint model.  The online controller can
    therefore average unobstructed rest observations and freeze as soon as the
    approach starts. Invalid measurements do not move the filtered position.
    """

    raw_position_b = np.asarray(raw_position_b, dtype=np.float32)
    raw_validity = np.asarray(raw_validity, dtype=bool).reshape(-1)
    controller_state = np.asarray(controller_state, dtype=np.int64).reshape(-1)
    if raw_position_b.ndim != 2 or raw_position_b.shape[1] != 3:
        raise ValueError(f"Expected raw positions [F,3], got {raw_position_b.shape}")
    frames = raw_position_b.shape[0]
    if frames == 0:
        raise ValueError("At least one visual frame is required")
    if len(raw_validity) != frames or len(controller_state) != frames:
        raise ValueError("Position, validity, and controller-state lengths must match")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    filtered = np.empty_like(raw_position_b)
    validity = np.empty((frames, 1), dtype=np.float32)
    current_position = raw_position_b[0].copy()
    current_validity = bool(raw_validity[0])
    for index in range(frames):
        if int(controller_state[index]) < freeze_state_id:
            current_validity = bool(raw_validity[index])
            if current_validity:
                if index == 0:
                    current_position = raw_position_b[index].copy()
                else:
                    current_position += alpha * (
                        raw_position_b[index] - current_position
                    )
        filtered[index] = current_position
        validity[index, 0] = float(current_validity)
    return filtered, validity


def freeze_reference_after_grasp(
    position_b: np.ndarray,
    controller_state: np.ndarray,
    *,
    freeze_state_id: int = DEFAULT_FREEZE_STATE_ID,
) -> np.ndarray:
    """Freeze a metrics-only target at the same state as the visual anchor."""

    position_b = np.asarray(position_b, dtype=np.float32)
    controller_state = np.asarray(controller_state, dtype=np.int64).reshape(-1)
    if position_b.ndim != 2 or position_b.shape[1] != 3:
        raise ValueError(f"Expected positions [F,3], got {position_b.shape}")
    if len(position_b) != len(controller_state):
        raise ValueError("Position and controller-state lengths must match")
    result = position_b.copy()
    update_indices = np.flatnonzero(controller_state < freeze_state_id)
    if len(update_indices):
        frozen = controller_state >= freeze_state_id
        result[frozen] = result[update_indices[-1]]
    return result
