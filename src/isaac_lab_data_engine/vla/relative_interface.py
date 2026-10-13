"""Versioned phase-free student interface; targets come only from vision."""
import numpy as np
from .dataset import STATE_NAMES

INTERFACE_NAME = "visual_relative_xyz_gripper_v1"
RELATIVE_STATE_NAMES = (*STATE_NAMES, "visual_relative_x", "visual_relative_y",
                        "visual_relative_z", "visual_pose_valid")
RELATIVE_ACTION_NAMES = ("delta_x", "delta_y", "delta_z", "gripper")


def relative_student_state(state):
    s = np.asarray(state, dtype=np.float32)
    if s.ndim < 1 or s.shape[-1] not in (29, 30) or not np.isfinite(s).all():
        raise ValueError("Expected finite legacy visual state with 29/30 features")
    out = s[..., :29].copy()
    out[..., 25:28] = s[..., 25:28] - s[..., 18:21]
    return out


def project_executed_action(action):
    """Retain executed xyz/gripper exactly, never regenerate teacher labels."""
    a = np.asarray(action, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] != 8 or not np.isfinite(a).all():
        raise ValueError("Expected finite executed 8-D action")
    return a[..., [0, 1, 2, 7]].copy()


def expand_relative_action(action):
    """4-D model output -> legacy wire/guard format with fixed quaternion."""
    a = np.asarray(action, dtype=np.float32)
    if a.ndim < 1 or a.shape[-1] != 4 or not np.isfinite(a).all():
        raise ValueError("Expected finite 4-D action")
    out = np.zeros((*a.shape[:-1], 8), np.float32)
    out[..., :3] = a[..., :3]
    out[..., 4] = 1.
    out[..., 7] = a[..., 3]
    return out


def contract():
    return {"name": INTERFACE_NAME, "state_dim": 29, "action_dim": 4,
            "state_names": list(RELATIVE_STATE_NAMES), "action_names": list(RELATIVE_ACTION_NAMES),
            "target_source": "latched_rgbd_visual_estimate", "object_truth_in_input": False,
            "phase_in_input": False, "relative_axes": "robot_base_not_tool_local",
            "orientation": "fixed_top_down_wxyz_0_1_0_0_all_policy_actions",
            "maximum_position_step_m": .04, "action_repeat": 5,
            "execution_horizon": 1, "policy_hz": 10, "control_hz": 50}
