"""Convert the project's synchronized trajectories into VLA training samples.

The VLA state deliberately contains robot proprioception only.  In particular,
``object_pose`` is never copied into the state vector: a visual policy must infer
the target from pixels instead of learning a privileged simulator shortcut.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


STATE_DIM = 25
ACTION_DIM = 8
STATE_NAMES = (
    *(f"joint_position_{index}" for index in range(9)),
    *(f"joint_velocity_{index}" for index in range(9)),
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qw",
    "eef_qx",
    "eef_qy",
    "eef_qz",
)
VISUAL_STATE_NAMES = (
    "visual_object_x",
    "visual_object_y",
    "visual_object_z",
    "visual_pose_valid",
)
VISUAL_CONDITIONED_STATE_NAMES = (*STATE_NAMES, *VISUAL_STATE_NAMES)
CONTROLLER_PHASE_NAME = "controller_phase"
VISUAL_FSM_CONDITIONED_STATE_NAMES = (
    *VISUAL_CONDITIONED_STATE_NAMES,
    CONTROLLER_PHASE_NAME,
)
ACTION_NAMES = (
    "target_x",
    "target_y",
    "target_z",
    "target_qw",
    "target_qx",
    "target_qy",
    "target_qz",
    "gripper",
)
DELTA_ACTION_NAMES = (
    "delta_x",
    "delta_y",
    "delta_z",
    *ACTION_NAMES[3:],
)


@dataclass(frozen=True)
class AlignedEpisode:
    """Camera-rate expert samples aligned to the original 50 Hz trajectory."""

    episode_dir: Path
    rgb: np.ndarray
    state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray
    state_trajectory_indices: np.ndarray
    trajectory_indices: np.ndarray
    task: str

    @property
    def frames(self) -> int:
        return int(self.action.shape[0])

    @property
    def observation_trajectory_indices(self) -> np.ndarray:
        """Zero-based trajectory rows supplying each observation/state sample."""

        return self.state_trajectory_indices

    @property
    def action_trajectory_indices(self) -> np.ndarray:
        """Zero-based trajectory rows supplying each action label.

        ``trajectory_indices`` remains the legacy alias for these action rows.
        """

        return self.trajectory_indices


def pack_proprioceptive_state(
    q: np.ndarray,
    dq: np.ndarray,
    eef_pose: np.ndarray,
    visual_object_position: np.ndarray | None = None,
    visual_pose_validity: np.ndarray | float | bool | None = None,
    controller_phase: np.ndarray | float | int | None = None,
) -> np.ndarray:
    """Pack proprioception and optional visual-target/FSM conditioning.

    The backward-compatible outputs are 25-D without visual state and 29-D
    with visual xyz/validity. Supplying ``controller_phase`` appends one raw
    scalar phase id (0..4) to the visual state and returns 30-D.
    """

    q = np.asarray(q, dtype=np.float32)
    dq = np.asarray(dq, dtype=np.float32)
    eef_pose = np.asarray(eef_pose, dtype=np.float32)
    if q.shape[-1:] != (9,) or dq.shape[-1:] != (9,) or eef_pose.shape[-1:] != (7,):
        raise ValueError(
            f"Expected q[...,9], dq[...,9], eef_pose[...,7], got {q.shape}, {dq.shape}, "
            f"{eef_pose.shape}"
        )
    prefix = np.broadcast_shapes(q.shape[:-1], dq.shape[:-1], eef_pose.shape[:-1])
    q = np.broadcast_to(q, (*prefix, 9))
    dq = np.broadcast_to(dq, (*prefix, 9))
    eef_pose = np.broadcast_to(eef_pose, (*prefix, 7))
    state = np.concatenate((q, dq, eef_pose), axis=-1).astype(np.float32, copy=False)
    if visual_object_position is None and visual_pose_validity is None:
        if controller_phase is not None:
            raise ValueError(
                "controller_phase requires visual_object_position and "
                "visual_pose_validity"
            )
        return state
    if visual_object_position is None or visual_pose_validity is None:
        raise ValueError(
            "visual_object_position and visual_pose_validity must be provided together"
        )

    visual_object_position = np.asarray(visual_object_position, dtype=np.float32)
    if visual_object_position.shape[-1:] != (3,):
        raise ValueError(
            "visual_object_position must have final dimension 3, got "
            f"{visual_object_position.shape}"
        )
    visual_pose_validity = np.asarray(visual_pose_validity, dtype=np.float32)
    if visual_pose_validity.ndim == 0 or visual_pose_validity.shape == prefix:
        visual_pose_validity = visual_pose_validity[..., None]
    if visual_pose_validity.shape[-1:] != (1,):
        raise ValueError(
            "visual_pose_validity must be scalar per sample, got "
            f"{visual_pose_validity.shape}"
        )
    augmented_prefix = np.broadcast_shapes(
        prefix,
        visual_object_position.shape[:-1],
        visual_pose_validity.shape[:-1],
    )
    state = np.broadcast_to(state, (*augmented_prefix, STATE_DIM))
    visual_object_position = np.broadcast_to(
        visual_object_position, (*augmented_prefix, 3)
    )
    visual_pose_validity = np.broadcast_to(
        visual_pose_validity, (*augmented_prefix, 1)
    )
    visual_state = np.concatenate(
        (state, visual_object_position, visual_pose_validity), axis=-1
    ).astype(np.float32, copy=False)
    if controller_phase is None:
        return visual_state

    controller_phase = np.asarray(controller_phase, dtype=np.float32)
    if controller_phase.ndim == 0 or controller_phase.shape == augmented_prefix:
        controller_phase = controller_phase[..., None]
    if controller_phase.shape[-1:] != (1,):
        raise ValueError(
            "controller_phase must be scalar per sample, got "
            f"{controller_phase.shape}"
        )
    final_prefix = np.broadcast_shapes(
        augmented_prefix, controller_phase.shape[:-1]
    )
    controller_phase = np.broadcast_to(controller_phase, (*final_prefix, 1))
    if not np.isfinite(controller_phase).all():
        raise ValueError("controller_phase must be finite")
    rounded_phase = np.rint(controller_phase)
    if not np.allclose(controller_phase, rounded_phase, atol=0.0, rtol=0.0):
        raise ValueError("controller_phase must contain integer-valued phase ids")
    if np.any((rounded_phase < 0.0) | (rounded_phase > 4.0)):
        raise ValueError("controller_phase must be within [0, 4]")
    visual_state = np.broadcast_to(
        visual_state, (*final_prefix, len(VISUAL_CONDITIONED_STATE_NAMES))
    )
    return np.concatenate((visual_state, controller_phase), axis=-1).astype(
        np.float32, copy=False
    )


def build_visual_fsm_bounded_actions(
    eef_pose: np.ndarray,
    visual_object_position: np.ndarray,
    controller_phase: np.ndarray,
    *,
    maximum_position_step_m: float = 0.04,
    approach_height_m: float = 0.12,
    lift_height_m: float = 0.18,
    position_target_mode: str = "absolute",
) -> np.ndarray:
    """Build smooth IK labels from visual position and FSM phase.

    The phase waypoint is expressed in the robot-root frame, then its xyz is
    clipped along the current-EEF-to-waypoint direction. By default the first
    three action values remain an absolute pose target. ``position_target_mode
    = "delta"`` instead stores ``bounded_target_xyz - current_eef_xyz`` while
    preserving the absolute quaternion and gripper semantics.
    """

    eef_pose = np.asarray(eef_pose, dtype=np.float32)
    visual_object_position = np.asarray(visual_object_position, dtype=np.float32)
    controller_phase = np.asarray(controller_phase)
    if eef_pose.ndim != 2 or eef_pose.shape[1] != 7:
        raise ValueError(f"Expected eef_pose [F,7], got {eef_pose.shape}")
    frames = eef_pose.shape[0]
    if visual_object_position.shape != (frames, 3):
        raise ValueError(
            f"Expected visual_object_position [{frames},3], got "
            f"{visual_object_position.shape}"
        )
    controller_phase = controller_phase.reshape(-1)
    if controller_phase.shape != (frames,):
        raise ValueError(
            f"Expected controller_phase [{frames}], got {controller_phase.shape}"
        )
    if not np.isfinite(eef_pose).all() or not np.isfinite(visual_object_position).all():
        raise ValueError("EEF and visual positions must be finite")
    if maximum_position_step_m <= 0.0:
        raise ValueError("maximum_position_step_m must be positive")
    if position_target_mode not in {"absolute", "delta"}:
        raise ValueError(
            "position_target_mode must be 'absolute' or 'delta', got "
            f"{position_target_mode!r}"
        )
    if approach_height_m < 0.0 or lift_height_m < 0.0:
        raise ValueError("Approach and lift heights must be non-negative")
    if not np.issubdtype(controller_phase.dtype, np.integer):
        if not np.isfinite(controller_phase).all() or not np.allclose(
            controller_phase, np.rint(controller_phase), atol=0.0, rtol=0.0
        ):
            raise ValueError("controller_phase must contain integer-valued phase ids")
    controller_phase = controller_phase.astype(np.int64, copy=False)
    if np.any((controller_phase < 0) | (controller_phase > 4)):
        raise ValueError("controller_phase must be within [0, 4]")

    rest = controller_phase == 0
    approach = controller_phase == 1
    lift = controller_phase == 4
    waypoint = visual_object_position.copy()
    waypoint[rest] = eef_pose[rest, :3]
    waypoint[approach, 2] += float(approach_height_m)
    waypoint[lift, 2] += float(lift_height_m)

    delta = waypoint - eef_pose[:, :3]
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(
        1.0,
        float(maximum_position_step_m) / np.maximum(distance, 1.0e-12),
    ).astype(np.float32, copy=False)
    bounded_delta = delta * scale
    bounded_position = eef_pose[:, :3] + bounded_delta

    actions = np.empty((frames, ACTION_DIM), dtype=np.float32)
    actions[:, :3] = (
        bounded_position if position_target_mode == "absolute" else bounded_delta
    )
    actions[:, 3:7] = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float32)
    actions[rest, 3:7] = eef_pose[rest, 3:7]
    actions[:, 7] = np.where(controller_phase <= 2, 1.0, -1.0)
    return actions


def discover_demonstrations(
    dataset_roots: list[str | Path],
    *,
    successes_only: bool = True,
    minimum_camera_frames: int = 2,
) -> list[Path]:
    """Return deterministic episode paths suitable for VLA imitation learning."""

    episodes: list[Path] = []
    for root_value in dataset_roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for episode_dir in sorted(root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]")):
            metadata_path = episode_dir / "metadata.json"
            trajectory_path = episode_dir / "trajectory.npz"
            camera_path = episode_dir / "camera.npz"
            if not (metadata_path.is_file() and trajectory_path.is_file() and camera_path.is_file()):
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if successes_only and not bool(metadata.get("success", False)):
                continue
            if int(metadata.get("camera_frames", 0)) < minimum_camera_frames:
                continue
            episodes.append(episode_dir)
    return episodes


def align_episode(
    episode_dir: str | Path,
    *,
    task: str = "Pick up the red cube and lift it 10 centimeters.",
    action_offset_steps: int = 0,
) -> AlignedEpisode:
    """Align 10 Hz camera observations to trajectory actions.

    ``frame_step`` is one-based in the project's dataset.  Trajectory arrays are
    zero-based, so a camera frame at step *s* uses state sample *s-1*.  The
    collector records post-step observations together with the action that was
    just applied.  For causal behavior cloning, pass ``action_offset_steps=1``
    so the current observation is paired with the next controller action.  The
    default remains zero for backward compatibility with existing exports.
    """

    if action_offset_steps < 0:
        raise ValueError("action_offset_steps must be non-negative")

    episode_dir = Path(episode_dir).expanduser().resolve()
    with np.load(episode_dir / "trajectory.npz") as trajectory:
        q = np.asarray(trajectory["q"], dtype=np.float32)
        dq = np.asarray(trajectory["dq"], dtype=np.float32)
        eef_pose = np.asarray(trajectory["eef_pose"], dtype=np.float32)
        actions = np.asarray(trajectory["action"], dtype=np.float32)
    with np.load(episode_dir / "camera.npz") as camera:
        rgb = np.asarray(camera["rgb"], dtype=np.uint8)
        frame_steps = np.asarray(camera["frame_step"], dtype=np.int64)
        timestamps = np.asarray(camera["timestamp"], dtype=np.float32)

    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action [T,{ACTION_DIM}], got {actions.shape}")
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB [F,H,W,3], got {rgb.shape}")
    if not (len(rgb) == len(frame_steps) == len(timestamps)):
        raise ValueError("Camera tensors have inconsistent frame counts")
    if np.any(frame_steps < 1) or np.any(frame_steps > len(actions)):
        raise ValueError("camera frame_step falls outside trajectory bounds")

    state_indices = frame_steps - 1
    action_indices = state_indices + action_offset_steps
    valid = action_indices < len(actions)
    if not np.any(valid):
        raise ValueError(
            f"No aligned frames remain in {episode_dir} with "
            f"action_offset_steps={action_offset_steps}"
        )
    rgb = rgb[valid]
    timestamps = timestamps[valid]
    state_indices = state_indices[valid]
    action_indices = action_indices[valid]

    state = pack_proprioceptive_state(
        q[state_indices], dq[state_indices], eef_pose[state_indices]
    )
    aligned_actions = actions[action_indices]
    if not np.isfinite(state).all() or not np.isfinite(aligned_actions).all():
        raise ValueError(f"Non-finite VLA sample in {episode_dir}")
    return AlignedEpisode(
        episode_dir=episode_dir,
        rgb=rgb,
        state=state,
        action=aligned_actions,
        timestamps=timestamps,
        state_trajectory_indices=state_indices,
        trajectory_indices=action_indices,
        task=task,
    )


def iter_aligned_episodes(
    episode_dirs: list[str | Path], *, task: str
) -> Iterator[AlignedEpisode]:
    for episode_dir in episode_dirs:
        yield align_episode(episode_dir, task=task)
