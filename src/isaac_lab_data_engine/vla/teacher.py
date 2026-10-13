"""Privileged expert targets are kept separate from student observations."""
from __future__ import annotations

import numpy as np

from .dataset import pack_proprioceptive_state


def teacher_target(*, mode, visual_position, true_position, lift_anchor, phase):
    if mode not in {"visual", "truth"}:
        raise ValueError("Teacher mode must be visual or truth")
    position = true_position if mode == "truth" else visual_position
    # Never chase the object upwards during lifting.
    if phase == 4:
        position = lift_anchor
    position = np.asarray(position, dtype=np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Teacher target must be a finite robot-frame xyz")
    return position.copy()


def student_observation(*, q, dq, eef, visual_position, visual_validity, phase):
    # Deliberately has NO true_position argument.
    return pack_proprioceptive_state(
        q, dq, eef, visual_position, visual_validity, phase
    )
