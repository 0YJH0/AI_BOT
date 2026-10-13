"""Unprivileged phase ablation and action-only orientation override."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class FeedbackPhaseConfig:
    pregrasp_enter_m: float = .025
    pregrasp_exit_m: float = .035
    descend_enter_m: float = .025
    descend_exit_m: float = .035
    dwell_s: float = .10
    grasp_dwell_s: float = .12
    grasp_width_min_m: float = .01
    grasp_width_max_m: float = .06
    grasp_finger_speed_max_m_s: float = .02
    grasp_visual_distance_max_m: float = .075
    approach_height_m: float = .12


class FeedbackPhase:
    """Monotonic FSM using visual geometry and finger feedback, never object truth.

    This is a composite ablation: relaxed entry threshold + hysteresis + dwell
    + nonempty-grip proxy. It cannot isolate the contribution of each change.
    """
    def __init__(self, dt=.02, config=None):
        self.dt = dt
        self.config = config or FeedbackPhaseConfig()
        self.phase = 0
        self.elapsed = 0.
        self.near = False
        self.near_time = 0.
        self.grip_time = 0.
        self.grip_confirmed = False

    def observe(self, eef_xyz, visual_xyz, finger_positions, finger_velocities, grasp_latched):
        c = self.config
        eef = np.asarray(eef_xyz, dtype=float)
        visual = np.asarray(visual_xyz, dtype=float)
        width = float(np.sum(finger_positions))
        speed = float(np.max(np.abs(finger_velocities)))
        grasp_distance = float(np.linalg.norm(eef - visual))
        pregrasp_distance = float(np.linalg.norm(eef - (visual + [0, 0, c.approach_height_m])))
        nonempty_stable = bool(grasp_latched and c.grasp_width_min_m <= width <= c.grasp_width_max_m
                               and speed <= c.grasp_finger_speed_max_m_s)
        if not c.grasp_width_min_m <= width <= c.grasp_width_max_m:
            self.grip_confirmed = False
        contact_proxy = bool(nonempty_stable and (
            grasp_distance <= c.grasp_visual_distance_max_m or self.grip_confirmed
        ))
        self.grip_time = self.grip_time + self.dt if contact_proxy else 0.
        if self.grip_time >= c.grasp_dwell_s:
            self.grip_confirmed = True
        old = self.phase
        reason = None
        if old == 0:
            if self.elapsed >= .20:
                self.phase, reason = 1, "rest_elapsed"
        elif old in (1, 2):
            distance = pregrasp_distance if old == 1 else grasp_distance
            enter = c.pregrasp_enter_m if old == 1 else c.descend_enter_m
            exit_ = c.pregrasp_exit_m if old == 1 else c.descend_exit_m
            if distance <= enter:
                self.near = True
            elif distance >= exit_:
                self.near = False
            self.near_time = self.near_time + self.dt if self.near else 0.
            if self.grip_time >= c.grasp_dwell_s:
                self.phase, reason = 3, "nonempty_grip_proxy"
            elif self.near_time >= c.dwell_s and self.elapsed >= (.35 if old == 1 else .25):
                self.phase, reason = old + 1, "geometry_hysteresis_dwell"
        elif old == 3:
            # No timer-only lift transition: a closed, empty gripper is not a grasp.
            if self.grip_time >= c.grasp_dwell_s and self.elapsed >= .20:
                self.phase, reason = 4, "stable_nonempty_grip_proxy"
        if self.phase != old:
            self.elapsed = 0.
            self.near = False
            self.near_time = 0.
        self.elapsed += self.dt
        return {"from": old, "to": self.phase, "reason": reason,
                "pregrasp_distance_m": pregrasp_distance, "grasp_distance_m": grasp_distance,
                "finger_width_m": width, "maximum_finger_speed_m_s": speed,
                "grip_proxy": contact_proxy, "grip_confirmed": self.grip_confirmed}

    def protocol(self):
        return {**asdict(self.config), "grip_feedback": "proprioceptive_proxy_not_truth_or_force_sensor",
                "monotonic": True, "timeout_only_advance": False, "composite_ablation": True}


class GripPhaseInput:
    """Model phase override using sustained proprioceptive grip evidence only."""
    def __init__(self, dt=.02):
        self.dt = dt
        self.stable_s = 0.

    def observe(self, legacy_phase, finger_positions, finger_velocities, latched, distance):
        width = float(np.sum(finger_positions))
        speed = float(np.max(np.abs(finger_velocities)))
        valid = bool(latched and .01 <= width <= .06 and speed <= .02 and
                     (distance <= .075 or self.stable_s >= .12))
        self.stable_s = self.stable_s + self.dt if valid else 0.
        phase = 4 if self.stable_s >= .32 else (3 if self.stable_s >= .12 else legacy_phase)
        return int(phase), dict(nonempty_grip_proxy=valid, stable_s=self.stable_s,
                               width_m=width, maximum_finger_speed_m_s=speed)


def override_orientation(action, mode, phase):
    """Preserve this prediction's xyz and gripper bit-for-bit."""
    if mode not in ("model", "fixed_top_down"):
        raise ValueError("Unknown orientation mode")
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError("Expected finite 8-D action")
    if mode == "fixed_top_down" and phase != 0:
        result[3:7] = (0., 1., 0., 0.)
    return result
