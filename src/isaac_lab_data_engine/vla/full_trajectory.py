"""Causal, execution-aligned demonstrations; never relabel a past trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from .dataset import build_visual_fsm_bounded_actions

FORMAT = "vla_execution_aligned_full_v1"


def validate_full_trajectory(arrays: dict, metadata: dict) -> int:
    state, action = arrays["state"], arrays["action"]
    n = len(state)
    if state.shape != (n, 30) or action.shape != (n, 8) or n < 15:
        raise ValueError("Expected a complete 30-D state / 8-D action sequence")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Nonfinite policy state/action")
    if np.any(np.linalg.norm(action[:, :3], axis=1) > 0.040001):
        raise ValueError("Delta exceeds the deployed 4 cm bound")
    if not np.allclose(np.linalg.norm(action[:, 3:7], axis=1), 1, atol=1e-5):
        raise ValueError("Invalid quaternion")
    phases = arrays["controller_phase"]
    if not np.array_equal(phases, state[:, 29]) or set(phases.tolist()) != set(range(5)):
        raise ValueError("All five phases, aligned with state, are required")
    if np.any(np.diff(phases) < 0) or any(np.count_nonzero(phases == p) < 2 for p in range(5)):
        raise ValueError("Missing or discontinuous manipulation phases")
    if not np.array_equal(np.diff(arrays["control_step"]), np.full(n - 1, 5)):
        raise ValueError("Policy decisions must be contiguous at 10 Hz")
    # Decision k is generated after physics step k, then really applied at
    # step k+1. This checks the actual simulator commands, not only labels.
    if not np.allclose(arrays["physics_action"][arrays["control_step"]],
                       arrays["applied_absolute_action"], atol=1e-6, rtol=0):
        raise ValueError("Recorded decision was not actually applied on the next physics step")
    if arrays["rgb"].shape[0] != n or arrays["depth"].shape[0] != n:
        raise ValueError("RGB-D/state/action count mismatch")
    if metadata.get("format") != FORMAT or not metadata.get("success"):
        raise ValueError("Only this format's physically successful rollouts may be exported")
    if metadata["maximum_lift_cm"] < 10 or metadata["sustained_success_steps"] < 10:
        raise ValueError("True lift / sustained grasp criteria failed")
    if metadata["model_requests"] or metadata["scripted_fallback_actions"]:
        raise ValueError("Unexpected model or fallback execution")
    if metadata.get("teacher_mode") == "truth":
        visual = np.asarray(metadata["student_visual_cache_b"], dtype=np.float32)
        if not np.array_equal(state[:, 25:28], np.repeat(visual[None], n, axis=0)):
            raise ValueError("Student visual state was overwritten; privileged-target leakage")
        control = arrays["teacher_control_position_b"]
        truth = arrays["true_object_position_b"]
        if control.shape != (n, 3) or truth.shape != (n, 3):
            raise ValueError("Missing separate teacher-only truth diagnostics")
        not_lift = phases != 4
        if not np.array_equal(control[not_lift], truth[not_lift]):
            raise ValueError("Truth teacher did not use the live object position")
        anchor = np.asarray(metadata["teacher_lift_anchor_b"], dtype=np.float32)
        if not np.array_equal(control[~not_lift], np.repeat(anchor[None], np.count_nonzero(~not_lift), axis=0)):
            raise ValueError("Lift anchor must remain fixed")
        expected = build_visual_fsm_bounded_actions(
            state[:, 18:25], control, phases, maximum_position_step_m=.04,
            approach_height_m=.12, lift_height_m=.18, position_target_mode="delta",
        )
        if not np.array_equal(arrays["teacher_intended_action"], expected):
            raise ValueError("Teacher proposal differs from the live truth teacher rule")
        applied = arrays["applied_absolute_action"]
        decoded = action.copy()
        decoded[:, :3] += state[:, 18:21]
        if not np.allclose(decoded, applied, atol=1e-7, rtol=0):
            raise ValueError("Training labels differ from the actually executed safety-filtered action")
    std = arrays["rgb"].astype(np.float32).reshape(n, -1).std(axis=1)
    ranges = arrays["rgb"].reshape(n, -1).max(axis=1) - arrays["rgb"].reshape(n, -1).min(axis=1)
    depth_valid = (np.isfinite(arrays["depth"]) & (arrays["depth"] > 0)).reshape(n, -1).mean(axis=1)
    if np.any(std < 30) or np.any(ranges < 128) or np.any(depth_valid < 0.4):
        raise ValueError("Blank or invalid RGB-D frame")
    return n


class FullTrajectoryRecorder:
    def __init__(self):
        self.decisions: list[dict] = []
        self.physics: list[dict] = []

    def decision(self, *, state, action, applied_absolute_action, rgb, depth, step,
                 teacher_control_position_b=None, true_object_position_b=None,
                 teacher_intended_action=None):
        record = {
            "state": np.asarray(state, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "applied_absolute_action": np.asarray(applied_absolute_action, dtype=np.float32).copy(),
            "rgb": np.asarray(rgb, dtype=np.uint8).copy(),
            "depth": np.asarray(depth, dtype=np.float16).squeeze().copy(),
            "control_step": int(step),
        }
        if teacher_control_position_b is not None:
            record["teacher_control_position_b"] = np.asarray(teacher_control_position_b, dtype=np.float32).copy()
            record["true_object_position_b"] = np.asarray(true_object_position_b, dtype=np.float32).copy()
        if teacher_intended_action is not None:
            record["teacher_intended_action"] = np.asarray(teacher_intended_action, dtype=np.float32).copy()
        self.decisions.append(record)

    def physics_step(self, *, eef, q, dq, action, object_position_w, step):
        self.physics.append({
            "eef": np.asarray(eef, dtype=np.float32).copy(),
            "q": np.asarray(q, dtype=np.float32).copy(),
            "dq": np.asarray(dq, dtype=np.float32).copy(),
            "action": np.asarray(action, dtype=np.float32).copy(),
            "object_position_w": np.asarray(object_position_w, dtype=np.float32).copy(),
            "step": int(step),
        })

    def arrays(self):
        if not self.decisions:
            raise ValueError("No recorded decisions")
        result = {k: np.asarray([row[k] for row in self.decisions]) for k in self.decisions[0]}
        result["controller_phase"] = result["state"][:, 29].astype(np.int64)
        for k in self.physics[0]:
            result[f"physics_{k}"] = np.asarray([row[k] for row in self.physics])
        return result

    def save(self, path: Path, metadata: dict):
        arrays = self.arrays()
        frames = validate_full_trajectory(arrays, metadata)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        metadata = metadata | {
            "frames": frames, "state_hz": 50, "policy_rgbd_hz": 10,
            "phase_frame_counts": {str(p): int(np.count_nonzero(arrays["controller_phase"] == p)) for p in range(5)},
            "action_source": (
                "online_executed_post_guard_target_minus_same_observation_eef_no_posthoc_relabeling"
                if metadata.get("teacher_mode") == "truth" else
                "recorded_pre_guard_delta_actually_sent_through_deployed_guard"
            ),
            "applied_action_source": "first_guard_filtered_absolute_target",
            "causal_alignment": "state_and_rgbd_before_action_at_same_control_step",
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata
