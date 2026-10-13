"""Absolute Franka joint targets; deliberately distinct from Cartesian 8-D actions."""
import json
from pathlib import Path
import numpy as np
from .hybrid_controller import HybridActionDecision
from .full_trajectory import FullTrajectoryRecorder

FORMAT = 'vla_joint_execution_aligned_v23'
ACTION_NAMES = [f'target_panda_joint{i}' for i in range(1, 8)] + ['gripper']


def validate_execution_settings(horizon, repeat, *, teacher_or_replay=False):
    # Saved checkpoint/collection contract remains horizon 1. Horizons 2/4 are
    # deployment-only ablation; each absolute joint target still holds 5 steps.
    if repeat != 5 or horizon not in (1, 2, 4) or (teacher_or_replay and horizon != 1):
        raise ValueError('Joint mode requires hold 5; student horizon 1, 2 or 4, teacher/replay horizon 1')


def contract():
    return dict(version=FORMAT, state_dim=25, action_dim=8, action_names=ACTION_NAMES,
                action_semantics='absolute_arm_joint_position_radians_plus_binary_gripper',
                policy_inputs=['RGB', 'language', 'robot_proprioception'],
                target_coordinates_in_input=False, phase_in_input=False,
                action_repeat=5, execution_horizon=1, maximum_joint_step_rad=.2,
                gripper_semantics='negative_close_positive_open',
                execution='direct_PD_joint_target_no_student_IK')


class JointActionGuard:
    def __init__(self, limits, maximum_step=.2):
        self.limits = np.asarray(limits, np.float32)
        if self.limits.shape != (7, 2) or not np.isfinite(self.limits).all():
            raise ValueError('Expected finite seven-joint limits')
        if not 0 < maximum_step <= .2 or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError('Invalid joint guard settings')
        self.maximum_step = maximum_step
        self.grasp_latched = False

    def filter(self, action, current_q):
        q = np.asarray(current_q, np.float32)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Invalid current arm configuration')
        hold = np.r_[q, -1. if self.grasp_latched else 1.].astype(np.float32)
        if action is None:
            return HybridActionDecision(hold, 'scripted_ik', 'vla_unavailable')
        a = np.asarray(action, np.float32).copy()
        if a.shape != (8,) or not np.isfinite(a).all():
            return HybridActionDecision(hold, 'scripted_ik', 'invalid_joint_action')
        bounded = np.clip(a[:7], self.limits[:, 0], self.limits[:, 1])
        bounded = q + np.clip(bounded - q, -self.maximum_step, self.maximum_step)
        bounded = np.clip(bounded, self.limits[:, 0], self.limits[:, 1])
        clipped = not np.allclose(bounded, a[:7], atol=1e-7, rtol=0)
        a[:7] = bounded
        self.grasp_latched |= bool(a[7] < 0)
        a[7] = -1 if self.grasp_latched else 1
        return HybridActionDecision(a, 'vla', 'accepted_with_safety_filter' if clipped else 'accepted',
                                   position_was_clipped=clipped)


class JointCommandBridge:
    """Use the existing Isaac IK only for teacher proposals, never for student execution.

    During warmup use the original arm action. During a rollout latch a PD
    joint setpoint for exactly five control steps (including both substeps).
    The caller logs the bounded target returned here, not measured future q.
    """
    def __init__(self, env):
        self.env = env
        self.robot = env.scene['robot']
        self.arm = env.action_manager.get_term('arm_action')
        self.original_apply = self.arm.apply_actions
        self.ids = self.arm._joint_ids
        if list(self.arm._joint_names) != [f'panda_joint{i}' for i in range(1, 8)]:
            raise ValueError('Unexpected Franka joint ordering')
        self.guard = JointActionGuard(self.robot.data.soft_joint_pos_limits[0, self.ids].cpu().numpy())
        self.target = None
        self.arm.apply_actions = self.apply

    def reset(self):
        self.target = None
        self.guard.grasp_latched = False

    def apply(self):
        if self.target is None:
            self.original_apply()
        else:
            self.robot.set_joint_position_target(self.target, joint_ids=self.ids)

    def teacher_target(self, cartesian_action):
        import torch
        cart = torch.as_tensor(cartesian_action, device=self.env.device).unsqueeze(0)
        self.arm.process_actions(cart[:, :7])
        self.original_apply()
        desired = self.robot.data.joint_pos_target[0, self.ids].detach().cpu().numpy().copy()
        return self.set_target(np.r_[desired, cartesian_action[7]])

    def set_target(self, action):
        import torch
        q = self.robot.data.joint_pos[0, self.ids].detach().cpu().numpy()
        decision = self.guard.filter(action, q)
        if decision.source == 'vla':
            self.target = torch.as_tensor(decision.action[:7], device=self.env.device).unsqueeze(0)
        return decision

    def held_action(self):
        if self.target is None:
            return None
        return np.r_[self.target[0].detach().cpu().numpy(),
                     -1 if self.guard.grasp_latched else 1].astype(np.float32)


def validate_joint_trajectory(arrays, metadata):
    state, action = arrays['state'], arrays['action']
    n = len(state)
    if state.shape != (n, 30) or action.shape != (n, 8) or n < 15:
        raise ValueError('Incomplete joint trajectory')
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError('Nonfinite state or joint labels')
    if metadata.get('format') != FORMAT or not metadata.get('success'):
        raise ValueError('Only newly collected successful joint rollouts accepted')
    if metadata['maximum_lift_cm'] < 10 or metadata['sustained_success_steps'] < 10:
        raise ValueError('True sustained 10 cm lift required')
    if metadata['model_requests'] or metadata['scripted_fallback_actions']:
        raise ValueError('Unexpected model/fallback in teacher collection')
    phases = arrays['controller_phase']
    if set(phases.tolist()) != set(range(5)) or np.any(np.diff(phases) < 0):
        raise ValueError('All ordered manipulation stages required')
    if not np.array_equal(phases, state[:, 29]) or not np.all(np.diff(arrays['control_step']) == 5):
        raise ValueError('State/action phase or frequency alignment error')
    if np.max(np.abs(action[:, :7] - state[:, :7])) > .200001:
        raise ValueError('Joint command exceeds decision-time safety step')
    if not np.isin(action[:, 7], [-1., 1.]).all():
        raise ValueError('Invalid gripper labels')
    for i, step in enumerate(arrays['control_step']):
        # Physics entry step+1 is generated from the observation after step.
        indices = np.flatnonzero((arrays['physics_step'] > step) & (arrays['physics_step'] <= step+5))
        if not len(indices) or not np.allclose(arrays['physics_joint_action'][indices], action[i], atol=1e-7, rtol=0):
            raise ValueError('Recorded label differs from actually held joint command')
        if not np.allclose(arrays['physics_q'][int(step)], state[i, :9], atol=1e-6, rtol=0):
            raise ValueError('Observation is not causally aligned with first execution step')
    rgb = arrays['rgb'].reshape(n, -1)
    depth = arrays['depth'].reshape(n, -1)
    if np.any(rgb.astype(np.float32).std(1) < 30) or np.any(rgb.max(1)-rgb.min(1) < 128):
        raise ValueError('Invalid RGB frame')
    if np.any((np.isfinite(depth) & (depth > 0)).mean(1) < .4):
        raise ValueError('Invalid depth frame')
    return n


class JointTrajectoryRecorder(FullTrajectoryRecorder):
    def physics_step(self, *, joint_action, **kwargs):
        super().physics_step(**kwargs)
        self.physics[-1]['joint_action'] = np.asarray(joint_action, np.float32).copy()

    def decision(self, *, joint_action, **kwargs):
        super().decision(**kwargs)
        self.decisions[-1]['joint_action'] = np.asarray(joint_action, np.float32).copy()

    def save(self, path: Path, metadata):
        arrays = self.arrays()
        arrays['cartesian_teacher_delta'] = arrays.pop('action')
        arrays['action'] = arrays.pop('joint_action')
        metadata = metadata | dict(format=FORMAT, interface_contract=contract(),
            action_source='live_bounded_IK_teacher_joint_setpoint_directly_executed_and_logged',
            causal_alignment='RGB/state before joint action; 5 control-step zero-order hold',
            phase_frame_counts={str(p): int(np.count_nonzero(arrays['controller_phase'] == p)) for p in range(5)})
        metadata['frames'] = validate_joint_trajectory(arrays, metadata)
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return metadata
