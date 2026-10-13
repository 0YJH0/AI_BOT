import numpy as np
import pytest

from isaac_lab_data_engine.vla.teacher import teacher_target, student_observation
from isaac_lab_data_engine.vla.dataset import build_visual_fsm_bounded_actions
from isaac_lab_data_engine.vla.full_trajectory import validate_full_trajectory
from test_vla_full_trajectory import example


def test_privileged_target_does_not_replace_visual_student_coordinates():
    visual = np.asarray([.55, .03, .03], dtype=np.float32)
    truth = np.asarray([.51, -.02, .025], dtype=np.float32)
    control = teacher_target(mode="truth", visual_position=visual, true_position=truth, lift_anchor=truth, phase=1)
    np.testing.assert_array_equal(control, truth)
    state = student_observation(q=np.zeros(9), dq=np.zeros(9), eef=[.46, 0, .38, 1, 0, 0, 0],
                                visual_position=visual, visual_validity=True, phase=1)
    np.testing.assert_array_equal(state[25:28], visual)
    assert not np.array_equal(state[25:28], control)
    assert state.shape == (30,)


def test_lift_uses_fixed_anchor_not_chasing_held_object():
    control = teacher_target(mode="truth", visual_position=[.7, .1, .03],
                             true_position=[.5, 0, .15], lift_anchor=[.5, 0, .025], phase=4)
    np.testing.assert_allclose(control, [.5, 0, .025])


def test_export_validator_detects_truth_coordinate_leakage():
    arrays, metadata = example()
    n = len(arrays["state"])
    arrays["state"][:, 21:25] = [0, 1, 0, 0]
    arrays["state"][:, 25:28] = [.55, .03, .03]
    arrays["teacher_control_position_b"] = np.tile(np.asarray([.51, -.02, .025], dtype=np.float32), (n, 1))
    arrays["true_object_position_b"] = arrays["teacher_control_position_b"].copy()
    arrays["action"] = build_visual_fsm_bounded_actions(
        arrays["state"][:, 18:25], arrays["teacher_control_position_b"],
        arrays["controller_phase"], position_target_mode="delta",
    )
    arrays["teacher_intended_action"] = arrays["action"].copy()
    arrays["applied_absolute_action"] = arrays["action"].copy()
    arrays["applied_absolute_action"][:, :3] += arrays["state"][:, 18:21]
    arrays["physics_action"][arrays["control_step"]] = arrays["applied_absolute_action"]
    metadata.update(teacher_mode="truth", student_visual_cache_b=[.55, .03, .03],
                    teacher_lift_anchor_b=[.51, -.02, .025])
    assert validate_full_trajectory(arrays, metadata) == n
    arrays["state"][:, 25:28] = arrays["true_object_position_b"]
    with pytest.raises(ValueError, match="privileged-target leakage"):
        validate_full_trajectory(arrays, metadata)


def test_safety_modified_action_is_recorded_instead_of_teacher_proposal():
    arrays, metadata = example()
    n = len(arrays["state"])
    arrays["state"][:, 21:25] = [0, 1, 0, 0]
    arrays["state"][:, 25:28] = [.55, .03, .03]
    arrays["teacher_control_position_b"] = np.tile(np.asarray([.51, -.02, .025], dtype=np.float32), (n, 1))
    arrays["true_object_position_b"] = arrays["teacher_control_position_b"].copy()
    proposal = build_visual_fsm_bounded_actions(
        arrays["state"][:, 18:25], arrays["teacher_control_position_b"],
        arrays["controller_phase"], position_target_mode="delta",
    )
    arrays["teacher_intended_action"] = proposal.copy()
    arrays["action"] = proposal.copy()
    arrays["action"][:, :3] *= .5  # Simulated workspace/safety correction.
    arrays["applied_absolute_action"] = arrays["action"].copy()
    arrays["applied_absolute_action"][:, :3] += arrays["state"][:, 18:21]
    arrays["physics_action"][arrays["control_step"]] = arrays["applied_absolute_action"]
    metadata.update(teacher_mode="truth", student_visual_cache_b=[.55, .03, .03],
                    teacher_lift_anchor_b=[.51, -.02, .025])
    assert validate_full_trajectory(arrays, metadata) == n
    arrays["action"] = proposal.copy()
    with pytest.raises(ValueError, match="actually executed safety-filtered"):
        validate_full_trajectory(arrays, metadata)
