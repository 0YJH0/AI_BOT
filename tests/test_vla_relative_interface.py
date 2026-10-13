import numpy as np
import pytest
from isaac_lab_data_engine.vla.relative_interface import (
    relative_student_state, project_executed_action, expand_relative_action, contract)


def test_relative_state_drops_teacher_phase():
    x = np.arange(30, dtype=np.float32)
    y = relative_student_state(x)
    assert y.shape == (29,)
    np.testing.assert_array_equal(y[:25], x[:25])
    np.testing.assert_array_equal(y[25:28], x[25:28]-x[18:21])
    x[29] = 4
    np.testing.assert_array_equal(relative_student_state(x), y)
    assert not contract()['phase_in_input'] and not contract()['object_truth_in_input']


def test_action_projection_preserves_executed_xyz_and_gripper():
    x = np.arange(16, dtype=np.float32).reshape(2, 8)
    a = project_executed_action(x)
    y = expand_relative_action(a)
    np.testing.assert_array_equal(y[:, :3], x[:, :3])
    np.testing.assert_array_equal(y[:, 7], x[:, 7])
    np.testing.assert_array_equal(y[:, 3:7], [[0, 1, 0, 0]]*2)
    np.testing.assert_array_equal(x, np.arange(16).reshape(2, 8))


def test_bad_interface_shapes_fail():
    with pytest.raises(ValueError): relative_student_state(np.zeros(25))
    with pytest.raises(ValueError): expand_relative_action(np.zeros(8))


def test_relative_rpc_tags_semantics_not_just_feature_width():
    from isaac_lab_data_engine.vla.rpc import VLARequest
    request = VLARequest(np.zeros((4, 4, 3), np.uint8), np.zeros(29, np.float32),
                         "lift", input_interface="relative4d")
    assert request.as_arrays()["input_interface"].item() == "relative4d"
    with pytest.raises(ValueError):
        VLARequest(request.rgb, np.zeros(30), "lift", "relative4d").as_arrays()


def test_phase_free_sampler_reads_scalar_or_singleton_diagnostic_not_validity_bit():
    from scripts.build_vla_sampling_plan import phase_values_from_columns
    states = np.ones((3, 29))
    expected = np.array([1, 2, 4])
    np.testing.assert_array_equal(phase_values_from_columns(states, expected), expected)
    np.testing.assert_array_equal(phase_values_from_columns(states, expected[:, None]), expected)
