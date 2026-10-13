import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_vla_proprio_v22 import project_state, contract
from isaac_lab_data_engine.vla.hybrid_controller import HybridActionGuard
from isaac_lab_data_engine.vla.dataset import pack_proprioceptive_state


def test_robot_only_projection_and_deployment_agree():
    state = np.arange(60, dtype=np.float32).reshape(2, 30)
    projected = project_state(state)
    assert projected.shape == (2, 25)
    for source, row in zip(state, projected):
        np.testing.assert_array_equal(row, pack_proprioceptive_state(source[:9], source[9:18], source[18:25]))
    state[:, 25:] = 999
    np.testing.assert_array_equal(projected, project_state(state))
    assert contract()["phase_in_input"] is False
    assert contract()["target_coordinates_in_input"] is False


def test_coordinate_free_gripper_gate_has_no_target_dependence():
    eef = np.array([.5, 0, .2, 1, 0, 0, 0], np.float32)
    action = np.r_[eef, -1].astype(np.float32)
    outputs = []
    for target in ([.5, 0, .2], [.8, .3, .02], [np.nan]*3):
        guard = HybridActionGuard(use_visual_grasp_gate=False)
        outputs.append(guard.filter(action, np.r_[eef, 1], current_eef_pose=eef,
            visual_object_position=np.array(target), perception_valid=True).action)
    for result in outputs:
        np.testing.assert_array_equal(result, action)


def test_original_guard_and_position_limits_preserved():
    eef = np.array([.5, 0, .2, 1, 0, 0, 0], np.float32)
    action = np.r_[eef, -1].astype(np.float32)
    original = HybridActionGuard().filter(action, np.r_[eef, 1], current_eef_pose=eef,
        visual_object_position=np.array([.8, 0, .02]), perception_valid=True)
    assert original.gripper_was_blocked and original.action[-1] == 1
    action[0] = .8
    result = HybridActionGuard(use_visual_grasp_gate=False).filter(action, np.r_[eef, 1],
        current_eef_pose=eef, visual_object_position=np.zeros(3), perception_valid=True)
    assert result.position_was_clipped
    assert np.linalg.norm(result.action[:3] - eef[:3]) <= .040001
    with pytest.raises(ValueError):
        project_state(np.zeros((4, 25)))
