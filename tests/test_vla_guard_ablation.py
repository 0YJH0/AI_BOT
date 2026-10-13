import importlib.util
from pathlib import Path

import numpy as np
import pytest

from isaac_lab_data_engine.vla.hybrid_controller import HybridActionGuard


def apply(guard, distance, grip=-1):
    eef = np.array([.5, 0., .1, 0., 1., 0., 0.], dtype=np.float32)
    action = np.r_[eef, grip]
    return guard.filter(action, action, current_eef_pose=eef,
                        visual_object_position=np.array([.5, 0., .1-distance]),
                        perception_valid=True)


def test_tighter_gate_changes_only_gripper_before_latch():
    a, b = apply(HybridActionGuard(), .05), apply(HybridActionGuard(grasp_enable_distance_m=.025), .05)
    np.testing.assert_array_equal(a.action[:7], b.action[:7])
    assert a.action[7] == -1 and b.action[7] == 1
    assert not a.gripper_was_blocked and b.gripper_was_blocked


def test_tighter_gate_still_latches_and_resets():
    guard = HybridActionGuard(grasp_enable_distance_m=.025)
    assert apply(guard, .02).action[7] == -1
    assert apply(guard, .2, grip=1).action[7] == -1
    guard.reset()
    assert apply(guard, .05).action[7] == 1


def test_pair_verifier_rejects_second_factor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('guard_compare',scripts/'compare_vla_v9_guard_ablation.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a={'grasp_enable_distance_m':.075,'vla_server':'a','controller_mode':'pure_vla_no_scripted_fallback',
       'vla_input_interface':'legacy','state_dimension':30,'phase_mechanism':'legacy','orientation_mode':'model',
       'execution_horizon':1,'action_repeat':5,'gripper_latch_mode':'hold_closed_until_episode_reset'}
    b={**a,'grasp_enable_distance_m':.025,'vla_server':'b'}
    module.check_config(a,b)
    with pytest.raises(ValueError,match='More than one control factor changed'):
        module.check_config(a,{**b,'action_repeat':10})
