import numpy as np
from isaac_lab_data_engine.vla.phase_diagnostics import FeedbackPhase, override_orientation


def observe(fsm, error=.1, grip=False, width=.04):
    return fsm.observe([.5,0,.15+error], [.5,0,.03], [width/2,width/2], [0,0], grip)


def test_no_timer_only_advance_from_approach():
    fsm=FeedbackPhase()
    for _ in range(600): observe(fsm)
    assert fsm.phase == 1


def test_hysteresis_keeps_entry_inside_exit_band():
    fsm=FeedbackPhase()
    fsm.phase=1; fsm.elapsed=1
    observe(fsm, error=.02)
    for _ in range(6): observe(fsm, error=.03)
    assert fsm.phase == 2


def test_exit_band_resets_dwell():
    fsm=FeedbackPhase()
    fsm.phase=1; fsm.elapsed=1
    observe(fsm, error=.02); observe(fsm, error=.04)
    assert not fsm.near and fsm.near_time == 0


def test_empty_gripper_cannot_trigger_lift():
    fsm=FeedbackPhase()
    fsm.phase=3
    for _ in range(100): observe(fsm, error=-.12, grip=True, width=0)
    assert fsm.phase == 3


def test_feedback_can_resolve_early_close_without_truth():
    fsm=FeedbackPhase()
    fsm.phase=1; fsm.elapsed=1
    for _ in range(30): observe(fsm, error=-.12, grip=True)
    assert fsm.phase == 4


def test_fixed_orientation_preserves_xyz_and_gripper_and_rest():
    a=np.array([.01,-.02,.03,.1,.2,.3,.4,-.5],dtype=np.float32)
    b=override_orientation(a,"fixed_top_down",2)
    assert np.array_equal(a[:3],b[:3]) and a[7] == b[7]
    assert np.array_equal(b[3:7],[0,1,0,0])
    assert np.array_equal(override_orientation(a,"fixed_top_down",0),a)
    assert np.array_equal(override_orientation(a,"model",4),a)


def test_confirmed_grip_does_not_chase_initial_visual_target_upwards():
    fsm=FeedbackPhase()
    fsm.phase=1; fsm.elapsed=1
    for _ in range(7): observe(fsm, error=-.12, grip=True)
    assert fsm.phase == 3 and fsm.grip_confirmed
    for _ in range(20): observe(fsm, error=.05, grip=True)
    assert fsm.phase == 4
