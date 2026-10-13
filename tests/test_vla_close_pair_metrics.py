import runpy
from pathlib import Path

grip_metrics = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/compare_vla_close_lift_pair.py'))['grip_metrics']


def row(step, distance, raw, executed, blocked=False):
    return {'control_step': step, 'grasp_distance_m': distance,
            'raw_policy_action': [0]*7+[raw], 'filtered_action': [0]*7+[executed],
            'finger_positions_m': [.04, .04], 'gripper_was_blocked': blocked}


def test_held_steps_and_lifting_close_are_not_new_early_requests():
    trace = [row(s, .09, -1, 1, True) for s in range(1, 6)]
    trace += [row(s, .01, -1, -1) for s in range(6, 11)]
    trace += [row(s, .15, -1, -1) for s in range(11, 16)]
    m = grip_metrics(trace)
    assert m['distinct_policy_requests'] == 3
    assert m['early_close_request_count_before_first_executed_close'] == 1
    assert m['early_close_intent_onsets_before_first_executed_close'] == 1
    assert m['first_executed_close_is_early'] is False
    assert m['blocked_control_steps'] == 5
    assert m['blocked_requests_before_first_executed_close'] == 1
    assert m['first_measured_finger_close'] is None


def test_no_executed_close_is_missing_not_zero_distance():
    m = grip_metrics([row(s, .1, -1, 1, True) for s in range(1, 11)])
    assert m['first_executed_close'] is None
    assert m['first_executed_close_is_early'] is None
    assert m['early_close_request_count_before_first_executed_close'] == 2


def test_first_command_and_measured_closure_are_distinct():
    trace = [row(s, .04, -1, -1) for s in range(1, 6)]
    trace[-1]['finger_positions_m'] = [.02, .02]
    m = grip_metrics(trace)
    assert m['first_executed_close']['control_step'] == 1
    assert m['first_measured_finger_close']['control_step'] == 5
    assert m['first_executed_close_is_early'] is True
