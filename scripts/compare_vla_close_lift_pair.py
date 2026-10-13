"""Compare paired scene results and pre-grasp closing, without counting action holds as requests."""
import argparse
import csv
import json
from pathlib import Path

SCENE_FIELDS = ('sample_key', 'object_x_m', 'object_y_m', 'object_yaw_deg', 'object_mass_kg',
                'static_friction', 'dynamic_friction', 'restitution', 'camera_position_m',
                'camera_target_m', 'lighting_intensity')
CONTROL_FIELDS = ('config_sha256', 'vision_checkpoint_sha256', 'controller_mode',
                  'vla_input_interface', 'student_phase_in_model_input', 'phase_mechanism',
                  'orientation_mode', 'execution_horizon', 'action_repeat', 'max_steps',
                  'minimum_actual_lift_m', 'success_criteria', 'visual_initialization', 'levels',
                  'task_instruction', 'camera_refresh_renders', 'camera_ready_timeout_s',
                  'camera_warmup_min_steps', 'camera_warmup_max_steps', 'camera_render_yield_s')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def grip_metrics(trace, repeat=5, threshold_cm=2.5):
    def event(row):
        return None if row is None else {
            'control_step': row['control_step'],
            'visual_distance_cm': 100 * row['grasp_distance_m'],
            'finger_mean_m': sum(row['finger_positions_m']) / 2,
        }
    executed = next((r for r in trace if r['filtered_action'][7] < 0), None)
    physical = next((r for r in trace if sum(r['finger_positions_m']) / 2 < .03), None)
    cutoff = executed['control_step'] if executed else float('inf')
    commands = [r for r in trace if (r['control_step'] - 1) % repeat == 0]
    first_intent = next((r for r in commands if r['raw_policy_action'][7] < 0), None)
    early_requests = early_onsets = blocked_requests = 0
    previous_close = False
    for r in commands:
        closing = r['raw_policy_action'][7] < 0
        # After the first accepted close, staying closed during lifting is not premature closing.
        if r['control_step'] <= cutoff:
            early = closing and 100 * r['grasp_distance_m'] > threshold_cm
            early_requests += int(early)
            early_onsets += int(early and not previous_close)
            blocked_requests += int(r['gripper_was_blocked'])
        previous_close = closing
    return {
        'first_close_intent': event(first_intent),
        'first_executed_close': event(executed),
        'first_measured_finger_close': event(physical),
        'early_close_request_count_before_first_executed_close': early_requests,
        'early_close_intent_onsets_before_first_executed_close': early_onsets,
        'first_executed_close_is_early': None if executed is None else 100*executed['grasp_distance_m'] > threshold_cm,
        'blocked_requests_before_first_executed_close': blocked_requests,
        'blocked_control_steps': sum(int(r['gripper_was_blocked']) for r in trace),
        'threshold_sensitivity_first_executed_close': {
            str(t): None if executed is None else 100*executed['grasp_distance_m'] > t
            for t in (1.5, 2.5, 5.0)
        },
        'distinct_policy_requests': len(commands),
    }


def load_arm(root):
    report = read_json(root / 'vla_benchmark_report.json')
    if report['status'] != 'VLA_BENCHMARK_OK' or report['overall']['num_episodes'] != 6:
        raise ValueError('Six completed diagnostic episodes required')
    if report['overall']['vla_error_count'] or report['overall']['scripted_ik_fallback_fraction']:
        raise ValueError('Communication errors or fallback invalidate pure-policy comparison')
    with (root / 'vla_benchmark_episodes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['sample_key'] for r in rows}) != 6:
        raise ValueError('Duplicate or missing scene keys')
    out = {}
    for r in rows:
        if r['visual_initialization_stable'] != 'True':
            raise ValueError('Unstable visual initialization')
        trace = read_json(root / 'phase_diagnostics' / f"{r['difficulty']}_{int(r['sample_index']):06d}.json")['trace']
        metrics = grip_metrics(trace, report['config']['action_repeat'])
        if metrics['distinct_policy_requests'] != int(r['vla_request_count']):
            raise ValueError('Action-repeat request accounting mismatch')
        out[r['sample_key']] = (r, {
            **metrics, 'success': r['success'] == 'True', 'failure_reason': r['failure_reason'],
            'maximum_lift_cm': float(r['maximum_lift_cm']),
            'final_lift_cm': 100*(float(r['final_object_height_m']) - float(r['initial_object_height_m'])),
            'minimum_true_ee_object_distance_cm': 100*float(r['minimum_ee_object_distance_m']),
            'initial_visual_error_cm': 100*float(r['initial_visual_position_error_m']),
        })
    return report, read_json(root / 'test_context.json'), out


def compare(baseline, candidate, output):
    a, ac, ar = load_arm(baseline)
    b, bc, br = load_arm(candidate)
    for field in CONTROL_FIELDS:
        if a['config'][field] != b['config'][field]:
            raise ValueError(f'Control/config mismatch: {field}')
    if ac['inference_seed'] != bc['inference_seed'] or set(ar) != set(br):
        raise ValueError('Inference seed or scene identities differ')
    pairs = []
    for key in ar:
        for field in SCENE_FIELDS:
            if ar[key][0][field] != br[key][0][field]:
                raise ValueError(f'Scene mismatch: {key}/{field}')
        pairs.append({'sample_key': key, 'baseline': ar[key][1], 'candidate': br[key][1],
                      'maximum_lift_gain_cm': br[key][1]['maximum_lift_cm']-ar[key][1]['maximum_lift_cm']})
    aggregates = {}
    for name in ('baseline', 'candidate'):
        rows = [p[name] for p in pairs]
        aggregates[name] = {
            'successes': sum(r['success'] for r in rows), 'episodes': len(rows),
            'early_first_executed_close_episodes': sum(r['first_executed_close_is_early'] is True for r in rows),
            'no_executed_close_episodes': sum(r['first_executed_close'] is None for r in rows),
            'early_close_requests_before_first_executed_close': sum(r['early_close_request_count_before_first_executed_close'] for r in rows),
            'mean_maximum_lift_cm': sum(r['maximum_lift_cm'] for r in rows)/len(rows),
            'max_lift_cm': max(r['maximum_lift_cm'] for r in rows),
        }
    payload = {
        'status': 'PAIRED_DIAGNOSTIC_COMPLETE', 'baseline': str(baseline), 'candidate': str(candidate),
        'baseline_checkpoint_sha256': ac['checkpoint_sha256'], 'candidate_checkpoint_sha256': bc['checkpoint_sha256'],
        'identical_scene_parameters_and_control_config': True,
        'definitions': {
            'early_close': 'Distance to latched visual target > 2.5 cm before/at first accepted close; diagnostic proxy, not contact ground truth.',
            'early_threshold_reason': 'Preselected conservative geometry threshold; also report 1.5/5 cm sensitivity.',
            'first_executed_close': 'First filtered gripper command < 0, before physics integrates this command.',
            'measured_finger_close': 'First pre-action observation with mean finger joint position < 3 cm; not proof of holding the cube.',
            'counts': 'Unique 10 Hz policy requests; 5 held control ticks are not five new requests.',
            'lift': 'Simulator object height minus episode initial object height; success uses actual >=10 cm requirement.',
        },
        'limitations': 'Six paired scenes; GPU simulation/RGB can differ slightly. Same global inference seed, not per-request forced common noise. No statistical generalization claim.',
        'aggregates': aggregates, 'pairs': pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'output': str(output), 'aggregates': aggregates, 'pairs': pairs}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.candidate, args.output)
