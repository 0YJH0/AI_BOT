"""Summarize per-control-step grip/geometry evidence, without rerunning policies."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    report = json.loads((root / 'vla_benchmark_report.json').read_text(encoding='utf-8-sig'))
    context = json.loads((root / 'test_context.json').read_text(encoding='utf-8-sig'))
    with (root / 'vla_benchmark_episodes.csv').open(encoding='utf-8-sig', newline='') as stream:
        episodes = list(csv.DictReader(stream))
    output = []
    for episode in episodes:
        name = f"{episode['difficulty']}_{int(episode['sample_index']):06d}.json"
        diagnostic = json.loads((root / 'phase_diagnostics' / name).read_text())
        trace = diagnostic['trace']
        first_latch = next((r for r in trace if r['grasp_latched_after_guard']), None)
        near = [r for r in trace if r['grasp_distance_m'] <= .075]
        commands = [r for r in trace if (r['control_step'] - 1) % 5 == 0]
        out = {key: episode.get(key) for key in (
            'difficulty', 'sample_index', 'success', 'failure_reason', 'maximum_lift_cm',
            'minimum_ee_object_distance_m', 'ever_grasp_closed', 'ever_object_held',
            'visual_initialization_stable', 'initial_visual_position_error_m',
            'vla_request_count', 'vla_error_count')}
        out.update({
            'min_pregrasp_distance_cm': 100 * min(r['pregrasp_distance_m'] for r in trace),
            'min_visual_grasp_distance_cm': 100 * min(r['grasp_distance_m'] for r in trace),
            'gripper_blocked_control_steps': diagnostic['gripper_blocked_control_steps'],
            'near_control_steps': len(near),
            'close_request_count': sum(r['raw_policy_action'][7] < 0 for r in commands),
            'total_request_count_from_trace': len(commands),
            'phase_transitions_diagnostic_only': diagnostic['phase_transitions'],
        })
        if first_latch:
            out['first_close_latch'] = {
                'step': first_latch['control_step'],
                'visual_distance_cm': 100 * first_latch['grasp_distance_m'],
                'eef_minus_visual_xyz_cm': [100 * (a-b) for a,b in zip(first_latch['eef_xyz_b'], first_latch['visual_xyz_b'])],
                'raw_delta_xyz_cm': [100 * v for v in first_latch['raw_policy_action'][:3]],
                'finger_positions_m': first_latch['finger_positions_m'],
            }
        output.append(out)
    result = {
        'status': 'DIAGNOSTIC_SUMMARY_COMPLETE', 'source': str(root),
        'checkpoint_sha256': context['checkpoint_sha256'],
        'scope': 'six_episode_diagnostic_not_statistical_success_estimate',
        'phase_is_policy_input': False,
        'caveat': 'Guard latch is a close-command latch, not proof of holding an object. Blocked steps include repeated actions.',
        'overall': report['overall'], 'episodes': output,
    }
    target = root / 'close_lift_diagnostic_summary.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    summarize(parser.parse_args().root)
