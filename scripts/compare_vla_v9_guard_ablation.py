"""Strict paired v9 ablation: ONLY first-close permission distance may differ."""
import argparse
import json
from pathlib import Path

from audit_vla_v9_regression import EXPECTED_HASH, read, verify_lock
from compare_vla_close_lift_pair import SCENE_FIELDS, load_arm


def run_path(root):
    reports = list(root.glob('visual_only_benchmark_*/vla_benchmark_report.json'))
    if len(reports) != 1:
        raise ValueError('Exactly one completed report per arm required')
    return reports[0].parent


def check_config(a, b):
    if a['grasp_enable_distance_m'] != .075 or b['grasp_enable_distance_m'] != .025:
        raise ValueError('Unexpected ablation distances')
    allowed = {'grasp_enable_distance_m', 'vla_server'}
    differences = [k for k in set(a) | set(b) if k not in allowed and a.get(k) != b.get(k)]
    if differences:
        raise ValueError(f'More than one control factor changed: {differences}')
    for cfg in (a, b):
        if (cfg['controller_mode'] != 'pure_vla_no_scripted_fallback' or
                cfg['vla_input_interface'] != 'legacy' or cfg['state_dimension'] != 30 or
                cfg['phase_mechanism'] != 'legacy' or cfg['orientation_mode'] != 'model' or
                cfg['execution_horizon'] != 1 or cfg['action_repeat'] != 5 or
                cfg['gripper_latch_mode'] != 'hold_closed_until_episode_reset'):
            raise ValueError('Not the intended v9 execution protocol')


def compare(root):
    verify_lock(read(root/'baseline_lock.json'))
    ap, bp = (run_path(root/name) for name in ('baseline_75mm', 'strict_25mm'))
    a, ac, ar = load_arm(ap)
    b, bc, br = load_arm(bp)
    check_config(a['config'], b['config'])
    if (ac['checkpoint_sha256'].lower() != EXPECTED_HASH or
            bc['checkpoint_sha256'].lower() != EXPECTED_HASH or
            ac['inference_seed'] != bc['inference_seed'] or set(ar) != set(br)):
        raise ValueError('Model, seed or scene keys changed')
    pairs = []
    for key in ar:
        for f in SCENE_FIELDS:
            if ar[key][0][f] != br[key][0][f]:
                raise ValueError(f'Scene changed: {key}/{f}')
        pairs.append({'sample_key':key,'baseline':ar[key][1],'strict':br[key][1],
                      'lift_gain_cm':br[key][1]['maximum_lift_cm']-ar[key][1]['maximum_lift_cm']})
    aggregates = {}
    for name in ('baseline','strict'):
        rows = [p[name] for p in pairs]
        aggregates[name] = {
            'episodes':len(rows),'successes':sum(r['success'] for r in rows),
            'levels':{level: {'episodes':sum(p['sample_key'].startswith(level+':') for p in pairs),
                             'successes':sum(p[name]['success'] for p in pairs if p['sample_key'].startswith(level+':'))}
                      for level in ('easy','medium')},
            'early_first_executed_close_episodes':sum(r['first_executed_close_is_early'] is True for r in rows),
            'no_executed_close_episodes':sum(r['first_executed_close'] is None for r in rows),
            'early_close_requests':sum(r['early_close_request_count_before_first_executed_close'] for r in rows),
            'blocked_requests_before_first_close':sum(r['blocked_requests_before_first_executed_close'] for r in rows),
            'max_lift_cm':max(r['maximum_lift_cm'] for r in rows),
            'mean_maximum_lift_cm':sum(r['maximum_lift_cm'] for r in rows)/len(rows),
        }
    payload = {'status':'V9_GUARD_SINGLE_FACTOR_COMPLETE', 'baseline':str(ap),'strict':str(bp),
               'checkpoint_sha256':EXPECTED_HASH,'factor':'grasp_enable_distance_m',
               'values_m':[.075,.025],'scene_and_other_control_parameters_equal':True,
               'aggregates':aggregates,'pairs':pairs,
               'interpretation_limits':[
                   'Six paired scenes; not a reliable population success rate.',
                   'Tighter gate mechanically reduces accepted distant closes; only successful actual lifting demonstrates benefit.',
                   'Raw premature-close requests need not decrease: unchanged model can keep requesting a blocked action.',
                   'Same global inference seed, not per-request shared diffusion noise; physics/RGB are not bit-identical.',
                   'Only the distance gate changed. Closed-gripper latch is NOT removed and no training is performed.',
               ]}
    (root/'comparison.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'status':payload['status'],'aggregates':aggregates,'output':str(root/'comparison.json')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--root',type=Path)
    g.add_argument('--validate-root',type=Path)
    args=parser.parse_args()
    if args.validate_root:
        load_arm(run_path(args.validate_root))
        print('VALID_SIX_EPISODE_ARM')
    else:
        compare(args.root)
