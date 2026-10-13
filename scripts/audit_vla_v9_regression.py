"""Lock the successful v9 baseline and audit later changes without false causal attribution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from compare_vla_close_lift_pair import SCENE_FIELDS, grip_metrics

ROOT = Path(__file__).resolve().parents[1]
VLA = ROOT / 'results/vla'
CHECKPOINT = VLA / 'training/smolvla_execution_aligned_full_lora16_20261011_truth_v9/checkpoints/006000/pretrained_model'
EXPECTED_HASH = '772d72f4b1bd74a5ed06d445724729885df0b19b42859dfcf6d4f7cd3dc70820'
BASELINE_HISTORY = VLA / 'fresh_lora_closed_loop_20261011_v10/continued_benchmark/visual_only_benchmark_20261011_195404_210'
LATER = {
    'v13_30k': VLA / 'relative4d_30k_diagnostic_20261012_v14/visual_only_benchmark_20261012_175444_556',
    'v14_36k': VLA / 'relative4d_close_lift_paired_20261012_v14/visual_only_benchmark_20261012_185329_247',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def lock_files():
    paths = list(CHECKPOINT.glob('*.json')) + list(CHECKPOINT.glob('*.safetensors'))
    paths += list((VLA / 'models/smolvla_base').glob('*.json'))
    paths += list((VLA / 'models/smolvla_base').glob('*.safetensors'))
    paths += list((ROOT / 'configs').glob('*.yaml'))
    paths += list((ROOT / 'src/isaac_lab_data_engine/vla').glob('*.py'))
    paths += [ROOT / 'scripts' / name for name in (
        'run_vla_benchmark.py', 'serve_smolvla.py', 'serve_smolvla_windows_v044_v4.py',
        'benchmark_vla_truth_student_v9.ps1', 'audit_vla_v9_regression.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def verify_lock(lock):
    changed = [name for name, sha in lock['files'].items()
               if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    if changed:
        raise ValueError(f'Frozen artifacts changed: {changed}')


def summary(path):
    report = read(path / 'vla_benchmark_report.json')
    c, o = report['config'], report['overall']
    return {'path': str(path), 'episodes': o['num_episodes'], 'successes': o['num_success'],
            'levels': {r['difficulty']: {'successes': r['num_success'], 'episodes': r['num_episodes']}
                       for r in report['levels']},
            'invalid_initializations': o.get('failure_counts', {}).get('invalid_visual_initialization', 0),
            'rpc_errors': o['vla_error_count'], 'fallback_fraction': o['scripted_ik_fallback_fraction'],
            'state_dimension': c['state_dimension'], 'config': c}


def episode_rows(path):
    with (path/'vla_benchmark_episodes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['sample_key']: r for r in rows}


def verify_scenes(a, b):
    if set(a) != set(b):
        raise ValueError('Scene keys differ')
    for key in a:
        for field in SCENE_FIELDS:
            if a[key][field] != b[key][field]:
                raise ValueError(f'Scene changed: {key}/{field}')


def audit(output):
    verify_lock(read(output/'baseline_lock.json'))
    reports = list(output.glob('visual_only_benchmark_*/vla_benchmark_report.json'))
    if len(reports) != 1:
        raise ValueError('Exactly one completed baseline required per regression run')
    run = reports[0].parent
    base = summary(run)
    rows = episode_rows(run)
    context = read(run/'test_context.json')
    if context['checkpoint_sha256'].lower() != EXPECTED_HASH:
        raise ValueError('Not the locked v9 checkpoint')
    if (base['episodes'] != 6 or len(rows) != 6 or base['invalid_initializations'] or
            base['rpc_errors'] or base['fallback_fraction']):
        raise ValueError('Baseline incomplete or invalid; do not count infrastructure failures as policy failures')
    if any(r['visual_initialization_stable'] != 'True' for r in rows.values()):
        raise ValueError('Unstable camera initialization')
    metrics = []
    for key, r in rows.items():
        trace = read(run/'phase_diagnostics'/f"{r['difficulty']}_{int(r['sample_index']):06d}.json")
        m = grip_metrics(trace['trace'], base['config']['action_repeat'])
        if m['distinct_policy_requests'] != int(r['vla_request_count']):
            raise ValueError('Trace/request accounting mismatch')
        metrics.append({'sample_key': key, 'success': r['success'] == 'True',
                        'maximum_lift_cm': float(r['maximum_lift_cm']),
                        'phase_transitions': trace['phase_transitions'], **m})
    comparisons = {}
    for name, path in LATER.items():
        other = summary(path)
        verify_scenes(rows, episode_rows(path))
        diffs = {k: {'v9': base['config'].get(k), name: other['config'].get(k)}
                 for k in ('state_dimension', 'vla_input_interface', 'condition_vla_on_controller_phase',
                           'execution_horizon', 'action_repeat', 'max_steps', 'minimum_actual_lift_m')
                 if base['config'].get(k) != other['config'].get(k)}
        comparisons[name] = {'summary': other, 'identical_scene_parameters': True,
                             'execution_config_differences': diffs, 'causal_single_factor': False}
    history = {}
    for name, path in {
        'v9_historical': BASELINE_HISTORY,
        'v10_fresh_same_legacy_interface': VLA/'fresh_lora_closed_loop_20261011_v10/fresh_benchmark/visual_only_benchmark_20261011_203605_690',
        **{name: next((VLA/'phase_ablation_20261011_v11'/name).glob('visual_only_benchmark_*'))
           for name in ('legacy_model','feedback_model','legacy_fixed','feedback_fixed')},
    }.items():
        history[name] = summary(path)
    changes = [
        {'factor': 'initialization', 'v9': 'continued v7 adapter -> verified truth data',
         'later': 'fresh base + new LoRA v10/v13', 'evidence': 'v10 legacy-interface comparison exists; training history intentionally differs'},
        {'factor': 'phase mechanism', 'v9': 'legacy phase is model input',
         'later': 'feedback/hysteresis v11', 'evidence': 'historical 2x2 diagnostic; A had six invalid initializations; v12 repaired Medium/Hard, not a strict simultaneous four-arm repeat'},
        {'factor': 'orientation', 'v9': 'model quaternion', 'later': 'fixed top-down',
         'evidence': 'v11 same-weight intervention exists; fixed override then applied in phases 1..4, v13 applies in all phases'},
        {'factor': 'state representation', 'v9': '30D incl absolute visual xyz + phase',
         'later': '29D relative visual xyz, no phase', 'evidence': 'bundled v13 change; unisolated', 'needs_retraining': True},
        {'factor': 'action dimension', 'v9': '8D delta xyz + quaternion + gripper',
         'later': '4D delta xyz + gripper', 'evidence': 'bundled v13 change; unisolated', 'needs_retraining': True},
        {'factor': 'prediction chunk', 'v9': 4, 'later': 1,
         'evidence': 'bundled v13 change; both execute one prediction held 5 ticks; unisolated', 'needs_retraining': True},
        {'factor': 'training exposure/objective', 'v9': '6000 continuation steps, legacy objective',
         'later': 'fresh 30k with per-sample weighting; then +6k close/lift sampling and gripper weight 2',
         'evidence': 'v14 six-scene comparison showed no closed-loop gain; sampler/loss/optimizer changed together'},
    ]
    result = {'status': 'V9_REGRESSION_AUDIT_COMPLETE', 'baseline': base, 'baseline_episode_metrics': metrics,
              'historical_evidence': history, 'later_comparisons': comparisons, 'change_ledger': changes,
              'camera_protocol': 'Current coherent consecutive-frame readiness; do not reintroduce the historical initialization bug.',
              'metric_definition': 'Early closing uses >2.5cm visual distance before/at first accepted closure; unique policy requests, not repeated holds; a proxy, not contact truth.',
              'limitations': 'Small sample, same scene settings not bit-identical pixels/physics/noise; historical protocols differ. No isolated attribution for bundled changes.',
              'next_experiments_not_started': [
                  'Use locked v9 as control for same-weight guard intervention.',
                  'For training ablations use same initial weights/data/budget/seeds; alter only chunk 4 versus 1.',
                  'Separate relative-coordinate and phase-input changes, then action/orientation changes; never feed new semantics to unchanged old weights as a valid adaptation test.',
              ], 'training_started': False}
    (output/'regression_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'status':result['status'], 'baseline_levels':base['levels'], 'output':str(output)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--freeze', action='store_true')
    args = p.parse_args()
    if args.freeze:
        if digest(CHECKPOINT/'adapter_model.safetensors') != EXPECTED_HASH:
            raise ValueError('v9 successful checkpoint fingerprint does not match')
        lock = {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': EXPECTED_HASH,
                'config': read(CHECKPOINT/'config.json'), 'files':lock_files(),
                'note': 'Content hash lock; no weights overwritten, deleted, or retrained.'}
        with (args.output/'baseline_lock.json').open('x', encoding='utf-8') as f:
            json.dump(lock, f, indent=2)
        print('V9_BASELINE_LOCKED')
    else:
        audit(args.output)
