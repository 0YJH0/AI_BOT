"""Audit phase-only diagnostics and fixed-budget single-factor lifting experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_vla_close_lift_pair import read_json, grip_metrics, SCENE_FIELDS, CONTROL_FIELDS


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def streak(values):
    best = current = 0
    for v in values:
        current = current + 1 if v else 0
        best = max(best, current)
    return best


def load(root):
    paths = list(root.rglob('vla_benchmark_report.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one report: {root}')
    directory = paths[0].parent
    report, ctx = read_json(paths[0]), read_json(directory/'test_context.json')
    if (report['status'] != 'VLA_BENCHMARK_OK' or report['overall']['num_episodes'] != 9 or
            report['overall']['vla_error_count'] or report['overall']['scripted_ik_fallback_fraction']):
        raise ValueError('Nine episodes without errors/fallback required')
    with (directory/'vla_benchmark_episodes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 9 or len({r['sample_key'] for r in rows}) != 9:
        raise ValueError('Missing or duplicate scene')
    episodes = {}
    for row in rows:
        if row['visual_initialization_stable'] != 'True':
            raise ValueError('Invalid visual initialization; do not count as policy failure')
        trace = read_json(directory/'phase_diagnostics'/f"{row['difficulty']}_{int(row['sample_index']):06d}.json")['trace']
        metrics = grip_metrics(trace)
        first = metrics['first_executed_close']
        after = [r for r in trace if first and r['control_step'] >= first['control_step']]
        commands = [r for r in after if (r['control_step']-1)%5 == 0]
        dz = [r['raw_policy_action'][2] for r in commands]
        episodes[row['sample_key']] = dict(row=row, success=row['success']=='True',
            maximum_lift_cm=float(row['maximum_lift_cm']), gripper=metrics,
            post_close_phase4_requests=sum(r['phase_input']==4 for r in commands),
            post_close_mean_dz_cm=float(np.mean(dz)*100) if dz else None,
            post_close_up_fraction=float(np.mean(np.array(dz)>.005)) if dz else None,
            longest_up_command_run=streak(v>.005 for v in dz),
            longest_geometric_hold_s=.02*streak(r['object_held_metric'] and r['gripper_closed_metric'] for r in after),
            max_true_lift_with_hold_cm=max([100*r['object_lift_m'] for r in after
                if r['object_held_metric'] and r['gripper_closed_metric']] or [0.]))
    return dict(report=report, context=ctx, episodes=episodes,
                successes=sum(e['success'] for e in episodes.values()),
                phase4_requests=sum(e['post_close_phase4_requests'] for e in episodes.values()),
                upward_fraction=float(np.mean([e['post_close_up_fraction'] for e in episodes.values()
                    if e['post_close_up_fraction'] is not None])) if any(
                    e['post_close_up_fraction'] is not None for e in episodes.values()) else 0.)


def paired(a, b, phase_only=False):
    fields = (*CONTROL_FIELDS, 'grasp_enable_distance_m', 'gripper_latch_mode')
    if not phase_only:
        fields += ('student_phase_mode',)
    for field in fields:
        if a['report']['config'][field] != b['report']['config'][field]:
            raise ValueError(f'Execution mismatch: {field}')
    if a['context']['inference_seed'] != b['context']['inference_seed']:
        raise ValueError('Different inference seed')
    if set(a['episodes']) != set(b['episodes']):
        raise ValueError('Different scene keys')
    if phase_only and a['context']['checkpoint_sha256'] != b['context']['checkpoint_sha256']:
        raise ValueError('Weights changed in phase-only experiment')
    for key in a['episodes']:
        for field in SCENE_FIELDS:
            if a['episodes'][key]['row'][field] != b['episodes'][key]['row'][field]:
                raise ValueError(f'Scene mismatch: {key}/{field}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=('phase','final','freeze','verify_inputs'), required=True)
    args = p.parse_args()
    root = args.root
    plan = read_json(root/'experiment_plan.json')
    if args.stage in ('freeze','verify_inputs'):
        if args.stage == 'freeze':
            project = Path(__file__).resolve().parents[1]
            files = list(Path(plan['parent_checkpoint']).rglob('*'))
            for split in ('train','validation'):
                files += list((Path(plan['export'])/split).rglob('*'))
            files += [Path(plan['sampling_plan']), Path(plan['sampling_plan']).with_suffix('.weights.npy')]
            files += list((project/'src/isaac_lab_data_engine/vla').glob('*.py'))
            files += [project/'scripts'/name for name in (
                'run_vla_benchmark.py','benchmark_vla_truth_student_v9.ps1','lerobot_train_windows.py',
                'train_smolvla_dagger_recovery_8gb_v7.ps1','run_vla_lift_single_factor_v21.ps1',
                'analyze_vla_lift_v21.py','build_vla_sampling_plan.py','scan_smolvla_checkpoints.py')]
            hashes = {str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}
            dump(root/'input_lock.json', hashes)
        else:
            for name, expected in read_json(root/'input_lock.json').items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'Frozen input changed: {name}')
        print('INPUTS_' + args.stage.upper() + '_OK')
        return
    if args.stage == 'phase':
        a, b = load(root/'phase_legacy'), load(root/'phase_grip_feedback')
        paired(a, b, phase_only=True)
        choose = b['phase4_requests']>0 and b['successes']>=a['successes'] and b['upward_fraction']>a['upward_fraction']
        report = dict(status='PHASE_INPUT_DIAGNOSTIC_COMPLETE', legacy=a, grip_feedback=b,
            selected_student_phase_mode='grip_feedback' if choose else 'legacy',
            selection_rule='Require phase4 exposure, no success-count regression, and higher post-close upward fraction; small-sample diagnostic.',
            definitions=dict(upward='raw delta z > 0.5 cm per unique request',
                hold='closed fingers and true EEF-object distance below benchmark threshold; geometric proxy, not contact proof'),
            limitations='Nine paired scenes, one inference seed; no statistical generalization.')
        dump(root/'phase_input_report.json', report)
        print(json.dumps({k:report[k] for k in ('status','selected_student_phase_mode')}))
        return
    arms = {name:load(root/('benchmark_'+name)) for name in ('control','lift_sampling','lift_z')}
    for other in ('lift_sampling','lift_z'):
        paired(arms['control'], arms[other])
    initialization = [read_json(root/(name+'_initialization.json')) for name in arms]
    if len({a['trainable_parameter_sha256'] for a in initialization}) != 1:
        raise ValueError('Initial adapter tensors differ')
    configs = {}
    objectives = {}
    for name, run in plan['runs'].items():
        checkpoint = Path(plan['project'])/'results/vla/training'/run/'checkpoints/006000/pretrained_model'
        cfg = read_json(checkpoint/'train_config.json')
        cfg.pop('job_name', None); cfg.pop('output_dir', None)
        configs[name] = cfg
        objectives[name] = read_json(checkpoint/'training_objective.json')
    if configs['control'] != configs['lift_sampling'] or configs['control'] != configs['lift_z']:
        raise ValueError('Training configs differ beyond environment sampler/Z loss')
    control_loss = objectives['control']['weighted_flow_matching']
    if objectives['lift_sampling']['weighted_flow_matching'] != control_loss:
        raise ValueError('Sampling arm changed loss')
    zloss = dict(objectives['lift_z']['weighted_flow_matching'])
    if zloss.pop('lift_z_axis_weight') != 5 or control_loss.get('lift_z_axis_weight') is not None:
        raise ValueError('Wrong phase Z loss')
    if zloss != {k:v for k,v in control_loss.items() if k != 'lift_z_axis_weight'}:
        raise ValueError('Z arm changed a second loss factor')
    if objectives['lift_z']['sampling_plan_sha256'] != objectives['control']['sampling_plan_sha256']:
        raise ValueError('Z arm changed sampler')
    dump(root/'single_factor_report.json', dict(status='LIFT_SINGLE_FACTOR_COMPLETE', arms=arms,
        same_initial_adapter=True, same_training_budget=6000,
        limitations='Single continuation seed; nine fixed scenes per arm. Loss scales differ; rank is held at 32.'))
    print('LIFT_SINGLE_FACTOR_COMPLETE')


if __name__ == '__main__':
    main()
