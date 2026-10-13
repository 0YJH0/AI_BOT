"""Summarize the five GUI Hard trials, keeping failed visual initialization explicit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from compare_vla_close_lift_pair import grip_metrics, read_json, SCENE_FIELDS, CONTROL_FIELDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expected-rank', type=int, choices=(16, 32), default=32)
    p.add_argument('--compare-root', type=Path)
    args = p.parse_args()
    files = list(args.root.rglob('vla_benchmark_report.json'))
    if len(files) != 1:
        raise ValueError('Expected one benchmark report')
    root = files[0].parent
    report, context = read_json(files[0]), read_json(root/'test_context.json')
    overall, cfg = report['overall'], report['config']
    if (report['status'] != 'VLA_BENCHMARK_OK' or overall['num_episodes'] != 5 or
            cfg['episode_counts'] != {'hard': 5} or overall['vla_error_count'] or
            overall['scripted_ik_fallback_fraction'] or cfg['minimum_actual_lift_m'] != .1):
        raise ValueError('Incomplete/error/fallback benchmark')
    checkpoint = Path(context['checkpoint'])
    actual = hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest()
    if actual.upper() != context['checkpoint_sha256'].upper():
        raise ValueError('Checkpoint changed')
    adapter = read_json(checkpoint/'adapter_config.json')
    if (adapter['r'] != args.expected_rank or adapter['lora_alpha'] != args.expected_rank / 2
            or not context['gui_visible']):
        raise ValueError('Unexpected LoRA rank/alpha or GUI mode')
    with (root/'vla_benchmark_episodes.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 5 or len({r['sample_key'] for r in rows}) != 5:
        raise ValueError('Missing/duplicate scenes')
    episodes = []
    for r in rows:
        path = root/'phase_diagnostics'/f"hard_{int(r['sample_index']):06d}.json"
        metrics = grip_metrics(read_json(path)['trace']) if path.exists() else None
        episodes.append(dict(sample_key=r['sample_key'], success=r['success']=='True',
            failure_reason=r['failure_reason'], visual_initialization_stable=r['visual_initialization_stable']=='True',
            maximum_lift_cm=float(r['maximum_lift_cm']),
            initial_visual_error_cm=(100*float(r['initial_visual_position_error_m'])
                if r['initial_visual_position_error_m'] not in ('', 'None', 'null') else None),
            gripper=metrics))
    summary = dict(status='HARD_GUI_DIAGNOSTIC_COMPLETE', checkpoint=str(checkpoint), rank=args.expected_rank,
        checkpoint_sha256=actual, successes=overall['num_success'], episodes=episodes,
        failure_counts=overall['failure_counts'], mean_inference_ms=overall['mean_vla_inference_ms'],
        limitations='Five fixed-seed Hard trials; GUI rendering can change timing. Not a statistically stable success rate.')
    (args.root/'hard_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if args.compare_root:
        reference = read_json(args.compare_root/'hard_summary.json')
        other_files = list(args.compare_root.rglob('vla_benchmark_report.json'))
        if len(other_files) != 1:
            raise ValueError('Expected one reference report')
        other_root = other_files[0].parent
        other_report = read_json(other_files[0])
        other_context = read_json(other_root/'test_context.json')
        if (other_report['status'] != 'VLA_BENCHMARK_OK' or
                other_report['overall']['vla_error_count'] or
                other_report['overall']['scripted_ik_fallback_fraction']):
            raise ValueError('Reference has errors or fallback')
        for field in (*CONTROL_FIELDS, 'grasp_enable_distance_m', 'gripper_latch_mode'):
            if cfg[field] != other_report['config'][field]:
                raise ValueError(f'Execution mismatch: {field}')
        for field in ('inference_seed', 'gui_visible', 'realtime_pacing'):
            if context[field] != other_context[field]:
                raise ValueError(f'Context mismatch: {field}')
        with (other_root/'vla_benchmark_episodes.csv').open(encoding='utf-8-sig', newline='') as f:
            other_rows = {r['sample_key']: r for r in csv.DictReader(f)}
        if len(other_rows) != 5 or set(other_rows) != {r['sample_key'] for r in rows}:
            raise ValueError('Scene keys mismatch')
        for row in rows:
            for field in SCENE_FIELDS:
                if row[field] != other_rows[row['sample_key']][field]:
                    raise ValueError(f'Scene parameters mismatch: {row["sample_key"]}/{field}')
        indexed = {r['sample_key']: r for r in reference['episodes']}
        comparison = dict(status='PAIRED_HARD_DIAGNOSTIC_COMPLETE', identical_scenes_and_execution=True,
            current_checkpoint=str(checkpoint), reference_checkpoint=reference['checkpoint'],
            current_successes=summary['successes'], reference_successes=reference['successes'],
            pairs=[dict(sample_key=r['sample_key'], current=r, reference=indexed[r['sample_key']])
                   for r in episodes], limitations='Five scene pairs; same global seed, no forced per-request common noise.')
        (args.root/'paired_hard_comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
