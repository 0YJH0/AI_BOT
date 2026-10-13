"""Compare fixed-scene horizon diagnostics without confusing actions with RPC calls."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def event(row):
    if row is None:
        return None
    return dict(control_step=row['control_step'],
        true_distance_cm=100*row['true_ee_object_distance_m'],
        visual_distance_cm=None if row.get('grasp_distance_m') is None else 100*row['grasp_distance_m'],
        raw_gripper=row['raw_policy_action'][7], executed_gripper=row['filtered_action'][7])


def load(root):
    report=read(root/'vla_benchmark_report.json')
    context=read(root/'test_context.json')
    if report['status']!='VLA_BENCHMARK_OK' or report['overall']['num_episodes']!=6:
        raise ValueError('Six complete episodes required')
    if report['overall']['vla_error_count'] or report['overall']['scripted_ik_fallback_fraction']:
        raise ValueError('RPC errors or fallback invalidate policy comparison')
    with (root/'vla_benchmark_episodes.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=6 or len({r['sample_key'] for r in rows})!=6:
        raise ValueError('Duplicate or missing episodes')
    out={}
    for row in rows:
        trace=read(root/'phase_diagnostics'/f"{row['difficulty']}_{int(row['sample_index']):06d}.json")['trace']
        actions=[r for r in trace if (r['control_step']-1)%report['config']['action_repeat']==0
                 and r.get('raw_policy_action') is not None and r.get('filtered_action') is not None]
        first_raw=next((r for r in actions if r['raw_policy_action'][7]<0),None)
        first_executed=next((r for r in actions if r['filtered_action'][7]<0),None)
        out[row['sample_key']]=(row,dict(success=row['success']=='True',failure_reason=row['failure_reason'],
            maximum_lift_cm=float(row['maximum_lift_cm']),
            minimum_true_distance_cm=100*float(row['minimum_ee_object_distance_m']),
            first_close_intent=event(first_raw), first_executed_close=event(first_executed),
            blocked_action_decisions=sum(bool(r['gripper_was_blocked']) for r in actions),
            clipped_action_decisions=sum(bool(r['position_was_clipped']) for r in actions),
            policy_requests=int(row['vla_request_count']), executed_action_decisions=len(actions),
            wall_time_s=float(row['wall_time_s']),
            camera_ready=row['visual_initialization_stable']=='True',
            initial_visual_error_cm=100*float(row['initial_visual_position_error_m'])))
    return report,context,out


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    a,ac,ar=load(args.baseline); b,bc,br=load(args.candidate)
    if ac['checkpoint_sha256'].lower()!=bc['checkpoint_sha256'].lower():
        raise ValueError('Weights differ')
    for field in ('config_sha256','action_repeat','max_steps','minimum_actual_lift_m','task_instruction',
                  'vla_input_interface','student_phase_in_model_input','phase_mechanism','student_phase_mode',
                  'orientation_mode','grasp_enable_distance_m','gripper_latch_mode','success_criteria','levels'):
        if a['config'].get(field)!=b['config'].get(field): raise ValueError('Control settings differ: '+field)
    if a['config']['execution_horizon']!=1 or b['config']['execution_horizon']!=2:
        raise ValueError('Expected horizon 1 vs 2')
    if ar.keys()!=br.keys(): raise ValueError('Scene keys differ')
    for key in ar:
        for field in ('object_x_m','object_y_m','object_yaw_deg','object_mass_kg','static_friction',
                      'dynamic_friction','restitution','camera_position_m','camera_target_m','lighting_intensity'):
            if ar[key][0][field]!=br[key][0][field]: raise ValueError(f'Scene mismatch {key}: {field}')
    result=dict(status='PAIRED_HORIZON_COMPARISON_COMPLETE',baseline=str(args.baseline),candidate=str(args.candidate),
        checkpoint_sha256=ac['checkpoint_sha256'],same_scene_parameters=True,episodes_per_arm=6,
        baseline_successes=a['overall']['num_success'],candidate_successes=b['overall']['num_success'],
        baseline_level_success_rates=a['overall']['level_success_rates'],
        candidate_level_success_rates=b['overall']['level_success_rates'],
        baseline_requests=a['overall']['vla_request_count'],candidate_requests=b['overall']['vla_request_count'],
        pairs=[dict(sample_key=k,horizon1=ar[k][1],horizon2=br[k][1]) for k in ar],
        limitations=['Six-scene diagnostic, not statistically established improvement.',
            'Historical baseline; GUI/rendering and inference-noise trajectory can differ. Camera initialization metrics retained.'])
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__': main()
