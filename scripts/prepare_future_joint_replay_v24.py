"""Audit existing v9 frames and prepare a paired three-scene future-q replay."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from isaac_lab_data_engine.vla.future_joint_labels import future_joint_labels


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-export',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    bundle={};audit=[];total=0;maximum=0.;sums=0.;count=0;over=.0
    for split in ('train','validation','test'):
        manifest=json.loads((args.source_export/split/'balanced_split_manifest.json').read_text(encoding='utf-8'))
        for eid,record in enumerate(manifest['records']):
            path=Path(record['source'])
            metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(path,allow_pickle=False) as a:
                state=a['state'];original=a['action'];steps=a['control_step']
                states,actions=future_joint_labels(state,original,steps)
                # State[k+1] is after five additional 50 Hz control steps.
                np.testing.assert_array_equal(a['physics_q'][steps[1:],:7],actions[:,:7])
                delta=np.abs(actions[:,:7]-states[:,:7])
                maximum=max(maximum,float(delta.max()));sums+=float(delta.sum());count+=delta.size
                over+=int(np.any(delta>.200001,axis=1).sum());total+=len(actions)
                key=record['level']
                if split=='validation' and f'{key}_states' not in bundle:
                    bundle[f'{key}_states']=states
                    bundle[f'{key}_actions']=actions
                    bundle[f'{key}_domain_json']=np.asarray(json.dumps(metadata['domain']))
                    bundle[f'{key}_source']=np.asarray(str(path.resolve()))
                    bundle[f'{key}_episode_id']=np.asarray(eid)
                    bundle[f'{key}_frame_indices']=np.arange(len(actions))
                    audit.append(dict(level=key,source=str(path.resolve()),frames=len(actions),
                        source_maximum_lift_cm=metadata['maximum_lift_cm'],source_success=metadata['success']))
    semantics=dict(interface='joint8',label_source='measured_arm_q_at_next_10Hz_frame',
        future_horizon_control_steps=5,future_horizon_seconds=.1,gripper_source='original_current_command',
        terminal_frames_dropped_per_episode=1,actual_recorded_joint_command=False)
    bundle['semantics_json']=np.asarray(json.dumps(semantics))
    np.savez_compressed(args.output/'replay_bundle.npz',**bundle)
    report=dict(status='FUTURE_JOINT_LABEL_TIME_ALIGNMENT_VERIFIED',semantics=semantics,
        total_frames=total,max_absolute_joint_change_rad=maximum,mean_absolute_joint_change_rad=sums/count,
        frames_exceeding_v23_point2rad_guard=int(over),examples=audit,
        note='Label timing audit only. Requires physical replay before full export and fine-tuning.')
    (args.output/'label_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
