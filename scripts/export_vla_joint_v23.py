"""Export only successful live joint-command demonstrations; never relabel old Cartesian data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from isaac_lab_data_engine.vla.joint_interface import FORMAT, ACTION_NAMES, contract, validate_joint_trajectory
from isaac_lab_data_engine.vla.dataset import STATE_NAMES
from export_vla_execution_aligned import scene_group

LEVELS = ('easy', 'medium', 'hard')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load(path):
    metadata = read(path.with_suffix('.json'))
    with np.load(path, allow_pickle=False) as a:
        arrays = {k: a[k] for k in a.files}
    validate_joint_trajectory(arrays, metadata)
    return arrays, metadata


def select(root, counts, seed):
    if counts[0] < 1 or counts[1] < 1 or counts[2] < 0:
        raise ValueError('Invalid independent split counts')
    rng = np.random.default_rng(seed)
    splits = {s: [] for s in ('train', 'validation', 'test')}
    seen = set()
    for level in LEVELS:
        records = []
        for path in sorted((root / level).glob('episode_*.npz')):
            _, meta = load(path)
            group = scene_group(meta)
            if group in seen: raise ValueError('Duplicate physical scene')
            seen.add(group)
            records.append((level, path, group))
        if len(records) < sum(counts): raise ValueError(f'{level}: insufficient verified joint demonstrations')
        chosen = [records[i] for i in rng.permutation(len(records))[:sum(counts)]]
        start = 0
        for split, count in zip(splits, counts):
            splits[split].extend(chosen[start:start+count])
            start += count
    return splits


def export_split(output, split, records):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq
    first, _ = load(records[0][1])
    h, w, c = first['rgb'].shape[1:]
    features = {
        'observation.images.table': dict(dtype='image', shape=(h,w,c), names=['height','width','channel']),
        'observation.state': dict(dtype='float32', shape=(25,), names=list(STATE_NAMES)),
        'action': dict(dtype='float32', shape=(8,), names=ACTION_NAMES),
    }
    for key in ('controller_phase','difficulty_id','loss_weight'):
        features['training.'+key] = dict(dtype='float32', shape=(1,), names=[key])
    destination = output / split
    ds = LeRobotDataset.create(repo_id='local/isaac-joint-v23-'+split, root=destination, fps=10,
        robot_type='franka_absolute_joint_targets_v23', features=features, use_videos=False, image_writer_threads=4)
    manifest_records = []
    phases = {str(p):0 for p in range(5)}
    try:
        for level, path, group in records:
            arrays, meta = load(path)
            for i, phase in enumerate(arrays['controller_phase']):
                # Phase is sampling metadata only, never a student input.
                ds.add_frame({'observation.images.table':arrays['rgb'][i],
                    'observation.state':arrays['state'][i,:25], 'action':arrays['action'][i],
                    'training.controller_phase':np.asarray([phase],np.float32),
                    'training.difficulty_id':np.asarray([LEVELS.index(level)],np.float32),
                    'training.loss_weight':np.asarray([1.],np.float32), 'task':meta['task']})
                phases[str(int(phase))] += 1
            ds.save_episode()
            manifest_records.append(dict(level=level, source=str(path.resolve()), frames=len(arrays['state']),
                sample_key=meta['sample_key'], scene_group=group, full_trajectory=True, dagger_recovery=False,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), maximum_lift_cm=meta['maximum_lift_cm']))
            print('JOINT_EXPORT_EPISODE='+json.dumps(dict(split=split,episodes=len(manifest_records))),flush=True)
    finally:
        ds.finalize()
    # Check saved values against raw live-execution records, not a recomputed label.
    expected_state = np.concatenate([load(p)[0]['state'][:,:25] for _,p,_ in records])
    expected_action = np.concatenate([load(p)[0]['action'] for _,p,_ in records])
    tables = [pq.read_table(p, columns=['observation.state','action']) for p in sorted((destination/'data').glob('*/*.parquet'))]
    saved_state = np.concatenate([np.asarray(t['observation.state'].to_pylist(),np.float32) for t in tables])
    saved_action = np.concatenate([np.asarray(t['action'].to_pylist(),np.float32) for t in tables])
    if not np.array_equal(expected_state, saved_state) or not np.array_equal(expected_action, saved_action):
        raise ValueError('Parquet changed state or actual executed joint labels')
    manifest = dict(format=FORMAT, split=split, records=manifest_records, episodes=len(records),
        frames=len(saved_state), repo_id='local/isaac-joint-v23-'+split,
        state_dim=25, action_dim=8, state_names=list(STATE_NAMES), action_names=ACTION_NAMES,
        action_target_mode='live_absolute_joint_setpoints', action_relabeling=False,
        privileged_object_state=False, visual_pose_conditioning=False, phase_in_policy_input=False,
        controller_phase_frame_counts=phases, interface_contract=contract(),
        level_episode_counts={l:sum(r['level']==l for r in manifest_records) for l in LEVELS},
        level_frame_counts={l:sum(r['frames'] for r in manifest_records if r['level']==l) for l in LEVELS})
    (destination/'balanced_split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (destination/'interface_contract.json').write_text(json.dumps(contract(),indent=2),encoding='utf-8')
    return {k:v for k,v in manifest.items() if k != 'records'}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--train-per-level',type=int,default=170)
    p.add_argument('--validation-per-level',type=int,default=15)
    p.add_argument('--test-per-level',type=int,default=15)
    p.add_argument('--seed',type=int,default=20261011)
    args=p.parse_args()
    if args.output_root.exists(): raise FileExistsError(args.output_root)
    summary=read(args.source_root/'collection_summary.json')
    if summary['status']!='FULL_TRAJECTORY_COLLECTION_OK': raise ValueError('Collection incomplete')
    splits=select(args.source_root,(args.train_per_level,args.validation_per_level,args.test_per_level),args.seed)
    exported=[export_split(args.output_root,s,records) for s,records in splits.items() if records]
    audit=dict(status='JOINT_EXECUTION_EXPORT_VERIFIED', no_posthoc_action_relabeling=True,
        scene_groups_disjoint=True, test_split_used_for_training=False, splits=exported, interface_contract=contract())
    (args.output_root/'export_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__': main()
