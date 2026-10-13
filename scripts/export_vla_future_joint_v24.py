"""Reuse v9 scenes with next-decision measured q labels, preserving original splits.

These are future position waypoints, NOT recovered original joint commands.
The v23 contract describes the shared deployment interface, not label provenance.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from isaac_lab_data_engine.vla.future_joint_labels import future_joint_labels
from isaac_lab_data_engine.vla.joint_interface import ACTION_NAMES, contract
from isaac_lab_data_engine.vla.dataset import STATE_NAMES

LEVELS = ('easy', 'medium', 'hard')
FORMAT = 'vla_future_measured_joint_waypoints_v24'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_records(root, limit=0):
    """Preserve record ordering as well as episode/scene split membership."""
    splits = {}
    seen = set()
    for split in ('train', 'validation', 'test'):
        manifest = read(root / split / 'balanced_split_manifest.json')
        if manifest['split'] != split:
            raise ValueError('Source split mismatch')
        records = []
        counts = {level: 0 for level in LEVELS}
        for record in manifest['records']:
            group = record['scene_group']
            if group in seen:
                raise ValueError('Source scenes overlap across episodes/splits')
            seen.add(group)
            level = record['level']
            if limit and counts[level] >= limit:
                continue
            counts[level] += 1
            records.append(record)
        if not records or any(count == 0 for count in counts.values()):
            raise ValueError('Missing difficulty')
        splits[split] = records
    return splits


def load_episode(record):
    path = Path(record['source'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['source_sha256']:
        raise ValueError(f'Source changed since v9 export: {path}')
    meta = read(path.with_suffix('.json'))
    if not meta.get('success') or meta['maximum_lift_cm'] < 10:
        raise ValueError('Not a successful 10 cm source demonstration')
    with np.load(path, allow_pickle=False) as raw:
        states, actions = future_joint_labels(raw['state'], raw['action'], raw['control_step'])
        steps = raw['control_step'][1:].astype(np.int64)
        if not np.array_equal(actions[:, :7], raw['physics_q'][steps, :7]):
            raise ValueError('Future decision frame differs from aligned physics joint state')
        rgb = raw['rgb'][:-1].copy()
        phases = raw['controller_phase'][:-1].copy()
    if len(states) != record['frames'] - 1 or len(rgb) != len(states) or len(phases) != len(states):
        raise ValueError('Episode frame alignment mismatch')
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError('Invalid original RGB')
    if not np.isin(actions[:, 7], [-1, 1]).all() or not np.isin(phases, range(5)).all():
        raise ValueError('Invalid gripper or phase metadata')
    return states[:, :25], actions, rgb, phases, meta


def export_split(output, split, records):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    import io

    first = load_episode(records[0])
    h, w, c = first[2].shape[1:]
    features = {
        'observation.images.table': dict(dtype='image', shape=(h,w,c), names=['height','width','channel']),
        'observation.state': dict(dtype='float32', shape=(25,), names=list(STATE_NAMES)),
        'action': dict(dtype='float32', shape=(8,), names=ACTION_NAMES),
    }
    for key in ('controller_phase', 'difficulty_id', 'loss_weight'):
        features['training.'+key] = dict(dtype='float32', shape=(1,), names=[key])
    destination = output / split
    repo_id = 'local/isaac-future-joint-v24-' + split
    ds = LeRobotDataset.create(repo_id=repo_id, root=destination, fps=10,
        robot_type='franka_future_measured_joint_waypoints', features=features,
        use_videos=False, image_writer_threads=4)
    exported, expected_states, expected_actions, image_samples = [], [], [], {}
    phase_counts = {str(p): 0 for p in range(5)}
    offset = 0
    guard_frames = 0
    try:
        for episode, record in enumerate(records):
            states, actions, rgb, phases, meta = first if episode == 0 else load_episode(record)
            guard_frames += int(np.any(np.abs(actions[:, :7]-states[:, :7]) > .2, axis=1).sum())
            for i, phase in enumerate(phases):
                ds.add_frame({'observation.images.table': rgb[i], 'observation.state': states[i],
                    'action': actions[i], 'training.controller_phase': np.asarray([phase], np.float32),
                    'training.difficulty_id': np.asarray([LEVELS.index(record['level'])], np.float32),
                    'training.loss_weight': np.asarray([1.], np.float32), 'task': meta['task']})
                phase_counts[str(int(phase))] += 1
            ds.save_episode()
            expected_states.append(states)
            expected_actions.append(actions)
            # Check a first/middle/final episode image per split after PNG/parquet export.
            if episode in (0, len(records)//2, len(records)-1):
                image_samples[offset] = rgb[0].copy()
            offset += len(states)
            exported.append(dict(record, frames=len(states), original_frames=record['frames'],
                                 terminal_frames_dropped=1))
            print('FUTURE_JOINT_EXPORT_EPISODE='+json.dumps(dict(split=split, episodes=episode+1,
                                                               total=len(records))), flush=True)
    finally:
        ds.finalize()
    columns = ['index', 'observation.state', 'action', 'observation.images.table']
    table = pa.concat_tables([pq.read_table(p, columns=columns)
        for p in sorted((destination/'data').glob('*/*.parquet'))]).sort_by([('index','ascending')])
    saved_state = np.asarray(table['observation.state'].to_pylist(), np.float32)
    saved_action = np.asarray(table['action'].to_pylist(), np.float32)
    if not np.array_equal(saved_state, np.concatenate(expected_states)) or not np.array_equal(saved_action, np.concatenate(expected_actions)):
        raise ValueError('Export changed current input or future target')
    for i, expected in image_samples.items():
        item = table['observation.images.table'][i].as_py()
        image = Image.open(io.BytesIO(item['bytes'])) if item.get('bytes') else Image.open(item['path'])
        with image:
            if not np.array_equal(np.asarray(image), expected):
                raise ValueError('RGB pixels changed during export')
    manifest = dict(format=FORMAT, split=split, repo_id=repo_id, records=exported,
        episodes=len(records), frames=len(saved_state), state_dim=25, action_dim=8,
        state_names=list(STATE_NAMES), action_names=ACTION_NAMES, interface_contract=contract(),
        action_target_mode='next_decision_measured_arm_q_plus_current_gripper_command',
        action_relabeling=True, actual_recorded_joint_command=False,
        future_horizon_control_steps=5, future_horizon_seconds=.1,
        original_split_preserved=True, terminal_frames_dropped=len(records),
        privileged_object_state=False, visual_pose_conditioning=False, phase_in_policy_input=False,
        controller_phase_frame_counts=phase_counts,
        frames_exceeding_point2rad_deployment_guard=guard_frames,
        level_episode_counts={l:sum(r['level']==l for r in exported) for l in LEVELS},
        level_frame_counts={l:sum(r['frames'] for r in exported if r['level']==l) for l in LEVELS},
        rgb_sample_exact_match_count=len(image_samples))
    (destination/'balanced_split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (destination/'interface_contract.json').write_text(json.dumps(contract(), indent=2), encoding='utf-8')
    return {k:v for k,v in manifest.items() if k!='records'}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-export', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--max-per-level', type=int, default=0, help='Smoke only; 0 exports all original records')
    args = p.parse_args()
    if args.output_root.exists(): raise FileExistsError(args.output_root)
    if args.max_per_level < 0: raise ValueError('Negative episode limit')
    splits = source_records(args.source_export, args.max_per_level)
    reports = [export_split(args.output_root, split, records) for split, records in splits.items()]
    report = dict(status='FUTURE_JOINT_EXPORT_VERIFIED', format=FORMAT, source_export=str(args.source_export.resolve()),
        smoke_subset=bool(args.max_per_level), splits=reports, original_split_preserved=True,
        test_split_used_for_training=False, original_files_modified=False,
        warning='Future measured position waypoints, not original command reconstruction. Deployment retains 0.2 rad guard.')
    (args.output_root/'export_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
