"""Freeze data/weights, verify paired training contracts, and compare final chunk experiments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from isaac_lab_data_engine.vla.chunk_ablation import compare_train_configs
from audit_vla_v9_regression import CHECKPOINT, EXPECTED_HASH, digest, lock_files, read, verify_lock
from compare_vla_close_lift_pair import load_arm, SCENE_FIELDS

EXPORT = ROOT/'exports/lerobot/pick_lift_execution_aligned_20261011_truth_v9'
SAMPLING = ROOT/'results/vla/execution_aligned_20261011_truth_v9/train_sampling_plan.json'


def checkpoint(root, chunk):
    plan = read(root/'experiment_plan.json')
    return ROOT/'results/vla/training'/plan[f'run_chunk{chunk}']/'checkpoints/006000/pretrained_model'


def freeze(root):
    if digest(CHECKPOINT/'adapter_model.safetensors') != EXPECTED_HASH:
        raise ValueError('Wrong v9 checkpoint')
    files = lock_files()
    extra = [SAMPLING, SAMPLING.with_suffix('.weights.npy')]
    for split in ('train','validation'):
        extra += list((EXPORT/split).rglob('*'))
    extra += [ROOT/'scripts'/name for name in ('lerobot_train_windows.py',
              'train_smolvla_dagger_recovery_8gb_v7.ps1','run_vla_chunk_ablation_v17.ps1',
              'audit_vla_chunk_ablation_v17.py','scan_smolvla_checkpoints.py')]
    for path in extra:
        if path.is_file(): files[str(path.relative_to(ROOT))] = digest(path)
    with (root/'input_lock.json').open('x',encoding='utf-8') as f:
        json.dump({'files':files,'checkpoint_sha256':EXPECTED_HASH},f,indent=2)
    print('CHUNK_EXPERIMENT_INPUTS_LOCKED')


def verify_training(root):
    verify_lock(read(root/'input_lock.json'))
    a, b = (checkpoint(root,n) for n in (4,1))
    compare_train_configs(read(a/'train_config.json'),read(b/'train_config.json'))
    ia, ib = (read(root/f'chunk{n}_initialization.json') for n in (4,1))
    if (ia['trainable_parameter_sha256'] != ib['trainable_parameter_sha256'] or
            ia['trainable_names'] != ib['trainable_names'] or
            ia['pretrained_path'] != ib['pretrained_path'] or
            ia['trainable_parameters'] != 742656 or ib['trainable_parameters'] != 742656):
        raise ValueError('Initial trainable weights do not match')
    if read(a/'training_objective.json') != read(b/'training_objective.json'):
        raise ValueError('Loss/sampling objectives differ')
    processor_files = [p.name for p in a.glob('policy_*')]
    for name in processor_files:
        if digest(a/name) != digest(b/name):
            raise ValueError(f'Normalization/processor mismatch: {name}')
    va,vb=(read(root/f'chunk{n}_validation.json') for n in (4,1))
    if va['indices'] != vb['indices'] or va['seed'] != vb['seed']:
        raise ValueError('Validation frames or seed differ')
    result={'status':'CHUNK_TRAINING_CONTRACT_VERIFIED','same_initial_adapter_tensors':True,
            'same_processors':True,'same_loss_and_sampler':True,'same_validation_frames':True,
            'only_training_config_difference':'chunk_size',
            'budget':'6000 optimizer updates per arm, batch=1; not equal wall time or future target count',
            'validation':{f'chunk{n}':read(root/f'chunk{n}_validation.json')['results'] for n in (4,1)},
            'caveat':'Chunk4 predicts/supervises more future actions; stochastic noise shapes differ. Same seeds do not guarantee bit-identical noise or augmentations.'}
    (root/'training_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result['status'])


def summarize(root):
    verify_training(root)
    arms={}
    for n in (4,1):
        reports=list((root/f'chunk{n}_closed_loop').glob('visual_only_benchmark_*/vla_benchmark_report.json'))
        if len(reports)!=1: raise ValueError('Exactly one completed benchmark required per arm')
        report,context,rows=load_arm(reports[0].parent)
        if context['checkpoint_sha256'].lower()!=digest(checkpoint(root,n)/'adapter_model.safetensors'):
            raise ValueError('Benchmark loaded a different checkpoint')
        arms[n]=(report,context,rows)
    a, b = arms[4], arms[1]
    for key in set(a[0]['config'])|set(b[0]['config']):
        if key!='vla_server' and a[0]['config'].get(key)!=b[0]['config'].get(key):
            raise ValueError(f'Execution config mismatch: {key}')
    if a[1]['inference_seed'] != b[1]['inference_seed'] or set(a[2]) != set(b[2]):
        raise ValueError('Inference seed or scene key mismatch')
    pairs=[]
    for key in a[2]:
        for field in SCENE_FIELDS:
            if a[2][key][0][field]!=b[2][key][0][field]: raise ValueError(f'Scene mismatch: {key}/{field}')
        pairs.append({'sample_key':key,'chunk4':a[2][key][1],'chunk1':b[2][key][1]})
    payload={'status':'CHUNK_ABLATION_COMPLETE','pairs':pairs,
             'successes':{f'chunk{n}':sum(p[f'chunk{n}']['success'] for p in pairs) for n in (4,1)},
             'episodes_per_arm':6,'selection':'Final 6000 checkpoint precommitted; not best-of-scan selection',
             'limitations':'Single training seed, six scene pairs; diagnostic not definitive population inference. Same global inference seed, not forced common noise.'}
    (root/'comparison.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'status':payload['status'],'successes':payload['successes']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=('freeze','verify','summarize'),required=True)
    args=p.parse_args()
    {'freeze':freeze,'verify':verify_training,'summarize':summarize}[args.stage](args.root)
