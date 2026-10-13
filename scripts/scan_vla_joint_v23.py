"""Fixed episode-disjoint validation of absolute joint targets (radians, not centimeters)."""
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from isaac_lab_data_engine.vla.joint_interface import contract
from scan_smolvla_checkpoints import load_policy, build_inference_batch, deterministic_action_noise, to_numpy, atomic_write_json


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,nargs='+',required=True)
    p.add_argument('--samples',type=int,default=240)
    p.add_argument('--seed',type=int,default=20261011)
    args=p.parse_args()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if args.dataset_root.name != 'validation': raise ValueError('Only independent validation split accepted')
    manifests=json.loads((args.dataset_root/'balanced_split_manifest.json').read_text(encoding='utf-8'))
    if manifests['interface_contract']!=contract(): raise ValueError('Wrong action semantics')
    checkpoints=[args.checkpoint_root/f'{step:06d}'/'pretrained_model' for step in args.steps]
    configs=[]
    for path in checkpoints:
        if json.loads((path/'interface_contract.json').read_text(encoding='utf-8'))!=contract():
            raise ValueError('Missing joint contract in checkpoint')
        cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
        if cfg['input_features']['observation.state']['shape']!=[25] or cfg['output_features']['action']['shape']!=[8]:
            raise ValueError('Feature contract mismatch')
        configs.append(cfg)
    chunks={c['chunk_size'] for c in configs}
    if len(chunks)!=1: raise ValueError('Checkpoint chunks differ')
    chunk=chunks.pop()
    ds=LeRobotDataset(manifests['repo_id'],root=args.dataset_root,
                      delta_timestamps={'action':[k/10 for k in range(chunk)]})
    indices=sorted(np.random.default_rng(args.seed).choice(len(ds),min(args.samples,len(ds)),replace=False).tolist())
    cached={i:ds[i] for i in indices}
    levels={i:r['level'] for i,r in enumerate(manifests['records'])}
    results=[]
    payload=dict(status='JOINT_VALIDATION_RUNNING',dataset_split='validation',indices=indices,
        seed=args.seed,interface_contract=contract(),results=results,
        evaluation_scope='fixed episode-disjoint validation frames; no test split used',
        metric_warning='Joint MAE is radians/degrees, NOT Cartesian positioning error or grasp success')
    for step,path in zip(args.steps,checkpoints):
        policy,pre,post=load_policy(path,'cuda')
        errors=[]; baseline=[]; grip=[]; by_level={}; by_phase={}
        for offset,i in enumerate(indices):
            sample=cached[i]
            batch=pre(build_inference_batch(torch,[sample]))
            noise=deterministic_action_noise(torch,[i],seed=args.seed,chunk_size=chunk,max_action_dim=policy.config.max_action_dim,device=batch['observation.state'].device)
            policy.reset()
            with torch.inference_mode(): predicted=to_numpy(post(policy.predict_action_chunk(batch,noise=noise)))[0,0]
            target=to_numpy(sample['action'])
            if target.ndim==2: target=target[0]
            if predicted.shape!=(8,) or not np.isfinite(predicted).all(): raise ValueError('Invalid joint prediction')
            error=np.abs(predicted[:7]-target[:7])
            base=np.abs(to_numpy(sample['observation.state'])[:7]-target[:7])
            correct=bool((predicted[7]<0)==(target[7]<0))
            level=levels[int(to_numpy(sample['episode_index']).item())]
            phase=str(int(to_numpy(sample['training.controller_phase']).reshape(-1)[0]))
            errors.append(error); baseline.append(base); grip.append(correct)
            by_level.setdefault(level,[]).append(float(error.mean()))
            by_phase.setdefault(phase,[]).append((float(error.mean()),correct))
            if (offset+1)%40==0: print(f'JOINT_VALIDATION_PROGRESS step={step} samples={offset+1}/{len(indices)}',flush=True)
        errors=np.asarray(errors)
        result=dict(step=step,samples=len(indices),joint_mae_rad=float(errors.mean()),
            joint_mae_deg=float(np.degrees(errors.mean())),joint_axis_mae_rad=errors.mean(0).tolist(),
            joint_error_p90_rad=float(np.percentile(errors.mean(1),90)),
            hold_current_q_baseline_mae_rad=float(np.mean(baseline)),gripper_sign_accuracy=float(np.mean(grip)),
            joint_mae_rad_by_level={k:float(np.mean(v)) for k,v in by_level.items()},
            phase_metrics={k:dict(samples=len(v),joint_mae_rad=float(np.mean([x[0] for x in v])),gripper_sign_accuracy=float(np.mean([x[1] for x in v]))) for k,v in by_phase.items()},finite_rate=1.)
        results.append(result)
        payload['best_checkpoint']=min(results,key=lambda r:r['joint_mae_rad'])
        atomic_write_json(args.output,payload)
        print('JOINT_CHECKPOINT_RESULT='+json.dumps(result),flush=True)
        del policy,pre,post; gc.collect(); torch.cuda.empty_cache()
    payload['status']='JOINT_CHECKPOINT_SCAN_COMPLETE'
    atomic_write_json(args.output,payload)

if __name__=='__main__': main()
