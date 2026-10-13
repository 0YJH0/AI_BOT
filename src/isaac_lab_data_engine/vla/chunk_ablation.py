"""Opt-in initialization and single-factor contract checks for v9 chunk experiments."""
import hashlib
import json
from pathlib import Path


def validate_interface(config, chunk):
    if (config['chunk_size'] != chunk or config['n_action_steps'] != 1 or
            list(config['input_features']['observation.state']['shape']) != [30] or
            list(config['output_features']['action']['shape']) != [8]):
        raise ValueError('Runtime policy does not match the intended 30/8D, chunk N, execution 1 contract')


def audit_initialization(model, output: Path, chunk: int):
    import dataclasses
    import torch
    cfg = dataclasses.asdict(model.config)
    validate_interface(cfg, chunk)
    h = hashlib.sha256()
    names = []
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            h.update(name.encode())
            h.update(str(tuple(parameter.shape)).encode())
            h.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            names.append(name)
    payload = {'chunk_size':chunk, 'n_action_steps':cfg['n_action_steps'],
               'state_dimension':30, 'action_dimension':8,
               'trainable_parameter_sha256':h.hexdigest(), 'trainable_names':names,
               'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
               'pretrained_path':str(cfg['pretrained_path']),
               'stage':'after_loading_v9_before_optimizer_training'}
    with output.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print('CHUNK_ABLATION_INITIALIZATION=' + json.dumps(payload), flush=True)


def compare_train_configs(a, b):
    """Ignore run identity, but reject any second experimental setting."""
    import copy
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    validate_interface(a['policy'], 4)
    validate_interface(b['policy'], 1)
    for cfg in (a, b):
        for key in ('output_dir', 'job_name'):
            cfg.pop(key, None)
        cfg['policy'].pop('chunk_size')
    if a != b:
        keys = [k for k in set(a)|set(b) if a.get(k) != b.get(k)]
        raise ValueError(f'Training differs beyond chunk_size and run identity: {keys}')
