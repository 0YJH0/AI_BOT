"""Copy a rank-16 adapter into rank-32 without changing its initial LoRA function."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def expand_pair(a, b, rank, generator):
    old = a.shape[0]
    if b.shape[1] != old or rank <= old:
        raise ValueError("Invalid rank expansion")
    aa = torch.empty((rank, a.shape[1]), dtype=a.dtype)
    torch.nn.init.kaiming_uniform_(aa, a=5**0.5, generator=generator)
    bb = torch.zeros((b.shape[0], rank), dtype=b.dtype)
    aa[:old], bb[:, :old] = a, b
    return aa, bb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads((source/'adapter_config.json').read_text())
    if (cfg['r'] != 16 or cfg['lora_alpha'] != 8 or cfg['use_rslora'] or
            cfg.get('rank_pattern') or cfg.get('alpha_pattern') or cfg.get('use_dora')):
        raise ValueError("Expected uniform vanilla r16 alpha8 adapter")
    original = load_file(str(source/'adapter_model.safetensors'))
    weights = dict(original)
    rng = torch.Generator().manual_seed(1000)
    pairs = []
    for key, a in original.items():
        if '.lora_A.' not in key:
            continue
        bkey = key.replace('.lora_A.', '.lora_B.')
        aa, bb = expand_pair(a, original[bkey], 32, rng)
        weights[key], weights[bkey] = aa, bb
        # Verify every layer's effective weight and deterministic projection output.
        old_delta = original[bkey].float() @ a.float() * 0.5
        new_delta = bb.float() @ aa.float() * 0.5
        torch.testing.assert_close(old_delta, new_delta, rtol=1e-5, atol=1e-7)
        x = torch.randn((3, a.shape[1]), generator=rng)
        before = (x @ a.float().T) @ original[bkey].float().T * 0.5
        after = (x @ aa.float().T) @ bb.float().T * 0.5
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        pairs.append(dict(name=key, maximum_delta_error=(old_delta-new_delta).abs().max().item(),
                          maximum_projection_error=(before-after).abs().max().item()))
    if len(pairs) != 37 or sum(v.numel() for v in weights.values()) != 1485312:
        raise ValueError("Unexpected adapter structure")
    shutil.copytree(source, output)
    cfg['r'], cfg['lora_alpha'] = 32, 16
    (output/'adapter_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    save_file(weights, str(output/'adapter_model.safetensors'))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit = dict(source=str(source), source_sha256=digest(source/'adapter_model.safetensors'),
                 output_sha256=digest(output/'adapter_model.safetensors'), rank=32, alpha=16,
                 trainable_parameters=1485312, initialization='old blocks preserved; extra A random, extra B zero',
                 verification='all 37 effective LoRA matrices and layer projection outputs; not full-policy rollout',
                 layers=pairs)
    (output/'rank_expansion_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('RANK_EXPANSION_VERIFIED=' + json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
