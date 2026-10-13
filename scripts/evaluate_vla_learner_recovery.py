"""Compare checkpoints on held-out learner-visited recovery states only."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.scan_smolvla_checkpoints import evaluate_checkpoint, load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.labels):
        raise ValueError("Each checkpoint needs one label")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.dataset_root / "balanced_split_manifest.json").read_text())
    if manifest["split"] != "validation":
        raise ValueError("Learner recovery comparison requires held-out validation data")
    indices, expected_episodes, levels = [], {}, {}
    offset = 0
    for episode_id, record in enumerate(manifest["records"]):
        levels[episode_id] = record["level"]
        for frame_id in record["learner_visited_frame_indices"]:
            if not 0 <= frame_id < record["frames"]:
                raise ValueError("Invalid learner frame index")
            indices.append(offset + frame_id)
            expected_episodes[offset + frame_id] = episode_id
        offset += record["frames"]
    if not indices:
        raise ValueError("No learner-visited validation states")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root,
                            delta_timestamps={"action": [i / info["fps"] for i in range(4)]})
    if len(dataset) != offset:
        raise ValueError("Manifest/dataset frame mismatch")
    samples = {i: dataset[i] for i in indices}
    for i, sample in samples.items():
        if int(sample["episode_index"].item()) != expected_episodes[i]:
            raise ValueError("Manifest/dataset episode order mismatch")
    results = []
    for label, checkpoint in zip(args.labels, args.checkpoints, strict=True):
        started = time.perf_counter()
        policy, preprocess, postprocess = load_policy(checkpoint, "cuda")
        result = evaluate_checkpoint(
            torch=torch, checkpoint_step=int(checkpoint.parent.name), policy=policy,
            preprocess=preprocess, postprocess=postprocess, dataset=dataset,
            selected_indices=indices, cached_samples=samples, episode_levels=levels,
            seed=args.seed, batch_size=2, progress_every=20, checkpoint_started=started,
            position_target_mode="delta")
        results.append({"label": label, "checkpoint": str(checkpoint.resolve()), **result})
        print("LEARNER_RECOVERY_RESULT=" + json.dumps(results[-1]), flush=True)
        del policy, preprocess, postprocess
        gc.collect()
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "status": "VLA_LEARNER_RECOVERY_COMPARISON_COMPLETE", "seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()), "indices": indices,
        "scope": "held-out learner-visited states; teacher-only recovery frames excluded",
        "results": results}, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
