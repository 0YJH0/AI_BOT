"""Audit exported delta labels and prepare immutable replay inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.dataset import build_visual_fsm_bounded_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest_path = args.dataset_root / "balanced_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["action_target_mode"] != "visual_fsm_bounded_delta":
        raise ValueError("Expected the current delta-label dataset")
    tables = [pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
              for path in sorted((args.dataset_root / "data").glob("chunk-*/*.parquet"))]
    rows = [row for table in tables for row in table.to_pylist()]
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
    actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
    semantics = manifest["action_target_semantics"]
    regenerated = build_visual_fsm_bounded_actions(
        states[:, 18:25], states[:, 25:28], states[:, 29],
        maximum_position_step_m=semantics["maximum_position_step_m"],
        approach_height_m=semantics["approach_height_m"],
        lift_height_m=semantics["lift_height_m"], position_target_mode="delta")
    maximum_error = float(np.max(np.abs(regenerated - actions)))
    if maximum_error > 1e-6:
        raise ValueError(f"Exported actions do not match the label rule: {maximum_error}")
    arrays = {}
    examples = []
    for level in ("easy", "medium", "hard"):
        episode_id = next(i for i, record in enumerate(manifest["records"])
                          if record["level"] == level and not record.get("dagger_recovery", False))
        record = manifest["records"][episode_id]
        selected = sorted((row for row in rows if row["episode_index"] == episode_id),
                          key=lambda row: row["frame_index"])
        episode_states = np.asarray([row["observation.state"] for row in selected], dtype=np.float32)
        episode_actions = np.asarray([row["action"] for row in selected], dtype=np.float32)
        source = Path(record["source"])
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        domain = json.loads((source / "domain_params.json").read_text(encoding="utf-8"))
        arrays[f"{level}_states"] = episode_states
        arrays[f"{level}_actions"] = episode_actions
        arrays[f"{level}_domain_json"] = np.asarray(json.dumps(domain))
        arrays[f"{level}_source"] = np.asarray(str(source))
        arrays[f"{level}_episode_id"] = np.asarray(episode_id)
        arrays[f"{level}_frame_indices"] = np.asarray([row["frame_index"] for row in selected])
        examples.append({"level": level, "episode_index": episode_id, "source": str(source),
                         "frames": len(selected), "source_success": metadata["success"],
                         "source_maximum_lift_cm": 100 * (metadata["maximum_object_height"] - metadata["initial_object_height"]),
                         "phases": dict(Counter(str(int(p)) for p in episode_states[:, 29]))})
    arrays["semantics_json"] = np.asarray(json.dumps(semantics))
    args.output_dir.mkdir(parents=True)
    bundle_path = args.output_dir / "replay_bundle.npz"
    np.savez_compressed(bundle_path, **arrays)
    result = {"dataset_root": str(args.dataset_root.resolve()), "frames_audited": len(rows),
              "maximum_absolute_label_reconstruction_error": maximum_error,
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
              "bundle": str(bundle_path.resolve()), "examples": examples,
              "scope": "Label equality audit and replay preparation only; not a closed-loop success result"}
    (args.output_dir / "label_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
