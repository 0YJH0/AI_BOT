"""Project audited successful trajectories to robot-only state without relabeling."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.dataset import STATE_NAMES, DELTA_ACTION_NAMES

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def contract():
    return dict(version="rgb_language_proprio_v22", state_dim=25, action_dim=8,
                state_names=list(STATE_NAMES), action_names=list(DELTA_ACTION_NAMES),
                phase_in_input=False, target_coordinates_in_input=False,
                policy_inputs=["rgb", "language", "robot_proprioception"],
                position_target_mode="delta", action_repeat=5, execution_horizon=1,
                maximum_position_step_m=.04, orientation="model_predicted_quaternion",
                gripper_safety="close_latch_without_target_distance_gate")


def project_state(state):
    state = np.asarray(state, np.float32)
    if state.ndim != 2 or state.shape[1] != 30 or not np.isfinite(state).all():
        raise ValueError("Expected finite audited [N,30] state")
    return state[:, :25].copy()


def export_split(source, destination):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from export_vla_relative4d_v13 import stats
    info = read(source / "meta/info.json")
    manifest = read(source / "balanced_split_manifest.json")
    if manifest.get("state_dim") != 30 or manifest.get("action_dim") != 8:
        raise ValueError("Expected original 30/8 data")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "meta", destination / "meta")
    episode_states, all_states, image_frames = {}, [], 0
    source_hashes = {}
    for path in sorted((source / "data").glob("*/*.parquet")):
        source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        table = pq.read_table(path)
        state = project_state(table["observation.state"].to_pylist())
        updated = table.set_column(table.schema.get_field_index("observation.state"),
            "observation.state", pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1), type=pa.float32()), 25))
        metadata = dict(updated.schema.metadata or {})
        if b"huggingface" in metadata:
            hf = json.loads(metadata[b"huggingface"])
            hf["info"]["features"]["observation.state"]["length"] = 25
            metadata[b"huggingface"] = json.dumps(hf).encode()
        updated = updated.replace_schema_metadata(metadata)
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, output, compression="zstd")
        persisted = pq.read_table(output)
        for name in table.column_names:
            if name != "observation.state" and not table[name].equals(persisted[name]):
                raise ValueError(f"Unchanged column differs: {name}")
        if not np.array_equal(np.asarray(persisted["observation.state"].to_pylist(), np.float32), state):
            raise ValueError("State projection changed during persistence")
        episodes = np.asarray(table["episode_index"].to_numpy())
        for episode in np.unique(episodes):
            episode_states.setdefault(int(episode), []).append(state[episodes == episode])
        all_states.append(state)
        image_frames += len(table)
        print(f"PROPRIO_EXPORT_FILE={source.name}/{path.name} frames={image_frames}", flush=True)
    ep_stats = {idx: stats(np.concatenate(values)) for idx, values in episode_states.items()}
    for path in sorted((destination / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        indices = table["episode_index"].to_pylist()
        for kind in ep_stats[indices[0]]:
            key = "stats/observation.state/" + kind
            table = table.set_column(table.schema.get_field_index(key), key,
                pa.array([ep_stats[idx][kind] for idx in indices], type=table.schema.field(key).type))
        pq.write_table(table, path)
    info["features"]["observation.state"].update(shape=[25], names=list(STATE_NAMES))
    info["robot_type"] = "franka_rgb_language_proprio_delta_ik"
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    ds_stats = read(source / "meta/stats.json")
    ds_stats["observation.state"] = stats(np.concatenate(all_states))
    (destination / "meta/stats.json").write_text(json.dumps(ds_stats, indent=2), encoding="utf-8")
    manifest.update(state_dim=25, state_names=list(STATE_NAMES), phase_in_policy_input=False,
        visual_pose_conditioning=False, visual_target_source="not_in_student_input",
        student_phase_input="none; diagnostic metadata is sampling-only",
        interface_contract=contract(), action_relabeling=False,
        source_export=str(source.resolve()),
        source_manifest_sha256=hashlib.sha256((source / "balanced_split_manifest.json").read_bytes()).hexdigest(),
        repo_id="local/isaac-proprio-v22-" + source.name)
    (destination / "balanced_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "interface_contract.json").write_text(json.dumps(contract(), indent=2), encoding="utf-8")
    return dict(split=source.name, episodes=len(episode_states), frames=image_frames,
        rgb_bytes_unchanged=True, actions_unchanged=True, language_unchanged=True,
        scene_split_preserved=True, state_columns=list(range(25)),
        state_normalization_recomputed=True, action_normalization_unchanged=True,
        source_parquet_sha256=source_hashes)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = read(args.source / "parquet_audit.json")
    if (audit["status"] != "EXECUTION_ALIGNED_PARQUET_AUDIT_OK"
            or not audit["scene_groups_disjoint"] or not audit["no_posthoc_action_relabeling"]):
        raise ValueError("Validated source dataset required")
    results = [export_split(args.source / s, args.output / s) for s in ("train", "validation", "test")]
    (args.output / "projection_audit.json").write_text(json.dumps(dict(
        status="PROPRIO25_PROJECTION_VERIFIED", interface_contract=contract(), splits=results,
        source_export=str(args.source.resolve()), test_split_used_for_training=False), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
