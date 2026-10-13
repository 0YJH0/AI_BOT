"""Create an immutable feature projection of validated data; copy RGB bytes verbatim."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.relative_interface import (
    contract, relative_student_state, project_executed_action,
    RELATIVE_STATE_NAMES, RELATIVE_ACTION_NAMES)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    result = {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
              "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [len(x)]}
    for q in (.01, .10, .50, .90, .99):
        result["q" + str(round(q*100)).zfill(2)] = np.quantile(x, q, axis=0).tolist()
    return result


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def export_split(source, destination):
    info, manifest = read(source / "meta/info.json"), read(source / "balanced_split_manifest.json")
    if manifest.get("state_dim") != 30 or manifest.get("action_dim") != 8:
        raise ValueError("Expected audited original 30/8-D data")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "meta", destination / "meta")
    episodes = {}
    all_s, all_a = [], []
    image_frames = 0
    for path in sorted((source / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        s = np.asarray(table["observation.state"].to_pylist(), np.float32)
        a = np.asarray(table["action"].to_pylist(), np.float32)
        ns, na = relative_student_state(s), project_executed_action(a)
        if not np.array_equal(na, a[:, [0, 1, 2, 7]]):
            raise ValueError("Action projection changed executed labels")
        updated = table
        for key, values, width in (("observation.state", ns, 29), ("action", na, 4)):
            array = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), width)
            updated = updated.set_column(updated.schema.get_field_index(key), key, array)
        metadata = dict(updated.schema.metadata or {})
        if b"huggingface" in metadata:
            hf = json.loads(metadata[b"huggingface"])
            hf["info"]["features"]["observation.state"]["length"] = 29
            hf["info"]["features"]["action"]["length"] = 4
            metadata[b"huggingface"] = json.dumps(hf).encode()
        updated = updated.replace_schema_metadata(metadata)
        output = destination / path.relative_to(source)
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(updated, output, compression="zstd")
        persisted = pq.read_table(output, columns=["observation.images.table", "observation.state", "action"])
        if not persisted["observation.images.table"].equals(table["observation.images.table"]):
            raise ValueError("RGB byte stream changed")
        if (not np.array_equal(np.asarray(persisted["action"].to_pylist(), np.float32), na)
                or not np.array_equal(np.asarray(persisted["observation.state"].to_pylist(), np.float32), ns)):
            raise ValueError("Persisted features differ")
        ep = np.asarray(table["episode_index"].to_numpy())
        for idx in np.unique(ep):
            record = episodes.setdefault(int(idx), {"s": [], "a": []})
            record["s"].append(ns[ep == idx]); record["a"].append(na[ep == idx])
        all_s.append(ns); all_a.append(na); image_frames += len(table)
        print("RELATIVE_EXPORT_FILE=" + json.dumps({"split": source.name, "file": path.name,
              "frames": image_frames}), flush=True)
    episode_stats = {idx: {"observation.state": stats(np.concatenate(v["s"])),
                           "action": stats(np.concatenate(v["a"]))} for idx, v in episodes.items()}
    for path in sorted((destination / "meta/episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        indices = table["episode_index"].to_pylist()
        for key in ("observation.state", "action"):
            for kind in episode_stats[indices[0]][key]:
                column = "stats/" + key + "/" + kind
                values = [episode_stats[idx][key][kind] for idx in indices]
                table = table.set_column(table.schema.get_field_index(column), column,
                                         pa.array(values, type=table.schema.field(column).type))
        pq.write_table(table, path)
    info["features"]["observation.state"].update(shape=[29], names=list(RELATIVE_STATE_NAMES))
    info["features"]["action"].update(shape=[4], names=list(RELATIVE_ACTION_NAMES))
    info["robot_type"] = "franka_relative_xyz_gripper_fixed_top_down"
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    ds_stats = read(source / "meta/stats.json")
    ds_stats["observation.state"] = stats(np.concatenate(all_s))
    ds_stats["action"] = stats(np.concatenate(all_a))
    (destination / "meta/stats.json").write_text(json.dumps(ds_stats, indent=2), encoding="utf-8")
    manifest.update(state_dim=29, action_dim=4, action_names=list(RELATIVE_ACTION_NAMES),
                    interface_contract=contract(), phase_in_policy_input=False,
                    action_relabeling=False, action_projection="exact original columns 0,1,2,7",
                    student_phase_input="none; training.controller_phase is sampling/diagnostics only",
                    source_export=str(source.resolve()),
                    source_manifest_sha256=hashlib.sha256((source / "balanced_split_manifest.json").read_bytes()).hexdigest(),
                    orientation_caveat="Original REST quaternion varied; new controller fixes it. Teacher pilot is required; full old rollouts are not claimed quaternion-identical.")
    manifest["repo_id"] = "local/isaac-relative4d-v13-" + source.name
    (destination / "balanced_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "interface_contract.json").write_text(json.dumps(contract(), indent=2), encoding="utf-8")
    return {"split": source.name, "episodes": len(episodes), "frames": image_frames,
            "rgb_bytes_unchanged": True, "executed_xyz_gripper_unchanged": True,
            "normalization_recomputed": True, "scene_split_preserved": True}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    source_audit = read(a.source / "parquet_audit.json")
    if source_audit["status"] != "EXECUTION_ALIGNED_PARQUET_AUDIT_OK":
        raise ValueError("Original export audit must pass")
    results = [export_split(a.source / s, a.output / s) for s in ("train", "validation", "test")]
    (a.output / "projection_audit.json").write_text(json.dumps({
        "status": "RELATIVE4D_PROJECTION_VERIFIED", "splits": results,
        "interface_contract": contract(), "action_relabeling": False}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
