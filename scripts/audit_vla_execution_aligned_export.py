"""Prove that exported policy states/actions equal the collected causal pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audit(root):
    all_groups = set()
    summaries = {}
    teacher_modes = set()
    for split in ("train", "validation", "test"):
        path = root / split / "balanced_split_manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        teacher_modes.add(manifest.get("teacher_mode", "visual"))
        info = json.loads((root / split / "meta/info.json").read_text(encoding="utf-8"))
        if any("truth" in feature or "true_object" in feature or "teacher_control" in feature
               for feature in info["features"]):
            raise ValueError("Privileged teacher diagnostics were exported as model features")
        if manifest.get("teacher_target_coordinates_in_policy_input", False):
            raise ValueError("Truth teacher coordinates must never become student input")
        if manifest.get("action_relabeling") is not False:
            raise ValueError("Export must preserve labels, not reconstruct them")
        groups = {r["scene_group"] for r in manifest["records"]}
        if groups & all_groups or len(groups) != len(manifest["records"]):
            raise ValueError("Scene leakage between splits")
        all_groups |= groups
        rows = []
        for parquet in sorted((root / split / "data").glob("chunk-*/*.parquet")):
            rows.extend(pq.read_table(parquet, columns=[
                "episode_index", "frame_index", "observation.state", "action",
            ]).to_pylist())
        if len(rows) != manifest["frames"]:
            raise ValueError("Frame count mismatch")
        for i, record in enumerate(manifest["records"]):
            selected = sorted((r for r in rows if r["episode_index"] == i), key=lambda r: r["frame_index"])
            if [r["frame_index"] for r in selected] != list(range(record["frames"])):
                raise ValueError("Noncontiguous exported frames")
            with np.load(record["source"], allow_pickle=False) as source:
                if manifest.get("teacher_mode") == "truth":
                    metadata = json.loads(Path(record["source"]).with_suffix(".json").read_text(encoding="utf-8"))
                    visual = np.asarray(metadata["student_visual_cache_b"], dtype=np.float32)
                    if not np.array_equal(source["state"][:, 25:28], np.repeat(visual[None], record["frames"], axis=0)):
                        raise ValueError("Student coordinates do not match the recorded visual estimate")
                for feature, key in (("observation.state", "state"), ("action", "action")):
                    exported = np.asarray([r[feature] for r in selected], dtype=np.float32)
                    if not np.array_equal(exported, source[key]):
                        raise ValueError(f"Collected/exported {feature} mismatch: {record['source']}")
        summaries[split] = {
            "episodes": manifest["episodes"], "frames": len(rows),
            "maximum_state_error": 0, "maximum_action_error": 0,
        }
    if not {"train", "validation"} <= set(summaries):
        raise ValueError("Missing training/validation splits")
    return {"status": "EXECUTION_ALIGNED_PARQUET_AUDIT_OK",
            "scene_groups_disjoint": True, "no_posthoc_action_relabeling": True,
            "teacher_modes": sorted(teacher_modes), "teacher_coordinates_excluded_from_student_input": True,
            "splits": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "parquet_audit.json"
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.root)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
