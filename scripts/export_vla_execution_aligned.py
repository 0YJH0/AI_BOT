"""Export collected causal demonstrations verbatim, without action relabeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.dataset import DELTA_ACTION_NAMES, VISUAL_FSM_CONDITIONED_STATE_NAMES
from isaac_lab_data_engine.vla.full_trajectory import FORMAT, validate_full_trajectory

LEVELS = ("easy", "medium", "hard")


def load_episode(path):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_full_trajectory(arrays, metadata)
    return arrays, metadata


def scene_group(metadata):
    domain = metadata["domain"]
    physical = {k: domain[k] for k in ("object", "camera", "lighting", "collection_initial_joint_position")}
    return hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest()


def select_splits(root, train_count, validation_count, test_count, seed):
    rng = np.random.default_rng(seed)
    splits = {s: [] for s in ("train", "validation", "test")}
    seen = set()
    counts = (train_count, validation_count, test_count)
    if train_count < 1 or validation_count < 1 or test_count < 0:
        raise ValueError("Train/validation must contain episodes; test may be empty for a pilot")
    for level in LEVELS:
        records = []
        for path in sorted((root / level).glob("episode_*.npz")):
            _, metadata = load_episode(path)
            group = scene_group(metadata)
            if group in seen:
                raise ValueError("Duplicate physical scene / initial robot state; cannot split safely")
            seen.add(group)
            records.append((level, path, group))
        if len(records) < sum(counts):
            raise ValueError(f"{level}: {len(records)} successful episodes; need {sum(counts)}")
        order = rng.permutation(len(records))[:sum(counts)]
        selected = [records[i] for i in order]
        start = 0
        for split, n in zip(splits, counts):
            splits[split].extend(selected[start:start+n])
            start += n
    return splits


def export_split(output, split, records, prefix):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first, _ = load_episode(records[0][1])
    teacher_modes = {
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8")).get("teacher_mode", "visual")
        for _, path, _ in records
    }
    if len(teacher_modes) != 1:
        raise ValueError("Cannot silently mix truth and visual teachers")
    teacher_mode = next(iter(teacher_modes))
    h, w, c = first["rgb"].shape[1:]
    features = {
        "observation.images.table": {"dtype": "image", "shape": (h, w, c), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (30,), "names": list(VISUAL_FSM_CONDITIONED_STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (8,), "names": list(DELTA_ACTION_NAMES)},
    }
    for k in ("controller_phase", "difficulty_id", "loss_weight"):
        features[f"training.{k}"] = {"dtype": "float32", "shape": (1,), "names": [k]}
    dataset = LeRobotDataset.create(
        repo_id=f"{prefix}-{split}", fps=10, root=output / split,
        robot_type="franka_isaac_lab_delta_ik_execution_aligned",
        features=features, use_videos=False, image_writer_threads=4,
    )
    manifest_records = []
    phase_counts = {str(p): 0 for p in range(5)}
    try:
        for level, path, group in records:
            arrays, metadata = load_episode(path)
            for i, phase in enumerate(arrays["controller_phase"]):
                weight = (2.0 if phase == 4 else 1.0) * (1.25 if level == "hard" else 1.0)
                dataset.add_frame({
                    "observation.images.table": arrays["rgb"][i],
                    "observation.state": arrays["state"][i],
                    "action": arrays["action"][i],  # Exact executed controller input. NO reconstruction.
                    "training.controller_phase": np.asarray([phase], dtype=np.float32),
                    "training.difficulty_id": np.asarray([LEVELS.index(level)], dtype=np.float32),
                    "training.loss_weight": np.asarray([weight], dtype=np.float32),
                    "task": metadata["task"],
                })
                phase_counts[str(int(phase))] += 1
            dataset.save_episode()
            manifest_records.append({
                "level": level, "source": str(path.resolve()), "frames": len(arrays["state"]),
                "scene_group": group, "sample_key": metadata["sample_key"],
                "full_trajectory": True, "dagger_recovery": False,
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "maximum_lift_cm": metadata["maximum_lift_cm"],
            })
    finally:
        dataset.finalize()
    manifest = {
        "format": FORMAT, "split": split, "repo_id": f"{prefix}-{split}",
        "episodes": len(records), "frames": sum(r["frames"] for r in manifest_records),
        "action_target_mode": (
            "execution_aligned_truth_teacher_delta" if teacher_mode == "truth" else "visual_fsm_bounded_delta"
        ), "action_relabeling": False,
        "action_target_semantics": {"maximum_position_step_m": .04, "approach_height_m": .12, "lift_height_m": .18},
        "state_dim": 30, "action_dim": 8, "action_names": list(DELTA_ACTION_NAMES),
        "privileged_object_state": False, "visual_pose_conditioning": True,
        "visual_target_source": "live_estimate_latched_at_collection",
        "teacher_mode": teacher_mode,
        "teacher_target_coordinates_in_policy_input": False,
        "student_phase_input": "feedback_phase_id; deployment must use visual-only feedback",
        "controller_phase_frame_counts": phase_counts,
        "level_episode_counts": {l: sum(r["level"] == l for r in manifest_records) for l in LEVELS},
        "level_frame_counts": {l: sum(r["frames"] for r in manifest_records if r["level"] == l) for l in LEVELS},
        "causal_alignment": "10 Hz observation before action; original 5-step deployed execution",
        "rgb_shape": [h, w, c], "records": manifest_records,
        "evaluation_caveat": "Success-conditioned demonstrations are not an unbiased benchmark; Easy object parameters remain fixed. Evaluate VLA without truth targets or truth-based phase decisions.",
    }
    (output / split / "balanced_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("EXECUTION_ALIGNED_EXPORT=" + json.dumps({k: v for k, v in manifest.items() if k != "records"}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-prefix", default="local/isaac-pick-lift-execution-aligned-v8")
    parser.add_argument("--train-per-level", type=int, default=170)
    parser.add_argument("--validation-per-level", type=int, default=15)
    parser.add_argument("--test-per-level", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    splits = select_splits(args.source_root, args.train_per_level, args.validation_per_level, args.test_per_level, args.seed)
    summaries = [export_split(args.output_root, split, records, args.repo_prefix)
                 for split, records in splits.items() if records]
    (args.output_root / "export_audit.json").write_text(json.dumps({
        "status": "EXECUTION_ALIGNED_EXPORT_OK",
        "source_root": str(args.source_root.resolve()),
        "no_posthoc_action_relabeling": True,
        "scene_groups_disjoint": True,
        "splits": [{k: v for k, v in s.items() if k != "records"} for s in summaries],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
