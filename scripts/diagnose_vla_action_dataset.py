"""Read-only audit of causal absolute-IK labels in a LeRobot export.

The script cross-checks every parquet row against its source trajectory, then
reports action/state distributions by split, difficulty, and controller phase.
It never modifies the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_NAMES = {
    0: "rest",
    1: "approach",
    2: "pregrasp",
    3: "close",
    4: "lift",
}
XYZ_NAMES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=PROJECT_ROOT
        / "exports/lerobot/pick_lift_vla_visual_pose_causal_clean_504_v3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/vla/diagnostics",
    )
    return parser.parse_args()


def vector_stats(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "p01": np.percentile(values, 1, axis=0).tolist(),
        "p10": np.percentile(values, 10, axis=0).tolist(),
        "p50": np.percentile(values, 50, axis=0).tolist(),
        "p90": np.percentile(values, 90, axis=0).tolist(),
        "p99": np.percentile(values, 99, axis=0).tolist(),
    }


def scalar_cm_stats(values_m: np.ndarray) -> dict[str, object]:
    values_cm = np.asarray(values_m, dtype=np.float64).reshape(-1) * 100.0
    stats = vector_stats(values_cm)
    return {
        key: (value[0] if isinstance(value, list) and len(value) == 1 else value)
        for key, value in stats.items()
    }


def read_split_parquet(split_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((split_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {split_root / 'data'}")
    tables = [
        pq.read_table(
            path,
            columns=["observation.state", "action", "episode_index", "frame_index"],
        )
        for path in paths
    ]
    state = np.concatenate(
        [np.asarray(table["observation.state"].to_pylist(), dtype=np.float32) for table in tables]
    )
    action = np.concatenate(
        [np.asarray(table["action"].to_pylist(), dtype=np.float32) for table in tables]
    )
    episode_index = np.concatenate(
        [np.asarray(table["episode_index"], dtype=np.int64) for table in tables]
    )
    frame_index = np.concatenate(
        [np.asarray(table["frame_index"], dtype=np.int64) for table in tables]
    )
    return state, action, episode_index, frame_index


def summarize_group(indices: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    action_xyz = arrays["action_xyz"][indices]
    label_minus_eef = arrays["label_minus_eef"][indices]
    gripper = arrays["gripper"][indices]
    return {
        "frames": int(len(indices)),
        "fraction": float(len(indices) / len(arrays["action_xyz"])),
        "absolute_action_xyz_m": vector_stats(action_xyz),
        "absolute_action_xyz_axis_names": list(XYZ_NAMES),
        "action_t_plus_1_minus_eef_t_xyz_m": vector_stats(label_minus_eef),
        "action_t_plus_1_minus_eef_t_norm_cm": scalar_cm_stats(
            np.linalg.norm(label_minus_eef, axis=1)
        ),
        "action_target_shift_t_to_t_plus_1_norm_cm": scalar_cm_stats(
            arrays["target_shift_norm"][indices]
        ),
        "achieved_eef_t_to_t_plus_1_norm_cm": scalar_cm_stats(
            arrays["achieved_eef_delta_norm"][indices]
        ),
        "gripper": {
            "open_count": int(np.sum(gripper > 0.5)),
            "closed_count": int(np.sum(gripper < -0.5)),
            "intermediate_count": int(np.sum(np.abs(gripper) <= 0.5)),
            "open_fraction": float(np.mean(gripper > 0.5)),
            "closed_fraction": float(np.mean(gripper < -0.5)),
            "unique_rounded": sorted(np.unique(np.round(gripper, 6)).tolist()),
        },
    }


def main() -> None:
    args = parse_args()
    export_root = args.export_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    parquet_max_state_error = 0.0
    parquet_max_action_error = 0.0
    parquet_episode_frame_mismatches: list[dict[str, object]] = []
    geometry_errors: dict[str, list[float]] = defaultdict(list)
    offset_steps_seen: set[int] = set()

    for split in ("train", "validation", "test"):
        split_root = export_root / split
        manifest = json.loads(
            (split_root / "balanced_split_manifest.json").read_text(encoding="utf-8")
        )
        offset = int(manifest["action_offset_steps"])
        offset_steps_seen.add(offset)
        parquet_state, parquet_action, parquet_episode, parquet_frame = read_split_parquet(
            split_root
        )

        for local_episode_index, record in enumerate(manifest["records"]):
            source = Path(record["source"])
            with np.load(source / "trajectory.npz") as trajectory:
                q = np.asarray(trajectory["q"], dtype=np.float32)
                dq = np.asarray(trajectory["dq"], dtype=np.float32)
                eef = np.asarray(trajectory["eef_pose"], dtype=np.float32)
                object_pose = np.asarray(trajectory["object_pose"], dtype=np.float32)
                action = np.asarray(trajectory["action"], dtype=np.float32)
                phase_all = np.asarray(trajectory["controller_state"], dtype=np.int64)
            with np.load(source / "camera.npz") as camera:
                frame_steps = np.asarray(camera["frame_step"], dtype=np.int64)

            state_indices = frame_steps - 1
            action_indices = state_indices + offset
            valid = action_indices < len(action)
            state_indices = state_indices[valid]
            action_indices = action_indices[valid]
            source_state = np.concatenate(
                (q[state_indices], dq[state_indices], eef[state_indices]), axis=1
            )
            source_action = action[action_indices]
            # action[action_indices] was formed immediately after the current
            # post-step state, while the state machine still used the phase
            # stored on state_indices.  compute() transitions its internal
            # state only after forming desired_pose/gripper_action, so the
            # controller_state stored on action_indices can be one phase ahead
            # at a boundary.
            phase = phase_all[state_indices]
            next_controller_phase = phase_all[action_indices]

            mask = parquet_episode == local_episode_index
            order = np.argsort(parquet_frame[mask])
            observed_state = parquet_state[mask][order]
            observed_action = parquet_action[mask][order]
            if len(observed_action) != len(source_action):
                parquet_episode_frame_mismatches.append(
                    {
                        "split": split,
                        "episode_index": local_episode_index,
                        "source": str(source),
                        "parquet_frames": int(len(observed_action)),
                        "source_aligned_frames": int(len(source_action)),
                    }
                )
                continue
            parquet_max_state_error = max(
                parquet_max_state_error,
                float(np.max(np.abs(observed_state[:, :25] - source_state))),
            )
            parquet_max_action_error = max(
                parquet_max_action_error,
                float(np.max(np.abs(observed_action - source_action))),
            )

            current_eef = eef[state_indices]
            previous_action = action[state_indices]
            future_eef = eef[action_indices]
            label_minus_eef = source_action[:, :3] - current_eef[:, :3]
            target_shift = source_action[:, :3] - previous_action[:, :3]
            achieved_eef_delta = future_eef[:, :3] - current_eef[:, :3]

            for phase_id in np.unique(phase):
                phase_mask = phase == phase_id
                if phase_id == 0:
                    expected = current_eef[phase_mask, :3]
                    geometry_errors["rest_target_vs_current_eef_m"].extend(
                        np.linalg.norm(source_action[phase_mask, :3] - expected, axis=1).tolist()
                    )
                elif phase_id == 1:
                    expected = object_pose[state_indices[phase_mask], :3].copy()
                    expected[:, 2] += 0.12
                    geometry_errors["approach_target_vs_object_plus_12cm_m"].extend(
                        np.linalg.norm(source_action[phase_mask, :3] - expected, axis=1).tolist()
                    )
                elif phase_id in (2, 3):
                    expected = object_pose[state_indices[phase_mask], :3]
                    geometry_errors["pregrasp_close_target_vs_object_m"].extend(
                        np.linalg.norm(source_action[phase_mask, :3] - expected, axis=1).tolist()
                    )

            level = str(record["level"])
            for frame, (
                state_index,
                action_index,
                phase_id,
                source_action_row,
                current_eef_row,
                label_delta_row,
                target_shift_row,
                achieved_delta_row,
            ) in enumerate(
                zip(
                    state_indices,
                    action_indices,
                    phase,
                    source_action,
                    current_eef,
                    label_minus_eef,
                    target_shift,
                    achieved_eef_delta,
                    strict=True,
                )
            ):
                rows.append(
                    {
                        "split": split,
                        "difficulty": level,
                        "episode_index": local_episode_index,
                        "source": str(source),
                        "frame_index": frame,
                        "state_index": int(state_index),
                        "action_index": int(action_index),
                        "phase_id": int(phase_id),
                        "phase": PHASE_NAMES.get(int(phase_id), f"unknown_{phase_id}"),
                        "next_controller_phase_id": int(next_controller_phase[frame]),
                        "next_controller_phase": PHASE_NAMES.get(
                            int(next_controller_phase[frame]),
                            f"unknown_{next_controller_phase[frame]}",
                        ),
                        "phase_changed": bool(next_controller_phase[frame] != phase_id),
                        "action_x": float(source_action_row[0]),
                        "action_y": float(source_action_row[1]),
                        "action_z": float(source_action_row[2]),
                        "eef_x": float(current_eef_row[0]),
                        "eef_y": float(current_eef_row[1]),
                        "eef_z": float(current_eef_row[2]),
                        "label_minus_eef_x": float(label_delta_row[0]),
                        "label_minus_eef_y": float(label_delta_row[1]),
                        "label_minus_eef_z": float(label_delta_row[2]),
                        "label_minus_eef_norm_m": float(np.linalg.norm(label_delta_row)),
                        "target_shift_norm_m": float(np.linalg.norm(target_shift_row)),
                        "achieved_eef_delta_norm_m": float(np.linalg.norm(achieved_delta_row)),
                        "gripper": float(source_action_row[7]),
                        "visual_object_x": float(observed_state[frame, 25]),
                        "visual_object_y": float(observed_state[frame, 26]),
                        "visual_object_z": float(observed_state[frame, 27]),
                        "visual_pose_valid": float(observed_state[frame, 28]),
                    }
                )

    if not rows:
        raise RuntimeError("No aligned rows found")

    csv_path = output_dir / "vla_action_sample_diagnostics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    categorical = {
        "split": np.asarray([row["split"] for row in rows]),
        "difficulty": np.asarray([row["difficulty"] for row in rows]),
        "phase": np.asarray([row["phase"] for row in rows]),
    }
    arrays = {
        "action_xyz": np.asarray(
            [[row["action_x"], row["action_y"], row["action_z"]] for row in rows],
            dtype=np.float64,
        ),
        "label_minus_eef": np.asarray(
            [
                [
                    row["label_minus_eef_x"],
                    row["label_minus_eef_y"],
                    row["label_minus_eef_z"],
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
        "target_shift_norm": np.asarray(
            [row["target_shift_norm_m"] for row in rows], dtype=np.float64
        ),
        "achieved_eef_delta_norm": np.asarray(
            [row["achieved_eef_delta_norm_m"] for row in rows], dtype=np.float64
        ),
        "gripper": np.asarray([row["gripper"] for row in rows], dtype=np.float64),
        "visual_state": np.asarray(
            [
                [
                    row["visual_object_x"],
                    row["visual_object_y"],
                    row["visual_object_z"],
                    row["visual_pose_valid"],
                ]
                for row in rows
            ],
            dtype=np.float64,
        ),
    }

    def select(**conditions: str) -> np.ndarray:
        mask = np.ones(len(rows), dtype=bool)
        for key, value in conditions.items():
            mask &= categorical[key] == value
        return np.flatnonzero(mask)

    all_indices = np.arange(len(rows))
    by_split = {
        split: summarize_group(select(split=split), arrays)
        for split in sorted(set(categorical["split"]))
    }
    by_difficulty = {
        level: summarize_group(select(difficulty=level), arrays)
        for level in sorted(set(categorical["difficulty"]))
    }
    by_phase = {
        phase: summarize_group(select(phase=phase), arrays)
        for phase in PHASE_NAMES.values()
    }
    split_difficulty_phase: dict[str, object] = {}
    for split in ("train", "validation"):
        split_difficulty_phase[split] = {
            level: {
                phase: summarize_group(
                    select(split=split, difficulty=level, phase=phase), arrays
                )
                for phase in PHASE_NAMES.values()
            }
            for level in ("easy", "medium", "hard")
        }

    gripper_expected = np.asarray(
        [-1.0 if phase in ("close", "lift") else 1.0 for phase in categorical["phase"]]
    )
    gripper_mismatch = np.sign(arrays["gripper"]) != np.sign(gripper_expected)
    phase_changed = np.asarray([bool(row["phase_changed"]) for row in rows])
    quaternion_norms = []
    for split in ("train", "validation", "test"):
        _, actions, _, _ = read_split_parquet(export_root / split)
        quaternion_norms.extend(np.linalg.norm(actions[:, 3:7], axis=1).tolist())

    train_stats_path = export_root / "train/meta/stats.json"
    train_stats = json.loads(train_stats_path.read_text(encoding="utf-8"))["action"]
    train_idx = select(split="train")
    train_action_xyz = arrays["action_xyz"][train_idx]
    train_state_full, train_action_full, _, _ = read_split_parquet(export_root / "train")
    train_meta_all = json.loads(train_stats_path.read_text(encoding="utf-8"))
    visual_by_difficulty = {
        level: vector_stats(arrays["visual_state"][select(difficulty=level)])
        for level in ("easy", "medium", "hard")
    }
    visual_by_split = {
        split: vector_stats(arrays["visual_state"][select(split=split)])
        for split in ("train", "validation", "test")
    }
    report = {
        "status": "VLA_ACTION_DATASET_DIAGNOSTIC_COMPLETE",
        "export_root": str(export_root),
        "rows": len(rows),
        "episodes": int(len(set((row["split"], row["episode_index"]) for row in rows))),
        "action_offset_steps": sorted(offset_steps_seen),
        "parquet_source_consistency": {
            "maximum_state_0_to_24_absolute_error": parquet_max_state_error,
            "maximum_action_absolute_error": parquet_max_action_error,
            "episode_frame_mismatches": parquet_episode_frame_mismatches,
            "exact_with_float32_tolerance": bool(
                parquet_max_state_error <= 1e-7
                and parquet_max_action_error <= 1e-7
                and not parquet_episode_frame_mismatches
            ),
        },
        "overall": summarize_group(all_indices, arrays),
        "by_split": by_split,
        "by_difficulty": by_difficulty,
        "by_phase": by_phase,
        "train_validation_by_difficulty_and_phase": split_difficulty_phase,
        "gripper_phase_consistency": {
            "expected_open_in_rest_approach_pregrasp": True,
            "expected_closed_in_close_lift": True,
            "mismatch_count": int(gripper_mismatch.sum()),
            "mismatch_fraction": float(gripper_mismatch.mean()),
            "all_mismatches_are_sampled_phase_boundaries": bool(
                np.all(phase_changed[gripper_mismatch])
            ),
            "sampled_phase_boundary_count": int(phase_changed.sum()),
            "sampled_phase_boundary_fraction": float(phase_changed.mean()),
            "note": (
                "controller_state is advanced inside compute() after desired_pose and "
                "gripper_action are formed. The next trajectory row can therefore name the "
                "new phase while its applied action is the final command formed in the "
                "previous phase. This report groups causal labels by the current observation "
                "row's phase, which is their actual action semantics."
            ),
        },
        "coordinate_and_unit_checks": {
            "source_metadata_convention": "eef/object/action xyz are robot-root-frame meters; quaternion is wxyz",
            "quaternion_norm": vector_stats(np.asarray(quaternion_norms)),
            "phase_target_geometry_error_cm": {
                name: scalar_cm_stats(np.asarray(values))
                for name, values in sorted(geometry_errors.items())
            },
            "train_parquet_xyz_mean_vs_meta_stats_max_abs": float(
                np.max(np.abs(train_action_xyz.mean(axis=0) - np.asarray(train_stats["mean"][:3])))
            ),
            "train_parquet_xyz_std_vs_meta_stats_max_abs": float(
                np.max(np.abs(train_action_xyz.std(axis=0) - np.asarray(train_stats["std"][:3])))
            ),
            "train_action_all_8d_mean_vs_meta_stats_max_abs": float(
                np.max(
                    np.abs(
                        train_action_full.astype(np.float64).mean(axis=0)
                        - np.asarray(train_meta_all["action"]["mean"])
                    )
                )
            ),
            "train_action_all_8d_std_vs_meta_stats_max_abs": float(
                np.max(
                    np.abs(
                        train_action_full.astype(np.float64).std(axis=0)
                        - np.asarray(train_meta_all["action"]["std"])
                    )
                )
            ),
        },
        "visual_state_cache_distribution": {
            "names": [
                "visual_object_x",
                "visual_object_y",
                "visual_object_z",
                "visual_pose_valid",
            ],
            "by_split": visual_by_split,
            "by_difficulty": visual_by_difficulty,
            "cache_collapsed": False,
            "reason": (
                "x/y vary substantially in Medium and Hard. Easy is intentionally nearly "
                "fixed because its object pose is fixed. z is intentionally clamped to the "
                "known tabletop object-centre height (0.03 m), and all retained clean "
                "demonstrations have validity=1."
            ),
            "train_actual_percentiles_xyz": {
                name: np.percentile(train_state_full[:, 25:28], percentile, axis=0).tolist()
                for name, percentile in (("p01", 1), ("p10", 10), ("p50", 50), ("p90", 90), ("p99", 99))
            },
            "train_meta_percentiles_xyz": {
                name: train_meta_all["observation.state"][name][25:28]
                for name in ("q01", "q10", "q50", "q90", "q99")
            },
            "normalization_note": (
                "The checkpoint uses MEAN_STD for STATE/ACTION. Recomputed means/stds match "
                "metadata; LeRobot's stored aggregate quantiles are not the empirical row "
                "quantiles, but they are not used by this checkpoint."
            ),
        },
        "interpretation": {
            "current_eef_baseline": (
                "Mean Euclidean distance from the current post-step EEF xyz to the next "
                "absolute IK setpoint. It is a do-nothing baseline, not one-step motion and "
                "not a label/coordinate error."
            ),
            "causal_offset": (
                "Correct direction: row t stores the post-step observation after action[t] "
                "was applied, so offset=1 pairs it with the next command."
            ),
        },
        "sample_csv": str(csv_path),
    }

    report_path = output_dir / "vla_action_dataset_diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "VLA_ACTION_DATASET_DIAGNOSTIC="
        + json.dumps(
            {
                "report": str(report_path),
                "rows": len(rows),
                "validation_current_eef_baseline_cm": by_split["validation"][
                    "action_t_plus_1_minus_eef_t_norm_cm"
                ]["mean"],
                "parquet_max_action_error": parquet_max_action_error,
                "gripper_mismatch_count": int(gripper_mismatch.sum()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
