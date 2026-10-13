"""Compare exact saved-action replay and live-state label-rule execution."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaac_lab_data_engine.vla.dataset import build_visual_fsm_bounded_actions


def summarize(root: Path) -> dict:
    audit = json.loads((root / "label_audit.json").read_text(encoding="utf-8"))
    rows_by_mode = {}
    configs = {}
    for mode in ("stored", "online"):
        run_dir = root / f"{mode}_cached"
        report = json.loads((run_dir / "vla_benchmark_report.json").read_text(encoding="utf-8"))
        configs[mode] = report["config"]
        verification = report["config"]["training_label_verification"]
        if verification["bundle_sha256"] != audit["bundle_sha256"]:
            raise ValueError("Replay bundle changed")
        if verification["minimum_actual_lift_m"] < 0.10:
            raise ValueError("The true 10 cm lift criterion was weakened")
        if report["config"]["scripted_ik_fallback_enabled"]:
            raise ValueError("Fallback must be disabled")
        with (run_dir / report["episode_results_file"]).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 3 or {r["difficulty"] for r in rows} != {"easy", "medium", "hard"}:
            raise ValueError("Expected one completed episode per level")
        summaries = {}
        for row in rows:
            level = row["difficulty"]
            trace = json.loads((run_dir / f"label_{level}_trace.json").read_text(encoding="utf-8"))
            if int(row["vla_request_count"]) or int(row["scripted_ik_action_count"]):
                raise ValueError("A model or fallback was executed")
            with np.load(root / "replay_bundle.npz", allow_pickle=False) as bundle:
                emitted = np.asarray([item["label_action"] for item in trace["trace"]], dtype=np.float32)
                if mode == "stored":
                    expected = bundle[f"{level}_actions"]
                else:
                    states = bundle[f"{level}_states"]
                    semantics = json.loads(str(bundle["semantics_json"].item()))
                    expected = build_visual_fsm_bounded_actions(
                        np.asarray([item["live_eef"] for item in trace["trace"]], dtype=np.float32),
                        np.repeat(states[:1, 25:28], len(emitted), axis=0),
                        np.asarray([item["live_phase"] for item in trace["trace"]]),
                        maximum_position_step_m=semantics["maximum_position_step_m"],
                        approach_height_m=semantics["approach_height_m"],
                        lift_height_m=semantics["lift_height_m"],
                        position_target_mode="delta",
                    )
            if emitted.shape != expected.shape or not np.array_equal(emitted, expected):
                raise ValueError(f"Executed label trace differs from {mode} targets: {level}")
            success = row["success"] == "True"
            lift = float(row["maximum_lift_cm"])
            if success and lift < 10.0:
                raise ValueError("Successful episode did not lift 10 cm")
            summaries[level] = {
                "success": success, "failure_reason": row["failure_reason"],
                "maximum_lift_cm": lift,
                "minimum_ee_object_distance_cm": 100 * float(row["minimum_ee_object_distance_m"]),
                "steps": int(row["steps"]), "final_phase": int(row["final_controller_state"]),
                "labels_executed": trace["labels_executed"],
                "maximum_executed_label_reconstruction_error": float(np.max(np.abs(emitted - expected))),
                "source": trace["source"], "sample_key": row["sample_key"],
                "visual_error_cm": 100 * float(row["initial_visual_position_error_m"]),
                "initial_eef": trace["trace"][0]["live_eef"],
                "final_label_eef_drift_cm": trace["trace"][-1].get("eef_position_drift_cm"),
                "row": row,
            }
        rows_by_mode[mode] = summaries
    for key in ("action_repeat", "execution_horizon", "vla_position_target_mode", "max_steps"):
        if configs["stored"][key] != configs["online"][key]:
            raise ValueError(f"Execution mismatch: {key}")
    scene_keys = (
        "sample_key", "object_x_m", "object_y_m", "object_yaw_deg", "object_mass_kg",
        "static_friction", "dynamic_friction", "restitution", "camera_position_m",
        "camera_target_m", "lighting_intensity", "visual_object_x_m",
        "visual_object_y_m", "visual_object_z_m",
    )
    for level in ("easy", "medium", "hard"):
        a, b = rows_by_mode["stored"][level], rows_by_mode["online"][level]
        if a["initial_eef"] != b["initial_eef"]:
            raise ValueError(f"Initial EEF mismatch: {level}")
        for key in scene_keys:
            if a["row"][key] != b["row"][key]:
                raise ValueError(f"Scene/input mismatch: {level}/{key}")
        a.pop("row")
        b.pop("row")
    return {
        "frames_audited": audit["frames_audited"],
        "maximum_label_reconstruction_error": audit["maximum_absolute_label_reconstruction_error"],
        "bundle_sha256": audit["bundle_sha256"],
        "comparison_checks": "same source scene, initial EEF, visual cache and execution settings",
        "minimum_true_lift_cm": 10.0,
        "model_requests": 0, "scripted_fallback_actions": 0,
        "label_rule_has_success_example": any(r["success"] for r in rows_by_mode["online"].values()),
        "label_rule_passes_all_three_examples": all(r["success"] for r in rows_by_mode["online"].values()),
        "stored_sequence_has_success_example": any(r["success"] for r in rows_by_mode["stored"].values()),
        "scope": (
            "Three source demonstrations only; online is a feedback label oracle, not VLA success. "
            "Stored is time-indexed action replay, not an observation-conditioned policy."
        ),
        "results": rows_by_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.root / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    result = summarize(args.root)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
