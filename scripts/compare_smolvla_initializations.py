"""Compare fresh and inherited LoRA on identical held-out frames/scenes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_training_parity(continued, fresh, continued_adapter, fresh_adapter):
    for key in ("dataset", "seed", "steps", "batch_size", "num_workers", "optimizer",
                "scheduler", "resume", "save_freq", "tolerance_s"):
        if continued[key] != fresh[key]:
            raise ValueError(f"Training mismatch: {key}")
    c_policy = {k: v for k, v in continued["policy"].items() if k != "pretrained_path"}
    f_policy = {k: v for k, v in fresh["policy"].items() if k != "pretrained_path"}
    if c_policy != f_policy:
        raise ValueError("Policy configuration mismatch beyond initialization")
    for key in ("r", "lora_alpha", "lora_dropout", "target_modules", "modules_to_save",
                "base_model_name_or_path", "bias", "use_dora", "use_rslora"):
        if continued_adapter.get(key) != fresh_adapter.get(key):
            raise ValueError(f"Adapter configuration mismatch: {key}")
    if continued["steps"] != 6000 or fresh["resume"]:
        raise ValueError("Expected 6000 offline updates with fresh optimizer")


def compare_scans(continued, fresh):
    for scan in (continued, fresh):
        if scan["status"] != "VLA_CHECKPOINT_SCAN_COMPLETE" or scan["dataset_split"] != "validation":
            raise ValueError("Completed independent validation scans required")
    for key in ("dataset_root", "repo_id", "dataset_length", "indices", "seed",
                "device", "batch_size_requested", "checkpoint_steps",
                "position_target_mode", "action_chunk_size", "threshold_position_mae_cm"):
        if continued["scan_config"][key] != fresh["scan_config"][key]:
            raise ValueError(f"Validation protocol mismatch: {key}")
    if continued["scan_config"]["position_target_mode"] != "delta":
        raise ValueError("Expected relative end-effector position labels")
    rows = []
    for c, f in zip(continued["results"], fresh["results"], strict=True):
        if c["step"] != f["step"] or c["samples"] != f["samples"]:
            raise ValueError("Checkpoint/sample mismatch")
        if c["finite_rate"] != 1 or f["finite_rate"] != 1:
            raise ValueError("Nonfinite predictions")
        if c["batch_size_effective"] != f["batch_size_effective"]:
            raise ValueError("Different effective validation batching")
        if abs(c["zero_delta_baseline_mae_cm"] - f["zero_delta_baseline_mae_cm"]) > 1e-6:
            raise ValueError("Validation label baseline differs")
        rows.append({
            "step": c["step"], "continued": c, "fresh": f,
            "fresh_minus_continued_position_cm": f["position_mae_cm"] - c["position_mae_cm"],
        })
    final = [r for r in rows if r["step"] == 6000]
    if len(final) != 1:
        raise ValueError("Final 6000 checkpoint required in each arm")
    return {
        "validation_samples": len(continued["indices"]), "checkpoint_comparison": rows,
        "primary_comparison": final[0],
        "both_final_checkpoints_pass_2cm_gate": all(
            final[0][arm]["position_mae_cm"] <= 2 for arm in ("continued", "fresh")
        ),
        "best_scanned_checkpoint": {arm: scan["best_checkpoint"]["step"] for arm, scan in (
            ("continued", continued), ("fresh", fresh)
        )},
    }


def load_benchmark(root, checkpoint):
    reports = list(Path(root).glob("*/vla_benchmark_report.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected exactly one completed benchmark: {root}")
    path = reports[0]
    report = read(path)
    context = read(path.parent / "test_context.json")
    if report["status"] != "VLA_BENCHMARK_OK" or context["checkpoint_step"] != 6000:
        raise ValueError("Incomplete or wrong-checkpoint benchmark")
    if context["checkpoint_sha256"].lower() != sha256(checkpoint / "adapter_model.safetensors"):
        raise ValueError("Benchmark weight hash mismatch")
    config = report["config"]
    if (config["controller_mode"] != "pure_vla_no_scripted_fallback"
            or config["scripted_ik_fallback_enabled"]
            or config["object_truth_usage"] != "success/failure metrics only"
            or config["minimum_actual_lift_m"] != .1):
        raise ValueError("Expected visual student, no fallback, actual 10 cm lift")
    overall = report["overall"]
    if (overall["scripted_ik_action_count"] or overall["training_label_action_count"]
            or overall["vla_error_count"]):
        raise ValueError("Fallback/teacher actions or inference failures invalidate comparison")
    with (path.parent / report["episode_results_file"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({r["sample_key"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate benchmark scenes")
    return report, rows, context


def compare_benchmarks(continued, fresh):
    c_report, c_rows, c_context = continued
    f_report, f_rows, f_context = fresh
    if c_context.get("inference_seed") != f_context.get("inference_seed"):
        raise ValueError("Inference seed mismatch")
    # Hardware description/server endpoint may differ; all scene/controller settings must not.
    c_config = {k: v for k, v in c_report["config"].items() if k not in ("hardware", "vla_server")}
    f_config = {k: v for k, v in f_report["config"].items() if k not in ("hardware", "vla_server")}
    if c_config != f_config:
        raise ValueError("Closed-loop protocol mismatch")
    scene_fields = ("sample_key", "level_seed", "sample_index", "object_x_m", "object_y_m",
                    "object_yaw_deg", "object_mass_kg", "static_friction", "dynamic_friction",
                    "restitution", "camera_position_m", "camera_target_m", "lighting_intensity")
    by_key = {r["sample_key"]: r for r in f_rows}
    if set(by_key) != {r["sample_key"] for r in c_rows}:
        raise ValueError("Closed-loop sample keys differ")
    paired = []
    metric_fields = ("success", "failure_reason", "maximum_lift_cm",
                     "minimum_ee_object_distance_m", "steps", "final_controller_state",
                     "mean_vla_inference_ms")
    for c in c_rows:
        f = by_key[c["sample_key"]]
        if any(c[key] != f[key] for key in scene_fields):
            raise ValueError(f"Scene parameters differ: {c['sample_key']}")
        paired.append({
            "sample_key": c["sample_key"],
            "continued": {k: c[k] for k in metric_fields},
            "fresh": {k: f[k] for k in metric_fields},
        })
    return {
        "scope": "small paired diagnostic; not a statistically stable success-rate claim",
        "episodes_per_arm": len(c_rows), "paired_episodes": paired,
        "continued": {"overall": c_report["overall"], "levels": c_report["levels"]},
        "fresh": {"overall": f_report["overall"], "levels": f_report["levels"]},
        "pixel_identical_images_claimed": False,
    }


def verify_gate_override(report, allowed, reason):
    if allowed and not reason.strip():
        raise ValueError("Diagnostic validation gate override requires a reason")
    if not report["both_final_checkpoints_pass_2cm_gate"] and not allowed:
        raise ValueError("Position-error gate failed")
    return {
        "requested": bool(allowed),
        "used": bool(allowed and not report["both_final_checkpoints_pass_2cm_gate"]),
        "reason": reason if allowed else None,
        "scope": "diagnostic experiment; does not change the actual offline gate result",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--continued-checkpoint", type=Path, required=True)
    parser.add_argument("--fresh-checkpoint", type=Path, required=True)
    parser.add_argument("--fresh-initialization", type=Path, required=True)
    parser.add_argument("--require-benchmarks", action="store_true")
    parser.add_argument("--allow-validation-gate-override", action="store_true")
    parser.add_argument("--validation-gate-override-reason", default="")
    args = parser.parse_args()
    root = args.experiment_root
    audit = read(args.fresh_initialization)
    if (audit["status"] != "FRESH_LORA_INITIALIZATION_VERIFIED" or audit["old_adapter_loaded"]
            or not audit["base_weights_frozen"] or audit["trainable_parameters"] != 742656
            or audit["zero_initialized_B_matrices"] != 37):
        raise ValueError("Fresh LoRA initialization not verified")
    checkpoints = (args.continued_checkpoint, args.fresh_checkpoint)
    verify_training_parity(
        *(read(p / "train_config.json") for p in checkpoints),
        *(read(p / "adapter_config.json") for p in checkpoints),
    )
    experiment = read(root / "experiment_context.json")
    for artifact in experiment["protected_artifacts"]:
        if sha256(artifact["path"]) != artifact["sha256"].lower():
            raise ValueError(f"Protected artifact changed: {artifact['path']}")
    scans = [read(root / (arm + "_validation240.json")) for arm in ("continued", "fresh")]
    report = compare_scans(*scans)
    report.update({
        "status": "OFFLINE_COMPARISON_COMPLETE", "fresh_initialization": audit,
        "training_config_parity_verified": True, "source_hashes_unchanged": True,
        "checkpoint_sha256": {arm: sha256(p / "adapter_model.safetensors") for arm, p in zip(
            ("continued", "fresh"), checkpoints, strict=True)},
        "test_split_used": False,
        "caution": "One training seed only. An advantage does not by itself prove harmful transfer; base weights were never polluted.",
    })
    if args.require_benchmarks:
        report["validation_gate_override"] = verify_gate_override(
            report, args.allow_validation_gate_override, args.validation_gate_override_reason
        )
        benchmarks = tuple(
            load_benchmark(root / (arm + "_benchmark"), p) for arm, p in zip(
                ("continued", "fresh"), checkpoints, strict=True
            )
        )
        if report["validation_gate_override"]["used"]:
            for benchmark in benchmarks:
                context = benchmark[2]
                if (not context.get("validation_gate_override_requested")
                        or context.get("validation_gate_override_reason") != args.validation_gate_override_reason):
                    raise ValueError("Benchmark diagnostic gate override not recorded")
        report["closed_loop"] = compare_benchmarks(*benchmarks)
        report["status"] = "PAIRED_COMPARISON_COMPLETE"
    # Only this experiment's derived report is updated after its second evaluation stage.
    output = root / "comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SMOLVLA_COMPARISON=" + json.dumps({
        "status": report["status"], "output": str(output),
        "continued_position_cm": report["primary_comparison"]["continued"]["position_mae_cm"],
        "fresh_position_cm": report["primary_comparison"]["fresh"]["position_mae_cm"],
        "gate_passed": report["both_final_checkpoints_pass_2cm_gate"],
    }), flush=True)


if __name__ == "__main__":
    main()
