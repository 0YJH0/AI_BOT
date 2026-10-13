"""Verify the fixed-weight 2x2 phase/orientation experiment and summarize traces."""
import argparse
import json
from pathlib import Path
from statistics import mean
from compare_smolvla_initializations import load_benchmark, read, sha256

SCENE_FIELDS = ("sample_key", "level_seed", "sample_index", "object_x_m", "object_y_m",
                "object_yaw_deg", "object_mass_kg", "static_friction", "dynamic_friction",
                "restitution", "camera_position_m", "camera_target_m", "lighting_intensity")


def summarize_trace(data):
    trace = data["trace"]
    phase1 = [r for r in trace if r["phase_input"] == 1]
    return {
        "decision_steps": len(trace), "phase1_steps": len(phase1),
        "minimum_phase1_pregrasp_distance_cm": min(
            (r["pregrasp_distance_m"] * 100 for r in phase1), default=None),
        "final_pregrasp_distance_cm": trace[-1]["pregrasp_distance_m"] * 100 if trace else None,
        "phase1_steps_inside_15mm": sum(r["pregrasp_distance_m"] <= .015 for r in phase1),
        "phase1_steps_inside_25mm": sum(r["pregrasp_distance_m"] <= .025 for r in phase1),
        "phase_transitions": data["phase_transitions"],
        "gripper_blocked_control_steps": data["gripper_blocked_control_steps"],
        "gripper_blocked_fraction": data["gripper_blocked_control_steps"] / len(trace) if trace else None,
        "position_clipped_control_steps": sum(bool(r.get("position_was_clipped")) for r in trace),
        "orientation_override_actions": data["orientation_override_actions"],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    context = read(args.root / "experiment_context.json")
    if sha256(args.checkpoint / "adapter_model.safetensors") != context["checkpoint_sha256"].lower():
        raise ValueError("Model hash changed")
    baseline_config = baseline_scenes = None
    arms = {}
    for variant in context["variants"]:
        name = variant["name"]
        report, rows, run_context = load_benchmark(args.root / name, args.checkpoint)
        config = report["config"]
        if (config["phase_mechanism"] != variant["phase"]
                or config["orientation_mode"] != variant["orientation"]
                or not config["phase_diagnostics_recorded"]):
            raise ValueError("Wrong intervention or missing diagnostics")
        expected_count = context["episodes_per_level"] * 3
        if len(rows) != expected_count or run_context["inference_seed"] != context["inference_seed"]:
            raise ValueError("Wrong sample count or inference seed")
        common = {k: v for k, v in config.items() if k not in (
            "phase_mechanism", "orientation_mode", "feedback_phase_protocol", "vla_server", "hardware"
        )}
        scenes = {r["sample_key"]: {k: r[k] for k in SCENE_FIELDS} for r in rows}
        if baseline_config is None:
            baseline_config, baseline_scenes = common, scenes
        elif common != baseline_config or scenes != baseline_scenes:
            raise ValueError("Scene/controller protocol differs beyond declared interventions")
        report_path = next((args.root / name).glob("*/vla_benchmark_report.json"))
        traces = []
        for row in rows:
            path = report_path.parent / "phase_diagnostics" / (
                row["difficulty"] + "_" + str(int(row["sample_index"])).zfill(6) + ".json"
            )
            trace = read(path)
            if trace["sample_key"] != row["sample_key"] or trace["object_truth_in_feedback"]:
                raise ValueError("Wrong scene trace or privileged feedback")
            # Verify the intervention's no-xyz/no-gripper-overwrite invariant
            # from saved raw prediction and decoded held target, before guard.
            if variant["orientation"] == "fixed_top_down":
                for item in trace["trace"]:
                    raw, held = item.get("raw_policy_action"), item.get("held_absolute_target")
                    if raw is not None and held is not None and raw[7] != held[7]:
                        raise ValueError("Orientation override modified gripper")
                    if raw is not None and held is not None:
                        anchor = item["decoded_eef_anchor_xyz_b"]
                        if any(abs(held[i] - (raw[i] + anchor[i])) > 1e-6 for i in range(3)):
                            raise ValueError("Orientation override modified decoded xyz")
                        if item["action_origin_phase"] != 0 and held[3:7] != [0., 1., 0., 0.]:
                            raise ValueError("Fixed orientation was not applied")
            traces.append({
                "sample_key": row["sample_key"], "difficulty": row["difficulty"],
                "success": row["success"].lower() == "true", "failure_reason": row["failure_reason"],
                "maximum_lift_cm": float(row["maximum_lift_cm"]),
                "final_controller_phase": int(row["final_controller_state"]),
                "trace_file": str(path), **summarize_trace(trace),
            })
        minima = [t["minimum_phase1_pregrasp_distance_cm"] for t in traces
                  if t["minimum_phase1_pregrasp_distance_cm"] is not None]
        arms[name] = {
            "overall": report["overall"], "levels": report["levels"], "episodes": traces,
            "total_gripper_blocked_steps": sum(t["gripper_blocked_control_steps"] for t in traces),
            "total_phase1_steps": sum(t["phase1_steps"] for t in traces),
            "mean_phase1_min_pregrasp_distance_cm": mean(minima) if minima else None,
        }
    comparisons = {}
    for intervention, reference in (
        ("feedback_model", "legacy_model"), ("legacy_fixed", "legacy_model"),
        ("feedback_fixed", "legacy_fixed"), ("feedback_fixed", "feedback_model"),
    ):
        a, b = arms[intervention], arms[reference]
        paired_a = {r["sample_key"]: r for r in a["episodes"]}
        pairs = [(paired_a[r["sample_key"]], r) for r in b["episodes"]]
        comparisons[intervention + "_vs_" + reference] = {
            "success_gain": a["overall"]["num_success"] - b["overall"]["num_success"],
            "paired_success_gains": sum(x["success"] and not y["success"] for x, y in pairs),
            "paired_success_losses": sum(y["success"] and not x["success"] for x, y in pairs),
            "phase1_step_change": a["total_phase1_steps"] - b["total_phase1_steps"],
            "gripper_blocked_step_change": a["total_gripper_blocked_steps"] - b["total_gripper_blocked_steps"],
        }
    result = {
        "status": "PHASE_ORIENTATION_ABLATION_COMPLETE", "checkpoint_sha256": context["checkpoint_sha256"],
        "scene_protocol_verified": True, "arms": arms, "paired_comparisons": comparisons,
        "cautions": [
            "Small paired diagnostic; do not claim statistically stable success rates or a sole root cause.",
            "Feedback arm jointly changes thresholds, hysteresis, dwell and grip proxy; not a single-factor FSM subcomponent test.",
            "Orientation override preserves each prediction xyz/gripper, not future rollouts or future predictions.",
            "Same initial RNG seed is not per-request common noise; rendered RGB is not claimed pixel-identical.",
            "Grip proxy uses finger position/velocity and visual distance, not force sensing or object truth.",
        ],
        "next_stage": "Review paired outcomes and traces before changing dataset/interface; no automatic relabeling or fine-tuning.",
    }
    output = args.root / "phase_ablation_report.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PHASE_ABLATION_REPORT=" + json.dumps({
        "output": str(output), "successes": {k: v["overall"]["num_success"] for k, v in arms.items()}
    }), flush=True)


if __name__ == "__main__":
    main()
