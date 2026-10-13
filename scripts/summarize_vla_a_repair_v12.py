"""Supplement A's initialization failures without modifying original reports."""
import argparse
import csv
import json
from pathlib import Path
from summarize_vla_phase_ablation import SCENE_FIELDS, summarize_trace


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--repair", type=Path, required=True)
    a = p.parse_args()
    old_report = next((a.original / "legacy_model").glob("*/vla_benchmark_report.json"))
    new_report = next(a.repair.glob("*/vla_benchmark_report.json"))
    def rows(path):
        return list(csv.DictReader(path.with_name("vla_benchmark_episodes.csv").open(encoding="utf-8-sig")))
    old, new = rows(old_report), rows(new_report)
    lookup = {r["sample_key"]: r for r in old}
    if len(new) != 6: raise ValueError("Expected exactly six repairs")
    context = json.loads(new_report.with_name("test_context.json").read_text(encoding="utf-8-sig"))
    original_context = json.loads((a.original / "experiment_context.json").read_text(encoding="utf-8-sig"))
    if context["checkpoint_sha256"].lower() != original_context["checkpoint_sha256"].lower():
        raise ValueError("Different model weights")
    supplements = []
    for row in new:
        prior = lookup[row["sample_key"]]
        if prior["failure_reason"] != "invalid_visual_initialization":
            raise ValueError("Do not replace a completed control rollout")
        if any(prior[k] != row[k] for k in SCENE_FIELDS):
            raise ValueError("Physical scene parameters changed")
        if row["visual_initialization_stable"] != "True" or int(row["vla_request_count"]) < 1:
            raise ValueError("Repair has not actually entered VLA control")
        if float(row["scripted_ik_fallback_fraction"]) != 0 or int(row["vla_error_count"]) != 0:
            raise ValueError("Unexpected fallback or inference error")
        path = new_report.parent / "phase_diagnostics" / (row["difficulty"]+"_"+row["sample_index"].zfill(6)+".json")
        data = json.loads(path.read_text())
        supplements.append({"sample_key": row["sample_key"], "difficulty": row["difficulty"],
            "success": row["success"].lower() == "true", "failure_reason": row["failure_reason"],
            "maximum_lift_cm": float(row["maximum_lift_cm"]),
            "final_controller_phase": int(row["final_controller_state"]),
            "visual_initialization_steps": int(row["visual_initialization_steps"]),
            "consecutive_valid_frames": int(row["visual_initialization_max_consecutive_valid_frames"]),
            "trace_file": str(path), **summarize_trace(data)})
        lookup[row["sample_key"]] = row
    levels = {l: {"episodes": 3, "successes": sum(r["success"].lower() == "true" for r in lookup.values() if r["difficulty"] == l)}
              for l in ("easy", "medium", "hard")}
    output = a.repair / "repaired_a_summary.json"
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps({"status": "A_REPAIR_VERIFIED", "checkpoint_sha256": context["checkpoint_sha256"],
        "original_preserved": True, "original_ablation_episodes": 36, "additional_episodes": 6,
        "repaired_a_successes": sum(x["successes"] for x in levels.values()), "repaired_a_episodes": 9,
        "levels": levels, "supplements": supplements,
        "strict_four_arm_paired_repetition": False,
        "caution": "A combines original Easy and repaired Medium/Hard; camera readiness protocol changed. Same scene parameters, not pixel-identical RGB or per-request common diffusion noise."}, indent=2), encoding="utf-8")
    print("A_REPAIR="+json.dumps({"output": str(output), "levels": levels}), flush=True)


if __name__ == "__main__": main()
