"""Audit checkpoint contracts and summarize ten robot-only Hard episodes."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_checkpoint(path):
    cfg = read(path / "config.json")
    if (cfg["input_features"]["observation.state"]["shape"] != [25]
            or cfg["output_features"]["action"]["shape"] != [8]
            or set(cfg["input_features"]) != {"observation.state", "observation.images.table"}):
        raise ValueError("Checkpoint is not RGB + language + robot-only state / executed 8-D action")
    return cfg


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    verify_checkpoint(args.checkpoint)
    if not args.benchmark_root:
        print("PROPRIO_CHECKPOINT_25_8_VERIFIED", flush=True)
        return
    reports = list(args.benchmark_root.rglob("vla_benchmark_report.json"))
    if len(reports) != 1:
        raise ValueError("Expected exactly one finished benchmark")
    root = reports[0].parent
    report = read(reports[0])
    cfg = report["config"]
    if (report["status"] != "VLA_BENCHMARK_OK" or report["overall"]["num_episodes"] != 10
            or cfg["state_dimension"] != 25 or not cfg["proprio_only"]
            or cfg["condition_vla_on_visual_pose"] or cfg["student_phase_in_model_input"]
            or cfg["external_target_used_for_control"] or cfg["scripted_ik_fallback_enabled"]):
        raise ValueError("Robot-only ten-episode benchmark contract failed")
    with (root / "vla_benchmark_episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 10 or len({r["sample_key"] for r in rows}) != 10:
        raise ValueError("Ten distinct scenes required")
    if any(r["difficulty"] != "hard" or int(r["vla_error_count"]) or
           float(r["scripted_ik_fallback_fraction"]) for r in rows):
        raise ValueError("Invalid episode or hidden fallback")
    episodes = []
    traces = {t["sample_key"]: t["trace"] for t in
              (read(f) for f in (root / "phase_diagnostics").glob("*.json"))}
    for row in rows:
        # Match by sample key rather than relying on a particular trace filename.
        trace = traces[row["sample_key"]]
        first = next((r for r in trace if r["filtered_action"][7] < 0), None)
        episodes.append(dict(sample_key=row["sample_key"], success=row["success"].lower() == "true",
            failure_reason=row["failure_reason"], maximum_lift_cm=float(row["maximum_lift_cm"]),
            first_close_true_distance_cm=None if first is None else 100*first["true_ee_object_distance_m"],
            first_close_true_xy_distance_cm=None if first is None else 100*first["true_grasp_xy_distance_m"],
            first_close_true_z_offset_cm=None if first is None else 100*first["true_grasp_z_offset_m"],
            mean_inference_ms=row["mean_vla_inference_ms"]))
    summary = dict(status="PROPRIO_HARD10_COMPLETE", checkpoint=str(args.checkpoint.resolve()),
        policy_inputs=["RGB", "language", "robot_proprioception"], target_or_phase_auxiliary_control=False,
        episodes=episodes, successes=sum(e["success"] for e in episodes), trials=10,
        note="Small diagnostic; fresh fine-tuning, not zero-shot original weights.")
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
