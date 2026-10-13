"""Summarize paired GPU VLA control-step budgets from saved benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def log_records(path: Path, prefix: str) -> list[dict]:
    return [json.loads(line[len(prefix):]) for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(prefix)]


def summarize_run(run: Path) -> dict:
    report = json.loads((run / "vla_benchmark_report.json").read_text(encoding="utf-8-sig"))
    with (run / "vla_benchmark_episodes.csv").open(encoding="utf-8-sig", newline="") as stream:
        episodes = list(csv.DictReader(stream))
    if report["status"] != "VLA_BENCHMARK_OK" or len(episodes) != 1:
        raise ValueError("Expected one completed fixed-scene episode")
    ready = log_records(run / "server_stdout.log", "SMOLVLA_SERVER_READY=")[0]
    memory = log_records(run / "server_stdout.log", "SMOLVLA_CUDA_MEMORY=")
    samples = json.loads((run / "gpu_memory_samples.json").read_text(encoding="utf-8-sig"))
    if isinstance(samples, dict):
        samples = [samples]
    row = episodes[0]
    checkpoint = Path(ready["checkpoint"])
    return {
        "run_dir": str(run.resolve()),
        "server": ready,
        "adapter_sha256": hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest(),
        "config": report["config"],
        "overall": report["overall"],
        "episode": row,
        "gpu_memory": {
            "sample_count": len(samples),
            "minimum_free_mib": min(s["free_mib"] for s in samples) if samples else None,
            "maximum_used_mib": max(s["used_mib"] for s in samples) if samples else None,
            "vla_peak_allocated_mib": max(s["peak_allocated_mib"] for s in memory) if memory else None,
            "vla_peak_reserved_mib": max(s["peak_reserved_mib"] for s in memory) if memory else None,
            "measurement": "nvidia-smi at approximately 1-second intervals during Isaac process lifetime; transient peaks may be missed",
            "cuda_mem_get_info_note": "CUDA-reported free memory may differ from dedicated VRAM reported by nvidia-smi on Windows WDDM; use nvidia-smi for whole-card headroom",
        },
        "progress": log_records(run / "isaac_stdout.log", "VLA_BENCHMARK_PROGRESS="),
    }


def compare_runs(runs: list[dict]) -> dict:
    if len(runs) != 2 or sorted(r["config"]["max_steps"] for r in runs) != [300, 600]:
        raise ValueError("Expected a 300-step and a 600-step run")
    config_keys = ("config_sha256", "controller_mode", "scripted_ik_fallback_enabled",
                   "condition_vla_on_visual_pose", "condition_vla_on_controller_phase",
                   "vla_position_target_mode", "execution_horizon", "action_repeat",
                   "success_criteria", "vision_checkpoint_sha256", "levels",
                   "camera_ready_timeout_s", "visual_initialization")
    scene_keys = ("sample_key", "object_x_m", "object_y_m", "object_yaw_deg", "object_mass_kg",
                  "static_friction", "dynamic_friction", "restitution", "camera_position_m",
                  "camera_target_m", "lighting_intensity")
    first, second = runs
    checks = {"same_checkpoint": first["server"]["checkpoint"] == second["server"]["checkpoint"],
              "same_adapter_sha256": first["adapter_sha256"] == second["adapter_sha256"],
              "same_inference_seed": first["server"]["seed"] == second["server"]["seed"],
              "both_cuda": all(r["server"]["device"] == "cuda" for r in runs),
              "same_gpu_allocator_limit": first["server"]["gpu_memory_fraction"] == second["server"]["gpu_memory_fraction"],
              "same_control_config": all(first["config"][k] == second["config"][k] for k in config_keys),
              "same_sampled_physical_scene": all(first["episode"][k] == second["episode"][k] for k in scene_keys)}
    if not all(checks.values()):
        raise ValueError(f"Comparison invariants failed: {checks}")
    return {"scope": "single_easy_scene_gpu_300_vs_600_diagnostic_not_statistical_benchmark",
            "invariants": checks,
            "notes": ["Fresh VLA server and Isaac process for each budget; weights, precision settings and RNG seeds unchanged.",
                      "Identical scene parameters do not imply bitwise-identical RGB renders or physics trajectories.",
                      "CPU and CUDA random samples/numerics can differ even with equal seeds; latency comparisons do not isolate policy quality."],
            "runs": sorted(runs, key=lambda r: r["config"]["max_steps"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs([summarize_run(path) for path in args.run_dir])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "invariants": result["invariants"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
