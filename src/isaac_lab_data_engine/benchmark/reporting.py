"""Pure-Python aggregation and atomic report writing for Phase 7."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_FIELDS = [
    "difficulty",
    "seed",
    "num_episodes",
    "num_success",
    "num_failure",
    "success_rate",
    "failure_rate",
    "average_completion_time_s",
    "average_episode_duration_s",
    "average_episode_reward",
    "grasp_failure_rate",
    "object_drop_rate",
    "lift_failure_rate",
    "approach_failure_rate",
    "timeout_rate",
    "environment_steps_per_second",
    "wall_time_s",
    "collision_count_available",
]

EPISODE_FIELDS = [
    "difficulty",
    "level_seed",
    "episode_index",
    "batch_index",
    "env_index",
    "sample_index",
    "success",
    "failure_type",
    "steps",
    "completion_time_s",
    "total_reward",
    "initial_object_height_m",
    "maximum_object_height_m",
    "final_object_height_m",
    "minimum_ee_object_distance_m",
    "final_controller_state",
    "ever_object_held",
    "ever_lifted",
    "object_x_m",
    "object_y_m",
    "object_yaw_deg",
    "object_mass_kg",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "camera_applied",
    "camera_position_m",
    "camera_target_m",
    "lighting_intensity",
]

SAMPLING_FIELDS = [
    "sampling_strategy",
    "hard_case_selected",
    "hard_case_probability",
    "source_failure_episode_index",
    "source_failure_type",
]


def classify_failure(
    *,
    success: bool,
    terminated: bool,
    truncated: bool,
    ever_grasp_closed: bool,
    ever_object_held: bool,
    ever_lifted: bool,
) -> str:
    if success:
        return "none"
    if ever_lifted:
        return "object_drop"
    if ever_grasp_closed and not ever_object_held:
        return "grasp_failure"
    if ever_object_held:
        return "lift_failure"
    if terminated:
        return "terminated"
    if truncated:
        return "timeout"
    return "approach_failure" if not ever_grasp_closed else "timeout"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        for key, value in serialized.items():
            if isinstance(value, (list, dict)):
                serialized[key] = json.dumps(value, separators=(",", ":"))
        writer.writerow(serialized)
    _atomic_write(path, stream.getvalue())


def summarize_level(
    difficulty: str,
    seed: int,
    episode_rows: list[dict[str, Any]],
    batch_metrics: list[dict[str, float | int]],
) -> dict[str, Any]:
    if not episode_rows:
        raise ValueError("Cannot summarize an empty benchmark level")
    count = len(episode_rows)
    successful = [row for row in episode_rows if row["success"]]
    failure_counts = Counter(row["failure_type"] for row in episode_rows if not row["success"])
    wall_time = sum(float(item["wall_time_s"]) for item in batch_metrics)
    simulated_steps = sum(int(item["simulated_environment_steps"]) for item in batch_metrics)

    def rate(name: str) -> float:
        return failure_counts[name] / count

    return {
        "difficulty": difficulty,
        "seed": seed,
        "num_episodes": count,
        "num_success": len(successful),
        "num_failure": count - len(successful),
        "success_rate": len(successful) / count,
        "failure_rate": (count - len(successful)) / count,
        "average_completion_time_s": (
            mean(float(row["completion_time_s"]) for row in successful) if successful else None
        ),
        "average_episode_duration_s": mean(
            float(row["completion_time_s"]) for row in episode_rows
        ),
        "average_episode_reward": mean(float(row["total_reward"]) for row in episode_rows),
        "grasp_failure_rate": rate("grasp_failure"),
        "object_drop_rate": rate("object_drop"),
        "lift_failure_rate": rate("lift_failure"),
        "approach_failure_rate": rate("approach_failure"),
        "timeout_rate": rate("timeout"),
        "failure_counts": dict(sorted(failure_counts.items())),
        "environment_steps_per_second": simulated_steps / wall_time,
        "wall_time_s": wall_time,
        # A ContactSensor is intentionally deferred; report unavailability
        # instead of publishing a misleading zero collision count.
        "collision_count_available": False,
    }


def write_benchmark_reports(
    output_dir: str | Path,
    *,
    config_snapshot: dict[str, Any],
    summaries: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_results.csv"
    episode_path = output_dir / "benchmark_episodes.csv"
    report_path = output_dir / "benchmark_report.json"
    _write_csv(summary_path, SUMMARY_FIELDS, summaries)
    episode_fields = EPISODE_FIELDS + (
        SAMPLING_FIELDS if any("sampling_strategy" in row for row in episode_rows) else []
    )
    _write_csv(episode_path, episode_fields, episode_rows)
    report = {
        "status": "PHASE7_BENCHMARK_OK",
        "created_at": _utc_now(),
        "config": config_snapshot,
        "summaries": summaries,
        "episode_results_file": episode_path.name,
        "collision_metric": {
            "available": False,
            "reason": "No ContactSensor is configured in the Phase 7 low-dimensional task.",
        },
    }
    _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2))
    return {
        "summary_csv": str(summary_path),
        "episodes_csv": str(episode_path),
        "report_json": str(report_path),
    }
