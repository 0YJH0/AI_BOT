"""Simulator-free aggregation and report writing for the VLA benchmark."""

from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


VLA_EPISODE_FIELDS = [
    "difficulty",
    "level_seed",
    "episode_index",
    "sample_index",
    "sample_key",
    "success",
    "failure_reason",
    "failure_type",
    "steps",
    "completion_time_s",
    "wall_time_s",
    "total_reward",
    "initial_object_height_m",
    "maximum_object_height_m",
    "final_object_height_m",
    "maximum_lift_cm",
    "minimum_ee_object_distance_m",
    "final_controller_state",
    "ever_grasp_closed",
    "ever_object_held",
    "ever_lifted",
    "terminated",
    "truncated",
    "source_counts",
    "decision_reason_counts",
    "executed_action_decisions",
    "vla_action_count",
    "training_label_action_count",
    "training_label_source",
    "training_label_request_count",
    "training_label_request_phase_counts",
    "scripted_ik_action_count",
    "scripted_ik_fallback_fraction",
    "vla_request_count",
    "vla_request_phase_counts",
    "vla_error_count",
    "last_vla_error",
    "mean_vla_inference_ms",
    "p95_vla_inference_ms",
    "mean_rpc_roundtrip_ms",
    "p95_rpc_roundtrip_ms",
    "visual_object_x_m",
    "visual_object_y_m",
    "visual_object_z_m",
    "initial_visual_position_error_m",
    "visual_initialization_steps",
    "visual_initialization_attempts",
    "visual_initialization_valid_frames",
    "visual_initialization_max_consecutive_valid_frames",
    "visual_initialization_spread_m",
    "visual_initialization_stable",
    "visual_initialization_position_source",
    "visual_initialization_camera_quality_valid",
    "visual_initialization_rgb_std",
    "visual_initialization_rgb_range",
    "visual_initialization_depth_valid_fraction",
    "visual_initialization_rgb_sha256",
    "visual_initialization_camera_frame",
    "visual_initialization_camera_pose_error_m",
    "object_x_m",
    "object_y_m",
    "object_yaw_deg",
    "object_mass_kg",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "camera_position_m",
    "camera_target_m",
    "lighting_intensity",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _flatten_samples(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        values.extend(float(value) for value in row.get(key, []))
    return values


def _merge_counts(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    merged: Counter[str] = Counter()
    for row in rows:
        merged.update({str(name): int(count) for name, count in row.get(key, {}).items()})
    return merged


def validate_unique_vla_samples(rows: list[dict[str, Any]]) -> None:
    """Reject duplicate fixed-validation samples before publishing a report."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        key = str(
            row.get("sample_key")
            or f"{row['difficulty']}:{row['level_seed']}:{row['sample_index']}"
        )
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate VLA benchmark samples: {sorted(set(duplicates))}")


def summarize_vla_rows(
    rows: list[dict[str, Any]],
    *,
    difficulty: str,
    seed: int | None,
) -> dict[str, Any]:
    """Aggregate success, failure, latency and action-source metrics."""

    if not rows:
        raise ValueError("Cannot summarize an empty VLA benchmark")
    successful = [row for row in rows if bool(row["success"])]
    failures = Counter(
        str(row["failure_reason"]) for row in rows if not bool(row["success"])
    )
    source_counts = _merge_counts(rows, "source_counts")
    reason_counts = _merge_counts(rows, "decision_reason_counts")
    request_phase_counts = _merge_counts(rows, "vla_request_phase_counts")
    label_phase_counts = _merge_counts(rows, "training_label_request_phase_counts")
    inference_ms = _flatten_samples(rows, "_vla_inference_ms_samples")
    roundtrip_ms = _flatten_samples(rows, "_rpc_roundtrip_ms_samples")
    vla_actions = int(source_counts.get("vla", 0))
    scripted_actions = int(source_counts.get("scripted_ik", 0))
    training_label_actions = int(source_counts.get("training_label", 0))
    executed_actions = vla_actions + scripted_actions + training_label_actions
    wall_time_s = sum(float(row.get("wall_time_s", 0.0)) for row in rows)
    steps = sum(int(row.get("steps", 0)) for row in rows)
    sample_indices = [int(row["sample_index"]) for row in rows]
    sample_keys = [
        str(
            row.get("sample_key")
            or f"{row['difficulty']}:{row['level_seed']}:{row['sample_index']}"
        )
        for row in rows
    ]
    return {
        "difficulty": difficulty,
        "seed": seed,
        "num_episodes": len(rows),
        "num_success": len(successful),
        "num_failure": len(rows) - len(successful),
        "success_rate": len(successful) / len(rows),
        "failure_counts": dict(sorted(failures.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "decision_reason_counts": dict(sorted(reason_counts.items())),
        "vla_request_phase_counts": dict(sorted(request_phase_counts.items())),
        "training_label_request_phase_counts": dict(sorted(label_phase_counts.items())),
        "executed_action_decisions": executed_actions,
        "vla_action_count": vla_actions,
        "training_label_action_count": training_label_actions,
        "training_label_request_count": sum(int(row.get("training_label_request_count", 0)) for row in rows),
        "scripted_ik_action_count": scripted_actions,
        "scripted_ik_fallback_fraction": (
            scripted_actions / executed_actions if executed_actions else 0.0
        ),
        "vla_action_fraction": vla_actions / executed_actions if executed_actions else 0.0,
        "vla_request_count": sum(int(row.get("vla_request_count", 0)) for row in rows),
        "vla_error_count": sum(int(row.get("vla_error_count", 0)) for row in rows),
        "mean_vla_inference_ms": mean(inference_ms) if inference_ms else None,
        "p50_vla_inference_ms": _percentile(inference_ms, 0.50),
        "p95_vla_inference_ms": _percentile(inference_ms, 0.95),
        "mean_rpc_roundtrip_ms": mean(roundtrip_ms) if roundtrip_ms else None,
        "p50_rpc_roundtrip_ms": _percentile(roundtrip_ms, 0.50),
        "p95_rpc_roundtrip_ms": _percentile(roundtrip_ms, 0.95),
        "average_completion_time_s": (
            mean(float(row["completion_time_s"]) for row in successful)
            if successful
            else None
        ),
        "average_episode_duration_s": mean(
            float(row["completion_time_s"]) for row in rows
        ),
        "wall_time_s": wall_time_s,
        "environment_steps_per_second": steps / wall_time_s if wall_time_s > 0.0 else None,
        "sample_index_min": min(sample_indices),
        "sample_index_max": max(sample_indices),
        "unique_sample_count": len(set(sample_keys)),
    }


def summarize_vla_benchmark(
    rows: list[dict[str, Any]],
    level_seeds: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ordered per-level summaries and one overall summary."""

    validate_unique_vla_samples(rows)
    level_summaries: list[dict[str, Any]] = []
    for difficulty, seed in level_seeds.items():
        level_rows = [row for row in rows if row["difficulty"] == difficulty]
        if level_rows:
            level_summaries.append(
                summarize_vla_rows(level_rows, difficulty=difficulty, seed=seed)
            )
    overall = summarize_vla_rows(rows, difficulty="overall", seed=None)
    overall["level_success_rates"] = {
        item["difficulty"]: item["success_rate"] for item in level_summaries
    }
    return level_summaries, overall


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=VLA_EPISODE_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        for key, value in serialized.items():
            if isinstance(value, (dict, list, tuple)):
                serialized[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        writer.writerow(serialized)
    _atomic_write(path, stream.getvalue())


def write_vla_benchmark_reports(
    output_dir: str | Path,
    *,
    config_snapshot: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    level_seeds: dict[str, int],
) -> dict[str, str]:
    """Atomically write episode CSV, per-level JSON and overall JSON."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    level_summaries, overall = summarize_vla_benchmark(episode_rows, level_seeds)
    episode_path = output_dir / "vla_benchmark_episodes.csv"
    levels_path = output_dir / "vla_benchmark_levels.json"
    report_path = output_dir / "vla_benchmark_report.json"
    _write_episode_csv(episode_path, episode_rows)
    _atomic_write(
        levels_path,
        json.dumps(
            {
                "status": "VLA_BENCHMARK_OK",
                "created_at": _utc_now(),
                "summaries": level_summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    _atomic_write(
        report_path,
        json.dumps(
            {
                "status": "VLA_BENCHMARK_OK",
                "created_at": _utc_now(),
                "config": config_snapshot,
                "overall": overall,
                "levels": level_summaries,
                "episode_results_file": episode_path.name,
                "level_summaries_file": levels_path.name,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return {
        "episodes_csv": str(episode_path),
        "levels_json": str(levels_path),
        "report_json": str(report_path),
    }
