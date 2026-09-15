"""Compare uniform and failure-aware data-generation rounds."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


RISK_TABLES = {
    "friction": "static_friction",
    "distance": "object_planar_distance_m",
    "mass": "object_mass_kg",
    "orientation": "object_yaw_deg",
}


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, stream.getvalue())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_episodes(path: Path, difficulty: str | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if difficulty:
        rows = [row for row in rows if row["difficulty"] == difficulty]
    if not rows:
        raise ValueError(f"No episode rows selected from {path}")
    for row in rows:
        row["success"] = row["success"].lower() == "true"
        for field in (
            "object_x_m",
            "object_y_m",
            "object_yaw_deg",
            "object_mass_kg",
            "static_friction",
            "completion_time_s",
            "total_reward",
        ):
            row[field] = float(row[field])
        row["object_planar_distance_m"] = math.hypot(row["object_x_m"], row["object_y_m"])
        if "hard_case_selected" in row:
            row["hard_case_selected"] = row["hard_case_selected"].lower() == "true"
    return rows


def _round_summary(name: str, strategy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if not row["success"]]
    failure_types = Counter(row["failure_type"] for row in failures)
    selected = [row for row in rows if row.get("hard_case_selected") is True]
    selected_failures = [row for row in selected if not row["success"]]
    return {
        "round": name,
        "strategy": strategy,
        "episode_count": len(rows),
        "success_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "failure_rate": len(failures) / len(rows),
        "failure_episodes_per_100": 100.0 * len(failures) / len(rows),
        "average_completion_time_s": mean(row["completion_time_s"] for row in rows),
        "average_episode_reward": mean(row["total_reward"] for row in rows),
        "hard_case_selected_count": len(selected),
        "hard_case_selected_rate": len(selected) / len(rows),
        "hard_case_selected_failure_count": len(selected_failures),
        "hard_case_selected_failure_rate": (
            len(selected_failures) / len(selected) if selected else None
        ),
        "approach_failure_count": failure_types["approach_failure"],
        "grasp_failure_count": failure_types["grasp_failure"],
        "lift_failure_count": failure_types["lift_failure"],
        "object_drop_count": failure_types["object_drop"],
        "timeout_count": failure_types["timeout"],
    }


def _risk_bin(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty risk table: {path}")
    return max(rows, key=lambda row: (float(row["failure_rate"]), int(row["sample_count"])))


def _in_interval(value: float, left: float, right: float) -> bool:
    return left <= value <= right


def _coverage_rows(
    baseline_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    failure_analysis_dir: Path,
) -> list[dict[str, Any]]:
    result = []
    for name, field in RISK_TABLES.items():
        risk = _risk_bin(failure_analysis_dir / f"failure_by_{name}.csv")
        left, right = float(risk["bin_left"]), float(risk["bin_right"])
        baseline_count = sum(
            _in_interval(float(row[field]), left, right) for row in baseline_rows
        )
        adaptive_count = sum(
            _in_interval(float(row[field]), left, right) for row in adaptive_rows
        )
        baseline_rate = baseline_count / len(baseline_rows)
        adaptive_rate = adaptive_count / len(adaptive_rows)
        result.append(
            {
                "parameter": field,
                "risk_bin_label": risk["bin_label"],
                "risk_bin_left": left,
                "risk_bin_right": right,
                "phase8_failure_rate_in_bin": float(risk["failure_rate"]),
                "uniform_count": baseline_count,
                "uniform_coverage_rate": baseline_rate,
                "failure_aware_count": adaptive_count,
                "failure_aware_coverage_rate": adaptive_rate,
                "coverage_change_percentage_points": 100.0 * (adaptive_rate - baseline_rate),
                "coverage_ratio": adaptive_rate / baseline_rate if baseline_rate else None,
            }
        )
    return result


def _plot_coverage(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        {
            "static_friction": "Low friction",
            "object_planar_distance_m": "Far distance",
            "object_mass_kg": "Low mass",
            "object_yaw_deg": "High yaw",
        }[row["parameter"]]
        for row in rows
    ]
    x = list(range(len(rows)))
    uniform = [100.0 * row["uniform_coverage_rate"] for row in rows]
    adaptive = [100.0 * row["failure_aware_coverage_rate"] for row in rows]
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    first = axis.bar([value - width / 2 for value in x], uniform, width, label="Uniform Hard", color="#8FA8C2")
    second = axis.bar([value + width / 2 for value in x], adaptive, width, label="Failure-aware", color="#D95F59")
    axis.set_title("High-risk region coverage by sampling round", loc="left", fontsize=13, weight="bold")
    axis.set_ylabel("Episodes in Phase 8 high-risk bin (%)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(adaptive + uniform) * 1.22)
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bars in (first, second):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=9,
            )
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    os.replace(temporary, path)


def _plot_round_yield(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = [100.0 * row["failure_rate"] for row in rows]
    labels = [row["strategy"] for row in rows]
    figure, axis = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    bars = axis.bar(labels, rates, color=["#8FA8C2", "#D95F59"], width=0.58)
    axis.set_title("Difficult episode yield", loc="left", fontsize=13, weight="bold")
    axis.set_ylabel("Failed episodes (%)")
    axis.set_ylim(0, max(rates) * 1.25)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{row['failure_count']}/{row['episode_count']}",
            ha="center",
        )
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    os.replace(temporary, path)


def compare_sampling_rounds(
    *,
    baseline_csv: str | Path,
    adaptive_csv: str | Path,
    failure_analysis_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    baseline_csv = Path(baseline_csv).resolve()
    adaptive_csv = Path(adaptive_csv).resolve()
    failure_analysis_dir = Path(failure_analysis_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_episodes(baseline_csv, difficulty="hard")
    adaptive_rows = _load_episodes(adaptive_csv)
    round_rows = [
        _round_summary("round_1", "Uniform Hard", baseline_rows),
        _round_summary("round_2", "Failure-aware", adaptive_rows),
    ]
    coverage_rows = _coverage_rows(baseline_rows, adaptive_rows, failure_analysis_dir)

    round_path = output_dir / "round_comparison.csv"
    coverage_path = output_dir / "high_risk_coverage.csv"
    coverage_png = output_dir / "high_risk_coverage.png"
    yield_png = output_dir / "difficult_episode_yield.png"
    _write_csv(round_path, list(round_rows[0]), round_rows)
    _write_csv(coverage_path, list(coverage_rows[0]), coverage_rows)
    _plot_coverage(coverage_png, coverage_rows)
    _plot_round_yield(yield_png, round_rows)

    baseline_failures = round_rows[0]["failure_count"]
    adaptive_failures = round_rows[1]["failure_count"]
    report_path = output_dir / "data_loop_report.json"
    report = {
        "status": "PHASE9_FAILURE_AWARE_SAMPLING_OK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_csv": str(baseline_csv),
        "baseline_sha256": _sha256(baseline_csv),
        "adaptive_csv": str(adaptive_csv),
        "adaptive_sha256": _sha256(adaptive_csv),
        "rounds": round_rows,
        "failure_yield_ratio": adaptive_failures / baseline_failures,
        "additional_failure_episodes": adaptive_failures - baseline_failures,
        "high_risk_coverage": coverage_rows,
        "interpretation": "Failure-aware sampling targets difficult data coverage; its lower success rate is not a policy regression comparison under an unchanged distribution.",
        "files": {
            "round_comparison_csv": str(round_path),
            "high_risk_coverage_csv": str(coverage_path),
            "high_risk_coverage_png": str(coverage_png),
            "difficult_episode_yield_png": str(yield_png),
            "report_json": str(report_path),
        },
    }
    _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2))
    return report
