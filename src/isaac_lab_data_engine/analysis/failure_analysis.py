"""Deterministic Phase 8 failure analysis for Phase 7 episode reports."""

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


REQUIRED_FIELDS = {
    "difficulty",
    "episode_index",
    "success",
    "failure_type",
    "object_x_m",
    "object_y_m",
    "object_yaw_deg",
    "object_mass_kg",
    "static_friction",
}

PARAMETERS = {
    "friction": ("static_friction", "Static friction", 5),
    "distance": ("object_planar_distance_m", "Object planar distance (m)", 5),
    "mass": ("object_mass_kg", "Object mass (kg)", 5),
    "orientation": ("object_yaw_deg", "Object yaw (deg)", 6),
}

BIN_FIELDS = [
    "parameter",
    "bin_index",
    "bin_label",
    "bin_left",
    "bin_right",
    "sample_count",
    "success_count",
    "failure_count",
    "failure_rate",
    "failure_rate_ci95_low",
    "failure_rate_ci95_high",
    "mean_parameter_value",
    "approach_failure_count",
    "grasp_failure_count",
    "lift_failure_count",
    "object_drop_count",
    "timeout_count",
    "terminated_count",
]


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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_benchmark_episodes(
    input_csv: str | Path,
    *,
    difficulty: str | None = "hard",
) -> list[dict[str, Any]]:
    """Load, validate, and enrich Phase 7 episode rows."""
    input_csv = Path(input_csv).resolve()
    with input_csv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Benchmark CSV is missing fields: {sorted(missing)}")
        raw_rows = list(reader)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_rows:
        if difficulty and raw["difficulty"] != difficulty:
            continue
        row = dict(raw)
        row["episode_index"] = int(raw["episode_index"])
        row["success"] = _parse_bool(raw["success"])
        for field in (
            "object_x_m",
            "object_y_m",
            "object_yaw_deg",
            "object_mass_kg",
            "static_friction",
        ):
            row[field] = float(raw[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"Non-finite {field} in episode {row['episode_index']}")
        row["object_planar_distance_m"] = math.hypot(row["object_x_m"], row["object_y_m"])
        row["is_failure"] = not row["success"]

        key = (row["difficulty"], row["episode_index"])
        if key in seen:
            raise ValueError(f"Duplicate episode key: {key}")
        seen.add(key)
        if row["success"] != (row["failure_type"] == "none"):
            raise ValueError(f"Inconsistent success/failure_type in episode {key}")
        rows.append(row)

    if not rows:
        label = difficulty if difficulty else "all"
        raise ValueError(f"No benchmark rows selected for difficulty={label!r}")
    return rows


def wilson_interval(failures: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial failure rate."""
    if count < 1 or failures < 0 or failures > count:
        raise ValueError("Expected 0 <= failures <= count and count > 0")
    rate = failures / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
    margin /= denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _equal_width_edges(values: list[float], bins: int) -> list[float]:
    if bins < 1:
        raise ValueError("bins must be positive")
    low, high = min(values), max(values)
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return [low, high]
    width = (high - low) / bins
    return [low + width * index for index in range(bins)] + [high]


def _bin_index(value: float, edges: list[float]) -> int:
    if len(edges) == 2 and edges[0] == edges[1]:
        return 0
    if value == edges[-1]:
        return len(edges) - 2
    width = edges[-1] - edges[0]
    index = int((value - edges[0]) / width * (len(edges) - 1))
    return min(max(index, 0), len(edges) - 2)


def binned_failure_rates(
    rows: list[dict[str, Any]],
    parameter: str,
    *,
    bins: int = 5,
) -> list[dict[str, Any]]:
    """Aggregate failure rates into deterministic equal-width bins."""
    values = [float(row[parameter]) for row in rows]
    edges = _equal_width_edges(values, bins)
    groups: list[list[dict[str, Any]]] = [[] for _ in range(len(edges) - 1)]
    for row in rows:
        groups[_bin_index(float(row[parameter]), edges)].append(row)

    result: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not group:
            continue
        failure_types = Counter(row["failure_type"] for row in group if row["is_failure"])
        failures = sum(bool(row["is_failure"]) for row in group)
        ci_low, ci_high = wilson_interval(failures, len(group))
        left, right = edges[index], edges[index + 1]
        closing = "]" if index == len(edges) - 2 else ")"
        result.append(
            {
                "parameter": parameter,
                "bin_index": index,
                "bin_label": f"[{left:.4g}, {right:.4g}{closing}",
                "bin_left": left,
                "bin_right": right,
                "sample_count": len(group),
                "success_count": len(group) - failures,
                "failure_count": failures,
                "failure_rate": failures / len(group),
                "failure_rate_ci95_low": ci_low,
                "failure_rate_ci95_high": ci_high,
                "mean_parameter_value": mean(float(row[parameter]) for row in group),
                **{
                    f"{name}_count": failure_types[name]
                    for name in (
                        "approach_failure",
                        "grasp_failure",
                        "lift_failure",
                        "object_drop",
                        "timeout",
                        "terminated",
                    )
                },
            }
        )
    return result


def _pearson_failure_correlation(rows: list[dict[str, Any]], parameter: str) -> float | None:
    x = [float(row[parameter]) for row in rows]
    y = [float(row["is_failure"]) for row in rows]
    x_mean, y_mean = mean(x), mean(y)
    x_var = sum((value - x_mean) ** 2 for value in x)
    y_var = sum((value - y_mean) ** 2 for value in y)
    if x_var == 0.0 or y_var == 0.0:
        return None
    covariance = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    return covariance / math.sqrt(x_var * y_var)


def _parameter_signal(
    rows: list[dict[str, Any]], parameter: str, bin_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    riskiest = max(bin_rows, key=lambda row: (row["failure_rate"], row["sample_count"]))
    safest = min(bin_rows, key=lambda row: (row["failure_rate"], -row["sample_count"]))
    return {
        "parameter": parameter,
        "pearson_failure_correlation": _pearson_failure_correlation(rows, parameter),
        "riskiest_bin": riskiest["bin_label"],
        "riskiest_bin_sample_count": riskiest["sample_count"],
        "riskiest_bin_failure_count": riskiest["failure_count"],
        "riskiest_bin_failure_rate": riskiest["failure_rate"],
        "safest_bin": safest["bin_label"],
        "safest_bin_failure_rate": safest["failure_rate"],
        "observed_failure_rate_gap": riskiest["failure_rate"] - safest["failure_rate"],
    }


def _plot_failure_rate(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    rates = [100.0 * float(row["failure_rate"]) for row in rows]
    lower = [
        100.0 * (float(row["failure_rate"]) - float(row["failure_rate_ci95_low"]))
        for row in rows
    ]
    upper = [
        100.0 * (float(row["failure_rate_ci95_high"]) - float(row["failure_rate"]))
        for row in rows
    ]
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    bars = axis.bar(x, rates, color="#D95F59", width=0.72)
    axis.errorbar(x, rates, yerr=[lower, upper], fmt="none", ecolor="#333333", capsize=4)
    axis.set_title(title, loc="left", fontsize=13, weight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Failure rate (%)")
    axis.set_xticks(x, [row["bin_label"] for row in rows], rotation=24, ha="right")
    axis.set_ylim(0.0, max(20.0, max(r + u for r, u in zip(rates, upper)) * 1.18))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, row in zip(bars, rows):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{row['failure_count']}/{row['sample_count']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    os.replace(temporary, output_path)


def _plot_failure_distribution(output_path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["failure_type"] for row in rows]
    counts = [row["failure_count"] for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = axis.bar(labels, counts, color="#4C78A8", width=0.65)
    axis.set_title("Hard benchmark failure distribution", loc="left", fontsize=13, weight="bold")
    axis.set_ylabel("Failure episodes")
    axis.set_ylim(0, max(counts + [1]) * 1.2)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, count + 0.2, str(count), ha="center")
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    os.replace(temporary, output_path)


def analyze_failure_file(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    difficulty: str | None = "hard",
) -> dict[str, Any]:
    """Run Phase 8 analysis and atomically publish CSV, JSON, and PNG artifacts."""
    input_csv = Path(input_csv).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_benchmark_episodes(input_csv, difficulty=difficulty)
    analysis_tables: dict[str, list[dict[str, Any]]] = {}
    signals: list[dict[str, Any]] = []
    files: dict[str, str] = {}

    for output_name, (field, label, bins) in PARAMETERS.items():
        table = binned_failure_rates(rows, field, bins=bins)
        analysis_tables[output_name] = table
        csv_path = output_dir / f"failure_by_{output_name}.csv"
        png_path = output_dir / f"failure_by_{output_name}.png"
        _write_csv(csv_path, BIN_FIELDS, table)
        _plot_failure_rate(
            png_path,
            table,
            title=f"Failure rate by {output_name}",
            x_label=label,
        )
        files[f"failure_by_{output_name}_csv"] = str(csv_path)
        files[f"failure_by_{output_name}_png"] = str(png_path)
        signals.append(_parameter_signal(rows, field, table))

    failure_counter = Counter(row["failure_type"] for row in rows if row["is_failure"])
    failure_distribution = [
        {
            "failure_type": name,
            "failure_count": count,
            "rate_over_all_episodes": count / len(rows),
            "share_of_failures": count / sum(failure_counter.values()),
        }
        for name, count in sorted(failure_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    distribution_path = output_dir / "failure_distribution.csv"
    distribution_png_path = output_dir / "failure_distribution.png"
    _write_csv(
        distribution_path,
        ["failure_type", "failure_count", "rate_over_all_episodes", "share_of_failures"],
        failure_distribution,
    )
    _plot_failure_distribution(distribution_png_path, failure_distribution)
    files["failure_distribution_csv"] = str(distribution_path)
    files["failure_distribution_png"] = str(distribution_png_path)

    failure_cases = [row for row in rows if row["is_failure"]]
    failure_case_fields = [
        "difficulty",
        "episode_index",
        "failure_type",
        "object_x_m",
        "object_y_m",
        "object_planar_distance_m",
        "object_yaw_deg",
        "object_mass_kg",
        "static_friction",
        "dynamic_friction",
        "maximum_object_height_m",
        "minimum_ee_object_distance_m",
        "final_controller_state",
    ]
    failure_cases_path = output_dir / "failure_cases.csv"
    _write_csv(failure_cases_path, failure_case_fields, failure_cases)
    files["failure_cases_csv"] = str(failure_cases_path)

    report_path = output_dir / "failure_analysis_report.json"
    files["report_json"] = str(report_path)
    report = {
        "status": "PHASE8_FAILURE_ANALYSIS_OK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "input_sha256": _sha256(input_csv),
        "difficulty": difficulty or "all",
        "episode_count": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "failure_count": len(failure_cases),
        "failure_rate": len(failure_cases) / len(rows),
        "failure_counts": dict(sorted(failure_counter.items())),
        "distance_definition": "sqrt(object_x_m^2 + object_y_m^2) in robot-base XY",
        "binning": "equal-width over the selected rows' observed range; final bin is right-closed",
        "confidence_interval": "Wilson score interval, 95%",
        "parameter_signals": signals,
        "limitations": [
            "Binned associations are descriptive and do not establish causality.",
            "Only 16 failures are present in the selected Hard benchmark, so intervals are wide.",
            "Camera parameters do not affect this state-based baseline.",
        ],
        "files": files,
    }
    _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2))
    return report
