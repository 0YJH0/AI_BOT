"""Tests for Phase 9 sampling-round comparison."""

from __future__ import annotations

import csv

from isaac_lab_data_engine.data_engine import compare_sampling_rounds


FIELDS = [
    "difficulty",
    "episode_index",
    "success",
    "failure_type",
    "object_x_m",
    "object_y_m",
    "object_yaw_deg",
    "object_mass_kg",
    "static_friction",
    "completion_time_s",
    "total_reward",
    "hard_case_selected",
]


def _write_episodes(path, adaptive=False):
    rows = []
    for index in range(4):
        failed = index >= (2 if adaptive else 3)
        rows.append(
            {
                "difficulty": "failure_aware" if adaptive else "hard",
                "episode_index": index + 1,
                "success": not failed,
                "failure_type": "lift_failure" if failed else "none",
                "object_x_m": 0.75 if adaptive or index == 3 else 0.45,
                "object_y_m": 0.0,
                "object_yaw_deg": 150.0 if adaptive or index == 3 else 0.0,
                "object_mass_kg": 0.05 if adaptive or index == 3 else 0.3,
                "static_friction": 0.1 if adaptive or index == 3 else 0.7,
                "completion_time_s": 2.5,
                "total_reward": 5.0,
                "hard_case_selected": adaptive,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_risk_tables(path):
    path.mkdir()
    mapping = {
        "friction": ("static_friction", 0.08, 0.2),
        "distance": ("object_planar_distance_m", 0.7, 0.8),
        "mass": ("object_mass_kg", 0.02, 0.1),
        "orientation": ("object_yaw_deg", 120.0, 180.0),
    }
    for name, (field, left, right) in mapping.items():
        with (path / f"failure_by_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["parameter", "bin_label", "bin_left", "bin_right", "failure_rate", "sample_count"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "parameter": field,
                    "bin_label": f"[{left}, {right}]",
                    "bin_left": left,
                    "bin_right": right,
                    "failure_rate": 0.5,
                    "sample_count": 4,
                }
            )


def test_round_comparison_reconciles_and_measures_enrichment(tmp_path):
    baseline = tmp_path / "baseline.csv"
    adaptive = tmp_path / "adaptive.csv"
    risks = tmp_path / "risks"
    output = tmp_path / "output"
    _write_episodes(baseline)
    _write_episodes(adaptive, adaptive=True)
    _write_risk_tables(risks)

    report = compare_sampling_rounds(
        baseline_csv=baseline,
        adaptive_csv=adaptive,
        failure_analysis_dir=risks,
        output_dir=output,
    )
    assert report["failure_yield_ratio"] == 2.0
    assert report["additional_failure_episodes"] == 1
    assert all(row["failure_aware_coverage_rate"] > row["uniform_coverage_rate"] for row in report["high_risk_coverage"])
    assert (output / "round_comparison.csv").is_file()
    assert (output / "high_risk_coverage.png").stat().st_size > 1000
