"""Simulator-free tests for Phase 8 failure analysis."""

import csv
import json

import pytest

from isaac_lab_data_engine.analysis import (
    analyze_failure_file,
    binned_failure_rates,
    load_benchmark_episodes,
    wilson_interval,
)


def _rows():
    return [
        {
            "difficulty": "hard",
            "episode_index": index + 1,
            "success": index < 3,
            "failure_type": "none" if index < 3 else "lift_failure",
            "object_x_m": 0.4 + 0.1 * index,
            "object_y_m": 0.0,
            "object_yaw_deg": -150.0 + 100.0 * index,
            "object_mass_kg": 0.05 + 0.1 * index,
            "static_friction": 0.1 + 0.2 * index,
            "dynamic_friction": 0.08 + 0.15 * index,
            "maximum_object_height_m": 0.5,
            "minimum_ee_object_distance_m": 0.02,
            "final_controller_state": 4,
            "object_planar_distance_m": 0.4 + 0.1 * index,
            "is_failure": index >= 3,
        }
        for index in range(4)
    ]


def _write_input(path):
    fields = [field for field in _rows()[0] if field not in {"object_planar_distance_m", "is_failure"}]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_rows())


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(2, 10)
    assert low < 0.2 < high
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_binned_failure_rates_reconcile_counts():
    result = binned_failure_rates(_rows(), "static_friction", bins=2)
    assert sum(row["sample_count"] for row in result) == 4
    assert sum(row["failure_count"] for row in result) == 1
    assert result[-1]["failure_rate"] == 0.5


def test_load_rejects_inconsistent_success(tmp_path):
    path = tmp_path / "episodes.csv"
    _write_input(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["failure_type"] = "lift_failure"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Inconsistent"):
        load_benchmark_episodes(path)


def test_end_to_end_analysis_outputs_reconcile(tmp_path):
    input_path = tmp_path / "episodes.csv"
    output_path = tmp_path / "analysis"
    _write_input(input_path)
    report = analyze_failure_file(input_path, output_path)

    assert report["episode_count"] == 4
    assert report["failure_count"] == 1
    assert report["failure_rate"] == 0.25
    assert all((output_path / f"failure_by_{name}.csv").is_file() for name in ("friction", "distance", "mass", "orientation"))
    assert all((output_path / f"failure_by_{name}.png").stat().st_size > 1000 for name in ("friction", "distance", "mass", "orientation"))
    saved = json.loads((output_path / "failure_analysis_report.json").read_text(encoding="utf-8"))
    assert saved["status"] == "PHASE8_FAILURE_ANALYSIS_OK"
