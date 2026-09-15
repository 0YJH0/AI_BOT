"""Simulator-free tests for Phase 9 failure-aware sampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from isaac_lab_data_engine.data_engine import (
    FailureAwareDomainRandomizer,
    build_failure_aware_config,
)
from isaac_lab_data_engine.randomization import create_domain_randomizer


BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "randomization_hard.yaml"


def _sources(tmp_path):
    failure_path = tmp_path / "failure_cases.csv"
    fields = [
        "episode_index",
        "failure_type",
        "object_x_m",
        "object_y_m",
        "object_yaw_deg",
        "object_mass_kg",
        "static_friction",
        "dynamic_friction",
    ]
    with failure_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "episode_index": 7,
                "failure_type": "lift_failure",
                "object_x_m": 0.75,
                "object_y_m": 0.25,
                "object_yaw_deg": 150.0,
                "object_mass_kg": 0.05,
                "static_friction": 0.12,
                "dynamic_friction": 0.08,
            }
        )
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    return failure_path, analysis_path


def _config(tmp_path, probability=1.0):
    failures, analysis = _sources(tmp_path)
    path = tmp_path / "adaptive.yaml"
    build_failure_aware_config(
        base_config=BASE_CONFIG,
        failure_cases_csv=failures,
        analysis_report=analysis,
        output_path=path,
        seed=99,
        hard_case_probability=probability,
    )
    return path


def test_failure_aware_samples_are_reproducible_and_in_range(tmp_path):
    path = _config(tmp_path)
    first = FailureAwareDomainRandomizer(path)
    second = FailureAwareDomainRandomizer(path)
    samples_a = [first.sample() for _ in range(20)]
    samples_b = [second.sample() for _ in range(20)]

    assert samples_a == samples_b
    assert all(sample["sampling"]["hard_case_selected"] for sample in samples_a)
    assert all(sample["sampling"]["source_failure_episode_index"] == 7 for sample in samples_a)
    for sample in samples_a:
        obj = sample["object"]
        assert 0.36 <= obj["position_m"][0] <= 0.78
        assert -0.30 <= obj["position_m"][1] <= 0.30
        assert 0.02 <= obj["mass_kg"] <= 0.45
        assert 0.08 <= obj["static_friction"] <= 0.95


def test_factory_and_uniform_exploration_branch(tmp_path):
    path = _config(tmp_path, probability=0.0)
    randomizer = create_domain_randomizer(path)
    assert isinstance(randomizer, FailureAwareDomainRandomizer)
    assert all(not randomizer.sample()["sampling"]["hard_case_selected"] for _ in range(10))


def test_source_hash_change_is_rejected(tmp_path):
    path = _config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    failure_path = (path.parent / config["strategy"]["failure_cases_csv"]).resolve()
    with failure_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256"):
        FailureAwareDomainRandomizer(path)
