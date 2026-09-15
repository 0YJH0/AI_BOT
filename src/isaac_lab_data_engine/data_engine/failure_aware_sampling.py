"""Failure-aware sampling driven by Phase 8 mined failure cases."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from isaac_lab_data_engine.randomization.domain_randomizer import DomainRandomizer, _range


REQUIRED_FAILURE_FIELDS = {
    "episode_index",
    "failure_type",
    "object_x_m",
    "object_y_m",
    "object_yaw_deg",
    "object_mass_kg",
    "static_friction",
    "dynamic_friction",
}

DEFAULT_JITTER_FRACTION = {
    "position_x_m": 0.06,
    "position_y_m": 0.06,
    "yaw_deg": 0.05,
    "mass_kg": 0.06,
    "static_friction": 0.06,
    "dynamic_friction_ratio": 0.04,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _relative_to_config(path: Path, config_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), config_path.parent.resolve())).as_posix()


def build_failure_aware_config(
    *,
    base_config: str | Path,
    failure_cases_csv: str | Path,
    analysis_report: str | Path,
    output_path: str | Path,
    seed: int = 9101,
    hard_case_probability: float = 0.70,
    jitter_fraction: dict[str, float] | None = None,
) -> Path:
    """Create a self-auditing failure-aware randomization YAML."""
    base_config = Path(base_config).resolve()
    failure_cases_csv = Path(failure_cases_csv).resolve()
    analysis_report = Path(analysis_report).resolve()
    output_path = Path(output_path).resolve()
    if not 0.0 <= hard_case_probability <= 1.0:
        raise ValueError("hard_case_probability must be in [0, 1]")
    for path in (base_config, failure_cases_csv, analysis_report):
        if not path.is_file():
            raise FileNotFoundError(path)

    jitter = dict(DEFAULT_JITTER_FRACTION)
    jitter.update(jitter_fraction or {})
    unknown = set(jitter) - set(DEFAULT_JITTER_FRACTION)
    if unknown:
        raise ValueError(f"Unknown jitter fields: {sorted(unknown)}")
    if any(not 0.0 <= float(value) <= 0.5 for value in jitter.values()):
        raise ValueError("Every jitter fraction must be in [0, 0.5]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": int(seed),
        "base_config": _relative_to_config(base_config, output_path),
        "strategy": {
            "type": "failure_replay_jitter",
            "hard_case_probability": float(hard_case_probability),
            "failure_cases_csv": _relative_to_config(failure_cases_csv, output_path),
            "failure_cases_sha256": _sha256(failure_cases_csv),
            "analysis_report": _relative_to_config(analysis_report, output_path),
            "analysis_report_sha256": _sha256(analysis_report),
            "jitter_fraction_of_base_range": {
                key: float(value) for key, value in jitter.items()
            },
        },
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


class FailureAwareDomainRandomizer(DomainRandomizer):
    """Sample near empirical failure cases while retaining uniform exploration."""

    def __init__(self, config_path: str | Path) -> None:
        adaptive_path = Path(config_path).resolve()
        adaptive = yaml.safe_load(adaptive_path.read_text(encoding="utf-8"))
        strategy = adaptive.get("strategy", {})
        if strategy.get("type") != "failure_replay_jitter":
            raise ValueError("Expected strategy.type=failure_replay_jitter")
        base_path = _resolve(adaptive_path, str(adaptive["base_config"]))
        super().__init__(base_path)

        self.base_config_path = base_path
        self.config_path = adaptive_path
        self.seed = int(adaptive["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.sample_index = 0
        self.hard_case_probability = float(strategy["hard_case_probability"])
        if not 0.0 <= self.hard_case_probability <= 1.0:
            raise ValueError("hard_case_probability must be in [0, 1]")
        self.jitter_fraction = {
            key: float(value)
            for key, value in strategy["jitter_fraction_of_base_range"].items()
        }
        if set(self.jitter_fraction) != set(DEFAULT_JITTER_FRACTION):
            raise ValueError("jitter_fraction_of_base_range has missing or unknown fields")
        if any(not 0.0 <= value <= 0.5 for value in self.jitter_fraction.values()):
            raise ValueError("Every jitter fraction must be in [0, 0.5]")

        self.failure_cases_path = _resolve(adaptive_path, str(strategy["failure_cases_csv"]))
        if _sha256(self.failure_cases_path) != str(strategy["failure_cases_sha256"]):
            raise ValueError("Failure cases SHA-256 does not match the adaptive config")
        analysis_path = _resolve(adaptive_path, str(strategy["analysis_report"]))
        if _sha256(analysis_path) != str(strategy["analysis_report_sha256"]):
            raise ValueError("Analysis report SHA-256 does not match the adaptive config")
        self.failure_cases = self._load_failure_cases()

    def _load_failure_cases(self) -> list[dict[str, Any]]:
        with self.failure_cases_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = REQUIRED_FAILURE_FIELDS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Failure cases CSV is missing fields: {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise ValueError("Failure-aware sampling requires at least one failure case")
        return rows

    def _jitter(self, value: float, key: str) -> float:
        low, high = _range(self.config["object"], key)
        standard_deviation = self.jitter_fraction[key] * (high - low)
        sampled = float(value) + float(self.rng.normal(0.0, standard_deviation))
        return float(np.clip(sampled, low, high))

    def sample(self) -> dict[str, Any]:
        sample = super().sample()
        selected = bool(self.rng.random() < self.hard_case_probability)
        source: dict[str, Any] | None = None
        if selected:
            source = self.failure_cases[int(self.rng.integers(0, len(self.failure_cases)))]
            obj = sample["object"]
            obj["position_m"][0] = self._jitter(float(source["object_x_m"]), "position_x_m")
            obj["position_m"][1] = self._jitter(float(source["object_y_m"]), "position_y_m")
            obj["yaw_deg"] = self._jitter(float(source["object_yaw_deg"]), "yaw_deg")
            obj["mass_kg"] = self._jitter(float(source["object_mass_kg"]), "mass_kg")
            obj["static_friction"] = self._jitter(
                float(source["static_friction"]), "static_friction"
            )
            source_ratio = float(source["dynamic_friction"]) / max(
                float(source["static_friction"]), 1.0e-12
            )
            dynamic_ratio = self._jitter(source_ratio, "dynamic_friction_ratio")
            obj["dynamic_friction"] = obj["static_friction"] * dynamic_ratio

        sample["sampling"] = {
            "strategy": "failure_replay_jitter",
            "hard_case_selected": selected,
            "hard_case_probability": self.hard_case_probability,
            "source_failure_episode_index": (
                int(source["episode_index"]) if source is not None else None
            ),
            "source_failure_type": source["failure_type"] if source is not None else None,
        }
        return sample
