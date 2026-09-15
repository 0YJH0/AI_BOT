"""Validated Phase 7 benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkLevel:
    name: str
    seed: int
    randomization_config: Path
    description: str


@dataclass(frozen=True)
class BenchmarkConfig:
    path: Path
    version: str
    task: str
    controller: str
    num_envs: int
    episodes_per_level: int
    max_steps: int
    output_dir: Path
    success_criteria: dict[str, float | int]
    levels: tuple[BenchmarkLevel, ...]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path).resolve()
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "benchmark_version",
        "task",
        "controller",
        "num_envs",
        "episodes_per_level",
        "max_steps",
        "output_dir",
        "success_criteria",
        "levels",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Benchmark config is missing keys: {sorted(missing)}")

    num_envs = int(payload["num_envs"])
    episodes_per_level = int(payload["episodes_per_level"])
    max_steps = int(payload["max_steps"])
    if min(num_envs, episodes_per_level, max_steps) < 1:
        raise ValueError("num_envs, episodes_per_level, and max_steps must be positive")

    criteria = payload["success_criteria"]
    criteria_keys = {
        "minimum_height_above_table_m",
        "maximum_mean_finger_position",
        "maximum_ee_object_distance_m",
        "sustained_steps",
    }
    if criteria_keys - set(criteria):
        raise ValueError("Benchmark success_criteria is incomplete")

    levels: list[BenchmarkLevel] = []
    for name, level_payload in payload["levels"].items():
        randomization_path = (path.parent / level_payload["randomization_config"]).resolve()
        if not randomization_path.is_file():
            raise FileNotFoundError(f"Missing randomization config for {name}: {randomization_path}")
        randomization_payload = yaml.safe_load(randomization_path.read_text(encoding="utf-8"))
        seed = int(level_payload["seed"])
        if int(randomization_payload["seed"]) != seed:
            raise ValueError(
                f"Level {name} seed {seed} differs from {randomization_path.name} seed "
                f"{randomization_payload['seed']}"
            )
        levels.append(
            BenchmarkLevel(
                name=str(name),
                seed=seed,
                randomization_config=randomization_path,
                description=str(level_payload["description"]),
            )
        )
    if not levels:
        raise ValueError("Benchmark must define at least one level")

    return BenchmarkConfig(
        path=path,
        version=str(payload["benchmark_version"]),
        task=str(payload["task"]),
        controller=str(payload["controller"]),
        num_envs=num_envs,
        episodes_per_level=episodes_per_level,
        max_steps=max_steps,
        output_dir=(path.parent / payload["output_dir"]).resolve(),
        success_criteria=dict(criteria),
        levels=tuple(levels),
    )
