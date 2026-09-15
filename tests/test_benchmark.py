"""Simulator-free tests for Phase 7 configuration and metrics."""

from pathlib import Path

from isaac_lab_data_engine.benchmark import (
    classify_failure,
    load_benchmark_config,
    summarize_level,
    write_benchmark_reports,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"


def test_benchmark_config_has_reproducible_ordered_levels():
    config = load_benchmark_config(CONFIG)

    assert [level.name for level in config.levels] == ["easy", "medium", "hard"]
    assert [level.seed for level in config.levels] == [7101, 7102, 7103]
    assert all(level.randomization_config.is_file() for level in config.levels)
    assert config.success_criteria["sustained_steps"] == 10


def test_failure_classification_is_explicit():
    assert classify_failure(
        success=True,
        terminated=False,
        truncated=False,
        ever_grasp_closed=True,
        ever_object_held=True,
        ever_lifted=True,
    ) == "none"
    assert classify_failure(
        success=False,
        terminated=False,
        truncated=False,
        ever_grasp_closed=True,
        ever_object_held=False,
        ever_lifted=False,
    ) == "grasp_failure"
    assert classify_failure(
        success=False,
        terminated=False,
        truncated=False,
        ever_grasp_closed=True,
        ever_object_held=True,
        ever_lifted=True,
    ) == "object_drop"


def test_level_summary_and_atomic_reports(tmp_path):
    rows = [
        {"success": True, "failure_type": "none", "completion_time_s": 2.4, "total_reward": 6.0},
        {
            "success": False,
            "failure_type": "grasp_failure",
            "completion_time_s": 6.0,
            "total_reward": 0.5,
        },
    ]
    metrics = [{"wall_time_s": 1.0, "simulated_environment_steps": 400}]
    summary = summarize_level("hard", 7, rows, metrics)

    assert summary["success_rate"] == 0.5
    assert summary["average_completion_time_s"] == 2.4
    assert summary["grasp_failure_rate"] == 0.5
    assert summary["collision_count_available"] is False

    episode_rows = []
    for index, row in enumerate(rows, start=1):
        episode_rows.append(
            {
                **row,
                "difficulty": "hard",
                "level_seed": 7,
                "episode_index": index,
                "camera_position_m": [1.0, 2.0, 3.0],
            }
        )
    files = write_benchmark_reports(
        tmp_path,
        config_snapshot={"test": True},
        summaries=[summary],
        episode_rows=episode_rows,
    )
    assert all(Path(path).is_file() for path in files.values())
