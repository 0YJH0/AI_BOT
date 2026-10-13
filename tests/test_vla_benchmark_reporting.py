from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import numpy as np

from isaac_lab_data_engine.benchmark.vla_execution import select_benchmark_action
from isaac_lab_data_engine.benchmark.vla_reporting import (
    summarize_vla_benchmark,
    validate_unique_vla_samples,
    write_vla_benchmark_reports,
)
from isaac_lab_data_engine.vla.hybrid_controller import HybridActionDecision


def _row(
    difficulty: str,
    seed: int,
    sample_index: int,
    *,
    success: bool,
    failure_reason: str,
    vla_actions: int,
    scripted_actions: int,
    inference_ms: list[float],
) -> dict[str, object]:
    total_actions = vla_actions + scripted_actions
    return {
        "difficulty": difficulty,
        "level_seed": seed,
        "episode_index": sample_index,
        "sample_index": sample_index,
        "sample_key": f"{difficulty}:{seed}:{sample_index:06d}",
        "success": success,
        "failure_reason": failure_reason,
        "failure_type": failure_reason,
        "steps": 100,
        "completion_time_s": 2.0,
        "wall_time_s": 1.0,
        "total_reward": 1.0,
        "source_counts": {"vla": vla_actions, "scripted_ik": scripted_actions},
        "decision_reason_counts": {"accepted": vla_actions},
        "executed_action_decisions": total_actions,
        "vla_action_count": vla_actions,
        "scripted_ik_action_count": scripted_actions,
        "scripted_ik_fallback_fraction": (
            scripted_actions / total_actions if total_actions else 0.0
        ),
        "vla_request_count": len(inference_ms),
        "vla_error_count": 0,
        "mean_vla_inference_ms": sum(inference_ms) / len(inference_ms),
        "_vla_inference_ms_samples": inference_ms,
        "_rpc_roundtrip_ms_samples": [value + 2.0 for value in inference_ms],
    }


def test_vla_summary_reports_fallback_failure_latency_and_unique_samples() -> None:
    rows = [
        _row(
            "easy",
            7101,
            1,
            success=True,
            failure_reason="none",
            vla_actions=8,
            scripted_actions=2,
            inference_ms=[10.0, 20.0],
        ),
        _row(
            "easy",
            7101,
            2,
            success=False,
            failure_reason="grasp_failure",
            vla_actions=0,
            scripted_actions=4,
            inference_ms=[30.0],
        ),
        _row(
            "hard",
            7103,
            1,
            success=True,
            failure_reason="none",
            vla_actions=5,
            scripted_actions=0,
            inference_ms=[40.0],
        ),
    ]

    levels, overall = summarize_vla_benchmark(rows, {"easy": 7101, "hard": 7103})

    assert [item["difficulty"] for item in levels] == ["easy", "hard"]
    assert levels[0]["success_rate"] == 0.5
    assert levels[0]["failure_counts"] == {"grasp_failure": 1}
    assert levels[0]["source_counts"] == {"scripted_ik": 6, "vla": 8}
    assert levels[0]["scripted_ik_fallback_fraction"] == pytest.approx(6 / 14)
    assert overall["source_counts"] == {"scripted_ik": 6, "vla": 13}
    assert overall["scripted_ik_fallback_fraction"] == pytest.approx(6 / 19)
    assert overall["mean_vla_inference_ms"] == 25.0
    assert overall["p95_vla_inference_ms"] == pytest.approx(38.5)
    # Easy and Hard both use sample_index=1, but their fixed sample keys differ.
    assert overall["unique_sample_count"] == 3


def test_duplicate_fixed_sample_is_rejected() -> None:
    row = _row(
        "medium",
        7102,
        1,
        success=True,
        failure_reason="none",
        vla_actions=3,
        scripted_actions=0,
        inference_ms=[5.0],
    )
    with pytest.raises(ValueError, match="Duplicate VLA benchmark samples"):
        validate_unique_vla_samples([row, dict(row)])


def test_training_label_execution_is_not_reported_as_vla_or_fallback() -> None:
    row = _row(
        "easy", 7101, 92, success=True, failure_reason="none",
        vla_actions=0, scripted_actions=0, inference_ms=[1.0],
    )
    row.update({
        "source_counts": {"training_label": 100},
        "training_label_request_count": 20,
        "vla_request_count": 0,
        "_vla_inference_ms_samples": [],
        "_rpc_roundtrip_ms_samples": [],
    })
    _, overall = summarize_vla_benchmark([row], {"easy": 7101})
    assert overall["executed_action_decisions"] == 100
    assert overall["training_label_action_count"] == 100
    assert overall["training_label_request_count"] == 20
    assert overall["vla_action_count"] == 0
    assert overall["vla_action_fraction"] == 0.0
    assert overall["scripted_ik_fallback_fraction"] == 0.0
    assert overall["vla_request_count"] == 0
    assert overall["mean_vla_inference_ms"] is None


def test_pure_vla_rejection_never_returns_scripted_action() -> None:
    scripted_action = np.arange(8, dtype=np.float32)
    decision = HybridActionDecision(
        action=scripted_action,
        source="scripted_ik",
        reason="vla_unavailable",
    )

    pure = select_benchmark_action(decision, scripted_fallback_enabled=False)
    hybrid = select_benchmark_action(decision, scripted_fallback_enabled=True)

    assert pure.action is None
    assert pure.source == "rejected_no_fallback"
    assert pure.failure_reason == "pure_vla_vla_unavailable"
    np.testing.assert_array_equal(hybrid.action, scripted_action)
    assert hybrid.source == "scripted_ik"
    assert hybrid.failure_reason is None


def test_vla_reports_write_episode_level_and_overall_artifacts(tmp_path: Path) -> None:
    rows = [
        _row(
            "easy",
            7101,
            1,
            success=True,
            failure_reason="none",
            vla_actions=9,
            scripted_actions=1,
            inference_ms=[12.0],
        )
    ]

    files = write_vla_benchmark_reports(
        tmp_path,
        config_snapshot={"scripted_ik_fallback_enabled": True},
        episode_rows=rows,
        level_seeds={"easy": 7101},
    )

    assert set(files) == {"episodes_csv", "levels_json", "report_json"}
    with Path(files["episodes_csv"]).open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert csv_rows[0]["source_counts"] == '{"vla":9,"scripted_ik":1}'
    assert float(csv_rows[0]["scripted_ik_fallback_fraction"]) == 0.1
    levels_payload = json.loads(Path(files["levels_json"]).read_text(encoding="utf-8"))
    report_payload = json.loads(Path(files["report_json"]).read_text(encoding="utf-8"))
    assert levels_payload["summaries"][0]["difficulty"] == "easy"
    assert report_payload["overall"]["scripted_ik_action_count"] == 1
    assert report_payload["config"]["scripted_ik_fallback_enabled"] is True
