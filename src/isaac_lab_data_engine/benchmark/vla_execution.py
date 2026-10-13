"""Simulator-free action-source resolution for hybrid and pure-VLA evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaac_lab_data_engine.vla.hybrid_controller import HybridActionDecision


@dataclass(frozen=True)
class BenchmarkActionSelection:
    """An action to execute, or an explicit pure-VLA rejection."""

    action: np.ndarray | None
    source: str
    failure_reason: str | None


def select_benchmark_action(
    decision: HybridActionDecision,
    *,
    scripted_fallback_enabled: bool,
) -> BenchmarkActionSelection:
    """Prevent a Scripted IK action from leaking into strict pure-VLA runs."""

    if decision.source == "scripted_ik" and not scripted_fallback_enabled:
        return BenchmarkActionSelection(
            action=None,
            source="rejected_no_fallback",
            failure_reason=f"pure_vla_{decision.reason}",
        )
    return BenchmarkActionSelection(
        action=decision.action,
        source=decision.source,
        failure_reason=None,
    )
