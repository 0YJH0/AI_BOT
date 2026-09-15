"""Build the Phase 9 randomization config from Phase 8 failure cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaac_lab_data_engine.data_engine import build_failure_aware_config


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "randomization_hard.yaml")
parser.add_argument(
    "--failure-cases",
    type=Path,
    default=ROOT / "results" / "failure_analysis" / "failure_cases.csv",
)
parser.add_argument(
    "--analysis-report",
    type=Path,
    default=ROOT / "results" / "failure_analysis" / "failure_analysis_report.json",
)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "configs" / "randomization_failure_aware.yaml",
)
parser.add_argument("--seed", type=int, default=9101)
parser.add_argument("--hard-case-probability", type=float, default=0.70)
args = parser.parse_args()


if __name__ == "__main__":
    path = build_failure_aware_config(
        base_config=args.base_config,
        failure_cases_csv=args.failure_cases,
        analysis_report=args.analysis_report,
        output_path=args.output,
        seed=args.seed,
        hard_case_probability=args.hard_case_probability,
    )
    print(f"PHASE9_CONFIG={path}")
