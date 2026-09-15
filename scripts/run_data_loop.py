"""Run or reuse the Phase 8 -> Phase 9 evaluation-driven data loop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from isaac_lab_data_engine.analysis import analyze_failure_file
from isaac_lab_data_engine.data_engine import (
    build_failure_aware_config,
    compare_sampling_rounds,
)


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reuse-existing", action="store_true", help="Only compare existing round outputs.")
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=256)
args = parser.parse_args()


def main() -> None:
    baseline = ROOT / "results" / "benchmark" / "benchmark_episodes.csv"
    analysis_dir = ROOT / "results" / "failure_analysis"
    adaptive_config = ROOT / "configs" / "randomization_failure_aware.yaml"
    adaptive_benchmark_config = ROOT / "configs" / "benchmark_failure_aware.yaml"
    adaptive_episodes = ROOT / "results" / "benchmark_failure_aware" / "benchmark_episodes.csv"

    if not args.reuse_existing:
        analyze_failure_file(baseline, analysis_dir, difficulty="hard")
        build_failure_aware_config(
            base_config=ROOT / "configs" / "randomization_hard.yaml",
            failure_cases_csv=analysis_dir / "failure_cases.csv",
            analysis_report=analysis_dir / "failure_analysis_report.json",
            output_path=adaptive_config,
            seed=9101,
            hard_case_probability=0.70,
        )
        environment = dict(os.environ)
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "run_benchmark.py"),
                "--headless",
                "--config",
                str(adaptive_benchmark_config),
                "--num-envs",
                str(args.num_envs),
                "--episodes_per_level",
                str(args.episodes),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
        )

    report = compare_sampling_rounds(
        baseline_csv=baseline,
        adaptive_csv=adaptive_episodes,
        failure_analysis_dir=analysis_dir,
        output_dir=ROOT / "results" / "data_loop",
    )
    print(
        "PHASE9_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "failure_yield_ratio": report["failure_yield_ratio"],
                "additional_failure_episodes": report["additional_failure_episodes"],
                "report_json": report["files"]["report_json"],
            }
        )
    )


if __name__ == "__main__":
    main()
