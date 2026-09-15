"""Analyze Phase 7 benchmark failures and generate Phase 8 reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaac_lab_data_engine.analysis import analyze_failure_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input",
    type=Path,
    default=PROJECT_ROOT / "results" / "benchmark" / "benchmark_episodes.csv",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=PROJECT_ROOT / "results" / "failure_analysis",
)
parser.add_argument(
    "--difficulty",
    default="hard",
    help="Difficulty to analyze; pass 'all' to include every level.",
)
args = parser.parse_args()


def main() -> None:
    difficulty = None if args.difficulty.lower() == "all" else args.difficulty.lower()
    report = analyze_failure_file(args.input, args.output_dir, difficulty=difficulty)
    print(
        "PHASE8_SUMMARY="
        + json.dumps(
            {
                "status": report["status"],
                "difficulty": report["difficulty"],
                "episodes": report["episode_count"],
                "failures": report["failure_count"],
                "failure_rate": report["failure_rate"],
                "report_json": report["files"]["report_json"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
