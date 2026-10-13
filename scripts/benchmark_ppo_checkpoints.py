"""Benchmark intermediate PPO checkpoints and select the best Hard policy."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument(
    "--iterations",
    type=int,
    nargs="+",
    default=(20, 40, 60, 80, 100),
    help="Checkpoint iteration numbers to evaluate.",
)
parser.add_argument("--levels", nargs="+", default=("easy", "medium", "hard"))
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--episodes-per-level", type=int, default=128)
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument("--action-space", choices=("joint", "ik_abs"), default="ik_abs")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--skip-existing", action="store_true")
args = parser.parse_args()


def _iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(match.group(1))


def _rates(report: dict) -> dict[str, float]:
    return {
        str(summary["difficulty"]): float(summary["success_rate"])
        for summary in report["summaries"]
    }


def main() -> None:
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if min(args.num_envs, args.episodes_per_level, args.max_steps) < 1:
        raise ValueError("num-envs, episodes-per-level, and max-steps must be positive")
    checkpoints = [run_dir / f"model_{iteration}.pt" for iteration in args.iterations]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(map(str, missing)))

    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else ROOT / "results" / "ppo_checkpoint_benchmarks" / run_dir.name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    rows: list[dict] = []

    for checkpoint in checkpoints:
        iteration = _iteration(checkpoint)
        checkpoint_output = output_root / f"iteration_{iteration:04d}"
        report_path = checkpoint_output / "benchmark_report.json"
        if not (args.skip_existing and report_path.is_file()):
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_ppo.py"),
                "--headless",
                "--device",
                args.device,
                "--checkpoint",
                str(checkpoint),
                "--agent-config",
                str(run_dir / "agent_config.yaml"),
                "--action-space",
                args.action_space,
                "--levels",
                *args.levels,
                "--num-envs",
                str(args.num_envs),
                "--episodes-per-level",
                str(args.episodes_per_level),
                "--max-steps",
                str(args.max_steps),
                "--output-dir",
                str(checkpoint_output),
            ]
            print("CHECKPOINT_BENCHMARK_START=" + json.dumps({"iteration": iteration}), flush=True)
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rates = _rates(report)
        rows.append(
            {
                "iteration": iteration,
                "checkpoint": str(checkpoint),
                "success_rates": rates,
                "report": str(report_path),
            }
        )

    # Hard is the primary target; Medium and Easy break ties deterministically.
    best = max(
        rows,
        key=lambda row: (
            row["success_rates"].get("hard", float("-inf")),
            row["success_rates"].get("medium", float("-inf")),
            row["success_rates"].get("easy", float("-inf")),
            -row["iteration"],
        ),
    )
    summary = {
        "status": "PPO_CHECKPOINT_BENCHMARK_OK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "levels": list(args.levels),
        "episodes_per_level": args.episodes_per_level,
        "num_envs": args.num_envs,
        "selection_metric": "hard success rate, then medium, then easy",
        "checkpoints": rows,
        "best_checkpoint": best,
    }
    summary_path = output_root / "checkpoint_benchmark_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, summary_path)
    print("PPO_CHECKPOINT_SELECTION=" + json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
