"""Create one comparable table from visual IK, state PPO, and visual PPO reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--visual-ik-report", type=Path, required=True)
parser.add_argument("--state-ppo-report", type=Path, required=True)
parser.add_argument("--visual-ppo-report", type=Path, required=True)
parser.add_argument(
    "--output-dir", type=Path, default=ROOT / "results" / "grasp_mode_comparison"
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "summaries" not in payload:
        raise ValueError(f"Report has no summaries: {path}")
    return payload


def main() -> None:
    args = parser.parse_args()
    inputs = {
        "visual_scripted_ik": args.visual_ik_report,
        "state_ppo": args.state_ppo_report,
        "visual_ppo": args.visual_ppo_report,
    }
    rows = []
    metadata = {}
    for mode, path in inputs.items():
        report = _load(path)
        metadata[mode] = {"report": str(path.resolve()), "config": report.get("config", {})}
        for summary in report["summaries"]:
            rows.append(
                {
                    "mode": mode,
                    "difficulty": summary["difficulty"],
                    "num_episodes": summary["num_episodes"],
                    "num_success": summary["num_success"],
                    "success_rate": summary["success_rate"],
                    "average_completion_time_s": summary["average_completion_time_s"],
                    "failure_counts": json.dumps(summary["failure_counts"], ensure_ascii=False),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "grasp_mode_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "grasp_mode_comparison.json"
    json_path.write_text(
        json.dumps({"modes": metadata, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "results": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
