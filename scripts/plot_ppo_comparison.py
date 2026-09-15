"""Plot the Phase 11 PPO versus scripted-IK benchmark comparison."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input",
    type=Path,
    default=ROOT / "results" / "ppo_evaluation_shaped" / "ppo_vs_scripted_ik.csv",
)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "results" / "ppo_evaluation_shaped" / "ppo_vs_scripted_ik.png",
)


def main() -> None:
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = [row["difficulty"].title() for row in rows]
    scripted = [100.0 * float(row["scripted_ik_success_rate"]) for row in rows]
    ppo = [100.0 * float(row["ppo_success_rate"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars_ik = axis.bar(x - width / 2, scripted, width, label="Scripted IK", color="#2878B5")
    bars_ppo = axis.bar(x + width / 2, ppo, width, label="PPO model_500", color="#D9534F")
    axis.bar_label(bars_ik, fmt="%.1f%%", padding=3)
    axis.bar_label(bars_ppo, fmt="%.1f%%", padding=3)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 108)
    axis.set_ylabel("Success rate (%)")
    axis.set_title("Phase 11: Scripted IK vs PPO under the same 10 cm lift criterion")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
