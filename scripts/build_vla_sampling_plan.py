"""Build a deterministic Hard/LIFT-weighted sampler plan for LeRobot training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.sampling import (  # noqa: E402
    CONTROLLER_PHASE_NAMES,
    LEVELS,
    build_frame_sampling_weights,
    deterministic_sampling_preview,
    file_sha256,
    save_sampling_plan,
    summarize_sampling_plan,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Samples per sampler epoch; 0 preserves the original frame count.",
    )
    parser.add_argument("--easy-weight", type=float, default=1.0)
    parser.add_argument("--medium-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=2.0)
    parser.add_argument("--balance-levels", action="store_true",
                        help="Normalize each difficulty's total sampling mass after phase weights.")
    parser.add_argument("--phase-0-weight", type=float, default=1.0)
    parser.add_argument("--phase-1-weight", type=float, default=1.0)
    parser.add_argument("--phase-2-weight", type=float, default=1.25)
    parser.add_argument("--phase-3-weight", type=float, default=1.5)
    parser.add_argument("--phase-4-weight", type=float, default=3.0)
    parser.add_argument(
        "--dagger-recovery-weight",
        type=float,
        default=4.0,
        help=(
            "Additional train-only sampling multiplier for manifest episodes "
            "marked dagger_recovery=true."
        ),
    )
    return parser.parse_args(argv)


def phase_values_from_columns(states, diagnostic_column=None):
    if diagnostic_column is None:
        return states[:, -1].astype(np.float64)
    values = np.asarray(diagnostic_column)
    if values.shape == (len(states), 1): values = values[:, 0]
    if values.shape != (len(states),):
        raise ValueError("Diagnostic phase must be scalar per frame")
    return values.astype(np.float64)


def _load_frame_attributes(
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object], Path]:
    """Read frame difficulty and phase without decoding any images."""

    manifest_path = dataset_root / "balanced_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "train":
        raise ValueError(
            "Sampling plans may only be built for the train split; "
            f"manifest split is {manifest.get('split')!r}"
        )
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("balanced_split_manifest.json has no episode records")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; run this script in the lerobot-vla environment"
        ) from exc

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_root / 'data'}")
    phase_free = bool(manifest.get("interface_contract", {}).get("phase_in_input") is False)
    columns = ["index", "episode_index", "frame_index", "observation.state"]
    if phase_free:
        columns.append("training.controller_phase")
    tables = [
        pq.read_table(
            path,
            columns=columns,
        )
        for path in parquet_files
    ]
    table = pa.concat_tables(tables, promote_options="default")
    indices = np.asarray(table["index"].combine_chunks().to_numpy(), dtype=np.int64)
    order = np.argsort(indices)
    indices = indices[order]
    expected_indices = np.arange(len(indices), dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("LeRobot global frame indices are not contiguous and unique")

    episode_indices = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(), dtype=np.int64
    )[order]
    frame_indices = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(), dtype=np.int64
    )[order]
    states = np.asarray(table["observation.state"].combine_chunks().to_pylist())[
        order
    ]
    if states.ndim != 2 or states.shape[1] < 1:
        raise ValueError(f"Expected a two-dimensional state array, got {states.shape}")
    diagnostic = (np.asarray(table["training.controller_phase"].combine_chunks().to_pylist())[order]
                  if phase_free else None)
    phase_values = phase_values_from_columns(states, diagnostic)
    if not np.isfinite(phase_values).all() or not np.allclose(
        phase_values, np.rint(phase_values), atol=0.0, rtol=0.0
    ):
        raise ValueError("Last state element is not an integer controller phase")
    phases = np.rint(phase_values).astype(np.int64)
    if np.any((phases < 0) | (phases > 4)):
        raise ValueError("Controller phase must be within [0, 4]")

    if int(manifest.get("frames", -1)) != len(indices):
        raise ValueError(
            f"Manifest/parquet frame mismatch: {manifest.get('frames')} vs {len(indices)}"
        )
    if int(manifest.get("episodes", -1)) != len(records):
        raise ValueError("Manifest episode count does not match record count")
    if np.any((episode_indices < 0) | (episode_indices >= len(records))):
        raise ValueError("Parquet contains an out-of-range episode index")

    levels_by_episode = np.asarray([record["level"] for record in records], dtype=object)
    if set(str(level) for level in levels_by_episode) - set(LEVELS):
        raise ValueError("Manifest contains an unknown difficulty level")
    levels = levels_by_episode[episode_indices].astype(str)
    recovery_by_episode = np.asarray(
        [bool(record.get("dagger_recovery", False)) for record in records],
        dtype=np.bool_,
    )
    recovery = recovery_by_episode[episode_indices]

    for episode_index, record in enumerate(records):
        mask = episode_indices == episode_index
        expected_frames = int(record["frames"])
        if int(np.count_nonzero(mask)) != expected_frames:
            raise ValueError(
                f"Episode {episode_index} frame-count mismatch: "
                f"manifest={expected_frames}, parquet={np.count_nonzero(mask)}"
            )
        if expected_frames and not np.array_equal(
            np.sort(frame_indices[mask]), np.arange(expected_frames, dtype=np.int64)
        ):
            raise ValueError(f"Episode {episode_index} frame indices are not contiguous")
    return levels, phases, recovery, manifest, manifest_path


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    levels, phases, recovery, manifest, manifest_path = _load_frame_attributes(dataset_root)
    num_samples = len(levels) if args.num_samples == 0 else args.num_samples
    if num_samples < 1:
        raise ValueError("--num-samples must be zero or positive")

    level_multipliers = {
        "easy": args.easy_weight,
        "medium": args.medium_weight,
        "hard": args.hard_weight,
    }
    phase_multipliers = {
        0: args.phase_0_weight,
        1: args.phase_1_weight,
        2: args.phase_2_weight,
        3: args.phase_3_weight,
        4: args.phase_4_weight,
    }
    weights = build_frame_sampling_weights(
        levels,
        phases,
        level_multipliers=level_multipliers,
        phase_multipliers=phase_multipliers,
    )
    if not np.isfinite(args.dagger_recovery_weight) or args.dagger_recovery_weight <= 0.0:
        raise ValueError("--dagger-recovery-weight must be finite and positive")
    weights = weights * np.where(recovery, args.dagger_recovery_weight, 1.0)
    if args.balance_levels:
        for level in LEVELS:
            mask = levels == level
            if not np.any(mask):
                raise ValueError(f"Cannot balance absent difficulty: {level}")
            weights[mask] /= weights[mask].sum()
    weights /= weights.mean()
    preview = deterministic_sampling_preview(
        weights, num_samples=num_samples, seed=args.seed
    )
    distribution = summarize_sampling_plan(
        levels,
        phases,
        weights,
        num_samples=num_samples,
        preview_indices=preview,
    )
    frame_identity = "\n".join(
        f"{index}:{level}:{phase}"
        for index, (level, phase) in enumerate(zip(levels, phases, strict=True))
    ).encode("utf-8")
    metadata = {
        "dataset_root": str(dataset_root),
        "dataset_repo_id": manifest.get("repo_id"),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "frame_identity_sha256": hashlib.sha256(frame_identity).hexdigest(),
        "seed": int(args.seed),
        "num_samples": int(num_samples),
        "replacement": True,
        "level_multipliers": level_multipliers,
        "phase_multipliers": {
            str(key): {"name": CONTROLLER_PHASE_NAMES[key], "multiplier": value}
            for key, value in phase_multipliers.items()
        },
        "dagger_recovery_multiplier": float(args.dagger_recovery_weight),
        "balance_levels": bool(args.balance_levels),
        "dagger_recovery": {
            "frames": int(np.count_nonzero(recovery)),
            "source_fraction": float(np.mean(recovery)),
            "expected_sampled_fraction": float(weights[recovery].sum() / weights.sum())
            if np.any(recovery)
            else 0.0,
        },
        "distribution": distribution,
        "validation_and_test_policy": "unchanged; sampler plan is train-only",
    }
    payload = save_sampling_plan(args.output.expanduser().resolve(), weights, metadata)
    summary = {
        key: payload[key]
        for key in (
            "strategy",
            "num_frames",
            "num_samples",
            "seed",
            "level_multipliers",
            "phase_multipliers",
            "dagger_recovery_multiplier",
            "dagger_recovery",
            "distribution",
        )
    }
    print("VLA_SAMPLING_PLAN=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return payload


if __name__ == "__main__":
    main()
