"""Audit compact VLA DAgger recovery episodes before LeRobot export."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.recovery import (  # noqa: E402
    build_pregrasp_recovery_actions,
    validate_recovery_episode,
)


LEVELS = ("easy", "medium", "hard")
SPLITS = ("train", "validation")
BIN_EDGES_M = np.asarray(
    [0.0, 0.0075, 0.015, 0.0225, 0.030, 0.045, np.inf], dtype=np.float64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--disjoint-from-root",
        type=Path,
        action="append",
        default=[],
        help="Reject any domain seed/sample pair also present under this raw dataset root.",
    )
    return parser.parse_args()


def _bin_labels() -> list[str]:
    labels = []
    for low, high in zip(BIN_EDGES_M[:-1], BIN_EDGES_M[1:], strict=True):
        labels.append(
            f"{low:.4f}_{high:.4f}" if np.isfinite(high) else f"{low:.4f}_inf"
        )
    return labels


def main() -> dict[str, object]:
    args = parse_args()
    root = args.source_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    collection = json.loads((root / "collection_summary.json").read_text(encoding="utf-8"))
    band_min = float(collection["band_m"]["minimum"])
    band_max = float(collection["band_m"]["maximum"])
    distances: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    learner_distances: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    seed_distances: dict[tuple[str, str], list[float]] = defaultdict(list)
    episode_counts: Counter[tuple[str, str]] = Counter()
    checkpoint_roles: set[str] = set()
    learner_frames_per_episode: list[int] = []
    domain_keys: dict[str, set[tuple[int, int]]] = defaultdict(set)
    label_max_abs_error = 0.0
    minimum_rgb_std = float("inf")
    minimum_rgb_range = float("inf")
    minimum_depth_fraction = float("inf")
    maximum_action_norm = 0.0
    total_frames = 0
    total_learner_frames = 0

    for split in SPLITS:
        for level in LEVELS:
            paths = sorted((root / split / level).glob("episode_*.npz"))
            if not paths:
                raise FileNotFoundError(f"No episodes for {split}/{level}")
            for path in paths:
                metadata_path = path.with_suffix(".json")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("split") != split or metadata.get("level") != level:
                    raise ValueError(f"Path/metadata mismatch: {metadata_path}")
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                frames = validate_recovery_episode(arrays)
                if frames != int(metadata["frames"]):
                    raise ValueError(f"Frame-count mismatch: {path}")
                if collection.get("require_complete_rollout") and frames != int(collection["frames_per_rollout"]):
                    raise ValueError(f"Incomplete recovery rollout: {path}")
                expected = build_pregrasp_recovery_actions(
                    arrays["eef_pose_b"], arrays["visual_position_b"]
                )
                label_max_abs_error = max(
                    label_max_abs_error,
                    float(np.max(np.abs(expected - arrays["action"]))),
                )
                action_norm = np.linalg.norm(arrays["action"][:, :3], axis=-1)
                maximum_action_norm = max(maximum_action_norm, float(action_norm.max()))
                current_distances = arrays["distance_to_pregrasp_m"].reshape(-1)
                learner = arrays["learner_visited"].reshape(-1).astype(bool)
                if not band_min <= float(current_distances[0]) <= band_max:
                    raise ValueError(f"Seed state outside collection band: {path}")
                if collection.get("require_learner_visited_in_band") and not np.all(
                    (current_distances[learner] >= band_min)
                    & (current_distances[learner] <= band_max)
                ):
                    raise ValueError(f"Learner state outside collection band: {path}")
                learner_frames_per_episode.append(int(learner.sum()))
                distances[(split, level)].append(current_distances)
                learner_distances[(split, level)].append(current_distances[learner])
                seed_distances[(split, level)].append(float(current_distances[0]))
                checkpoint_roles.add(str(metadata.get("policy_checkpoint_role", "missing")))
                total_frames += frames
                total_learner_frames += int(learner.sum())
                episode_counts[(split, level)] += 1
                minimum_rgb_std = min(
                    minimum_rgb_std, float(arrays["rgb_std"].min())
                )
                minimum_rgb_range = min(
                    minimum_rgb_range, float(arrays["rgb_range"].min())
                )
                minimum_depth_fraction = min(
                    minimum_depth_fraction,
                    float(arrays["depth_valid_fraction"].min()),
                )
                domain = metadata["domain_parameters"]
                domain_keys[split].add(
                    (int(domain["randomization_seed"]), int(domain["sample_index"]))
                )

    overlap = domain_keys["train"] & domain_keys["validation"]
    if overlap:
        raise RuntimeError(f"Train/validation randomized-scene overlap: {sorted(overlap)}")
    if label_max_abs_error > 1.0e-6:
        raise RuntimeError(f"Expert label recomputation mismatch: {label_max_abs_error}")
    if maximum_action_norm > 0.040001:
        raise RuntimeError(f"Recovery delta exceeds 4 cm: {maximum_action_norm}")
    if minimum_rgb_std < 30.0 or minimum_rgb_range < 128.0 or minimum_depth_fraction < 0.40:
        raise RuntimeError("Recovery frames fail the RGB-D quality gate")
    if set(learner_frames_per_episode) != {1}:
        raise RuntimeError(
            "Each DAgger recovery episode must contain exactly one learner-visited frame: "
            f"observed={sorted(set(learner_frames_per_episode))}"
        )

    external_overlap: dict[str, int] = {}
    current_domains = domain_keys["train"] | domain_keys["validation"]
    for comparison_root in args.disjoint_from_root:
        comparison_root = comparison_root.expanduser().resolve()
        comparison_domains: set[tuple[int, int]] = set()
        for metadata_path in comparison_root.rglob("episode_*.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            domain = metadata["domain_parameters"]
            comparison_domains.add(
                (int(domain["randomization_seed"]), int(domain["sample_index"]))
            )
        if not comparison_domains:
            raise FileNotFoundError(f"No episode metadata under {comparison_root}")
        count = len(current_domains & comparison_domains)
        external_overlap[str(comparison_root)] = count
        if count:
            raise RuntimeError(
                f"Domain overlap with {comparison_root}: {count} seed/sample pairs"
            )

    labels = _bin_labels()
    distributions: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        distributions[split] = {}
        for level in LEVELS:
            values = np.concatenate(distances[(split, level)])
            learner_values = np.concatenate(learner_distances[(split, level)])
            seed_values = np.asarray(seed_distances[(split, level)], dtype=np.float64)
            counts, _ = np.histogram(values, bins=BIN_EDGES_M)
            learner_counts, _ = np.histogram(learner_values, bins=BIN_EDGES_M)
            distributions[split][level] = {
                "episodes": int(episode_counts[(split, level)]),
                "frames": int(len(values)),
                "learner_visited_frames": int(len(learner_values)),
                "learner_within_collection_band_frames": int(
                    np.count_nonzero(
                        (learner_values >= band_min) & (learner_values <= band_max)
                    )
                ),
                "seed_distance_min_cm": float(seed_values.min() * 100.0),
                "seed_distance_mean_cm": float(seed_values.mean() * 100.0),
                "seed_distance_max_cm": float(seed_values.max() * 100.0),
                "seed_within_1_to_2cm_episodes": int(
                    np.count_nonzero((seed_values >= 0.010) & (seed_values <= 0.020))
                ),
                "distance_mean_cm": float(values.mean() * 100.0),
                "distance_p50_cm": float(np.quantile(values, 0.50) * 100.0),
                "distance_p90_cm": float(np.quantile(values, 0.90) * 100.0),
                "within_3cm_frames": int(np.count_nonzero(values <= 0.03)),
                "within_1cm_of_threshold_frames": int(
                    np.count_nonzero(np.abs(values - 0.015) <= 0.01)
                ),
                "distance_bin_counts": dict(zip(labels, counts.tolist(), strict=True)),
                "learner_distance_bin_counts": dict(
                    zip(labels, learner_counts.tolist(), strict=True)
                ),
            }

    summary = {
        "status": "VLA_DAGGER_RECOVERY_AUDIT_OK",
        "source_root": str(root),
        "episodes": int(sum(episode_counts.values())),
        "frames": total_frames,
        "learner_visited_frames": total_learner_frames,
        "policy_checkpoint_roles": sorted(checkpoint_roles),
        "label_max_abs_error": label_max_abs_error,
        "maximum_delta_norm_m": maximum_action_norm,
        "camera_quality_minimums": {
            "rgb_std": minimum_rgb_std,
            "rgb_range": minimum_rgb_range,
            "depth_valid_fraction": minimum_depth_fraction,
        },
        "train_validation_domain_overlap": 0,
        "external_domain_overlap": external_overlap,
        "distributions": distributions,
    }
    if args.output:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("DAGGER_RECOVERY_AUDIT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
