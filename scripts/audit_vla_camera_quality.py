"""Read-only RGB quality audit for successful VLA camera episodes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/vla/camera_quality_audit.json"
KNOWN_LEVELS = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class SourceSpec:
    """One recursive source root and its optional explicit level label."""

    raw: str
    path: Path
    level: str | None


def parse_source_spec(value: str) -> SourceSpec:
    """Parse ``level=path`` or an unlabeled path."""

    level: str | None = None
    path_text = value
    if "=" in value:
        candidate_level, candidate_path = value.split("=", 1)
        if candidate_level.strip() and candidate_path.strip():
            level = candidate_level.strip()
            path_text = candidate_path.strip()
    path = Path(path_text).expanduser().resolve()
    return SourceSpec(raw=value, path=path, level=level)


def _candidate_episode_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if root.name.startswith("episode_") and (root / "metadata.json").is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("episode_*")
        if path.is_dir() and (path / "metadata.json").is_file()
    )


def _read_metadata(episode_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _infer_level(
    source: SourceSpec, episode_dir: Path, metadata: dict[str, Any]
) -> str:
    if source.level:
        return source.level
    for key in ("level", "difficulty", "benchmark_level"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        relative_parents = episode_dir.relative_to(source.path).parts[:-1]
    except ValueError:
        relative_parents = ()
    for part in reversed(relative_parents):
        if part.lower() in KNOWN_LEVELS:
            return part.lower()
    if source.path.name.lower() in KNOWN_LEVELS:
        return source.path.name.lower()
    return "unlabeled"


def _find_camera_file(episode_dir: Path, metadata: dict[str, Any]) -> Path | None:
    candidates: list[str] = []
    declared = metadata.get("camera_file")
    if isinstance(declared, str) and Path(declared).name == declared:
        candidates.append(declared)
    candidates.extend(("camera_frames.npz", "camera.npz"))
    for name in dict.fromkeys(candidates):
        path = episode_dir / name
        if path.is_file():
            return path
    return None


def inspect_episode_camera(
    episode_dir: Path,
    metadata: dict[str, Any],
    *,
    maximum_frames: int = 3,
    minimum_std: float = 5.0,
    minimum_range: float = 20.0,
) -> dict[str, Any]:
    """Inspect up to the first three RGB frames without changing the episode."""

    if maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    if minimum_std < 0.0 or minimum_range < 0.0:
        raise ValueError("RGB thresholds must be non-negative")
    camera_path = _find_camera_file(episode_dir, metadata)
    if camera_path is None:
        return {
            "valid": False,
            "reason": "missing_camera_npz",
            "camera_path": None,
            "inspected_frames": 0,
            "frame_metrics": [],
        }

    try:
        with np.load(camera_path, allow_pickle=False) as camera:
            if "rgb" not in camera.files:
                raise KeyError("rgb")
            rgb = np.asarray(camera["rgb"])
    except (OSError, ValueError, KeyError) as error:
        return {
            "valid": False,
            "reason": f"camera_load_error:{type(error).__name__}",
            "camera_path": str(camera_path.resolve()),
            "inspected_frames": 0,
            "frame_metrics": [],
        }

    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        return {
            "valid": False,
            "reason": f"invalid_rgb_shape:{list(rgb.shape)}",
            "camera_path": str(camera_path.resolve()),
            "inspected_frames": 0,
            "frame_metrics": [],
        }
    count = min(maximum_frames, int(rgb.shape[0]))
    if count == 0:
        return {
            "valid": False,
            "reason": "no_rgb_frames",
            "camera_path": str(camera_path.resolve()),
            "inspected_frames": 0,
            "frame_metrics": [],
        }

    frame_metrics = []
    reasons: set[str] = set()
    for index in range(count):
        frame = np.asarray(rgb[index, ..., :3], dtype=np.float32)
        if not np.isfinite(frame).all():
            standard_deviation = None
            dynamic_range = None
            frame_valid = False
            reasons.add("nonfinite_rgb")
        else:
            standard_deviation = float(frame.std())
            dynamic_range = float(frame.max() - frame.min())
            frame_valid = (
                standard_deviation >= minimum_std
                and dynamic_range >= minimum_range
            )
            if standard_deviation < minimum_std:
                reasons.add("low_std")
            if dynamic_range < minimum_range:
                reasons.add("low_range")
        frame_metrics.append(
            {
                "frame_index": index,
                "global_std": standard_deviation,
                "dynamic_range": dynamic_range,
                "valid": frame_valid,
            }
        )
    valid = all(item["valid"] for item in frame_metrics)
    return {
        "valid": valid,
        "reason": "ok" if valid else "+".join(sorted(reasons)),
        "camera_path": str(camera_path.resolve()),
        "inspected_frames": count,
        "frame_metrics": frame_metrics,
    }


def _empty_counts() -> dict[str, int]:
    return {
        "successful_episodes": 0,
        "valid_episodes": 0,
        "invalid_episodes": 0,
        "inspected_frames": 0,
    }


def _increment(counts: dict[str, int], result: dict[str, Any]) -> None:
    counts["successful_episodes"] += 1
    counts["valid_episodes" if result["valid"] else "invalid_episodes"] += 1
    counts["inspected_frames"] += int(result["inspected_frames"])


def audit_sources(
    sources: list[SourceSpec],
    *,
    maximum_frames: int = 3,
    minimum_std: float = 5.0,
    minimum_range: float = 20.0,
) -> dict[str, Any]:
    """Audit successful episodes recursively and return a JSON-ready report."""

    if not sources:
        raise ValueError("At least one source is required")
    overall = _empty_counts()
    by_level: dict[str, dict[str, int]] = defaultdict(_empty_counts)
    source_summaries = []
    invalid_samples = []
    seen: set[Path] = set()
    duplicates_skipped = 0

    for source_index, source in enumerate(sources):
        source_counts = _empty_counts()
        source_levels: dict[str, dict[str, int]] = defaultdict(_empty_counts)
        metadata_errors: list[str] = []
        for episode_dir in _candidate_episode_dirs(source.path):
            resolved_episode = episode_dir.resolve()
            if resolved_episode in seen:
                duplicates_skipped += 1
                continue
            metadata = _read_metadata(resolved_episode)
            if metadata is None:
                metadata_errors.append(str(resolved_episode))
                continue
            if not bool(metadata.get("success", False)):
                continue
            seen.add(resolved_episode)
            level = _infer_level(source, resolved_episode, metadata)
            result = inspect_episode_camera(
                resolved_episode,
                metadata,
                maximum_frames=maximum_frames,
                minimum_std=minimum_std,
                minimum_range=minimum_range,
            )
            _increment(source_counts, result)
            _increment(source_levels[level], result)
            _increment(by_level[level], result)
            _increment(overall, result)
            if not result["valid"]:
                invalid_samples.append(
                    {
                        "source_index": source_index,
                        "source": str(source.path),
                        "level": level,
                        "episode_path": str(resolved_episode),
                        **result,
                    }
                )
        source_summaries.append(
            {
                "source_index": source_index,
                "source_argument": source.raw,
                "source_path": str(source.path),
                "explicit_level": source.level,
                **source_counts,
                "levels": dict(sorted(source_levels.items())),
                "metadata_error_count": len(metadata_errors),
                "metadata_error_paths": metadata_errors,
            }
        )

    return {
        "status": "VLA_CAMERA_QUALITY_AUDIT_OK",
        "read_only": True,
        "thresholds": {
            "maximum_initial_frames": maximum_frames,
            "minimum_global_std": minimum_std,
            "minimum_dynamic_range": minimum_range,
            "episode_rule": "all inspected initial RGB frames must pass both thresholds",
        },
        "camera_file_resolution": (
            "metadata.camera_file, then camera_frames.npz, then camera.npz"
        ),
        "overall": overall,
        "by_level": dict(sorted(by_level.items())),
        "sources": source_summaries,
        "duplicate_episode_paths_skipped": duplicates_skipped,
        "invalid_samples": invalid_samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Recursive source as PATH or LEVEL=PATH; may be repeated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--min-std", type=float, default=5.0)
    parser.add_argument("--min-range", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_sources(
        [parse_source_spec(value) for value in args.source],
        maximum_frames=args.frames,
        minimum_std=args.min_std,
        minimum_range=args.min_range,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print("VLA_CAMERA_QUALITY_AUDIT=" + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
