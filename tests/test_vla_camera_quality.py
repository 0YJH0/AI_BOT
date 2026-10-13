"""Simulator-free tests for the recursive VLA camera quality audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.audit_vla_camera_quality import (
    audit_sources,
    inspect_episode_camera,
    parse_source_spec,
)


def _episode(
    root: Path,
    name: str,
    *,
    success: bool,
    rgb: np.ndarray | None,
    camera_name: str = "camera_frames.npz",
) -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    metadata = {"success": success, "camera_frames": 0 if rgb is None else len(rgb)}
    if rgb is not None:
        metadata["camera_file"] = camera_name
        np.savez_compressed(episode / camera_name, rgb=rgb)
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return episode


def _textured_frames(count: int = 4) -> np.ndarray:
    checkerboard = (np.indices((8, 10)).sum(axis=0) % 2 * 255).astype(np.uint8)
    frame = np.repeat(checkerboard[..., None], 3, axis=-1)
    return np.repeat(frame[None], count, axis=0)


def test_inspection_checks_only_first_three_frames(tmp_path: Path) -> None:
    frames = _textured_frames(4)
    frames[3] = 0
    episode = _episode(tmp_path, "episode_000001", success=True, rgb=frames)

    result = inspect_episode_camera(episode, {"camera_file": "camera_frames.npz"})

    assert result["valid"]
    assert result["inspected_frames"] == 3
    assert all(item["global_std"] >= 5.0 for item in result["frame_metrics"])
    assert all(item["dynamic_range"] >= 20.0 for item in result["frame_metrics"])


def test_recursive_audit_summarizes_sources_levels_and_invalid_paths(
    tmp_path: Path,
) -> None:
    easy = tmp_path / "balanced" / "easy"
    valid_episode = _episode(
        easy, "episode_000001", success=True, rgb=_textured_frames()
    )
    invalid_episode = _episode(
        easy,
        "episode_000002",
        success=True,
        rgb=np.full((3, 8, 10, 3), 127, dtype=np.uint8),
    )
    _episode(easy, "episode_000003", success=False, rgb=_textured_frames())

    report = audit_sources([parse_source_spec(f"easy={easy}")])

    assert report["overall"]["successful_episodes"] == 2
    assert report["overall"]["valid_episodes"] == 1
    assert report["overall"]["invalid_episodes"] == 1
    assert report["by_level"]["easy"]["valid_episodes"] == 1
    assert report["sources"][0]["levels"]["easy"]["invalid_episodes"] == 1
    assert report["invalid_samples"][0]["episode_path"] == str(
        invalid_episode.resolve()
    )
    assert valid_episode.resolve() != invalid_episode.resolve()


def test_direct_source_supports_current_camera_npz_name(tmp_path: Path) -> None:
    medium = tmp_path / "medium"
    _episode(
        medium,
        "episode_000010",
        success=True,
        rgb=_textured_frames(2),
        camera_name="camera.npz",
    )

    report = audit_sources([parse_source_spec(str(medium))])

    assert report["overall"]["valid_episodes"] == 1
    assert report["by_level"]["medium"]["successful_episodes"] == 1
    assert report["invalid_samples"] == []


def test_missing_camera_is_reported_invalid(tmp_path: Path) -> None:
    hard = tmp_path / "hard"
    episode = _episode(hard, "episode_000020", success=True, rgb=None)

    report = audit_sources([parse_source_spec(str(hard))])

    assert report["overall"]["invalid_episodes"] == 1
    assert report["invalid_samples"][0]["episode_path"] == str(episode.resolve())
    assert report["invalid_samples"][0]["reason"] == "missing_camera_npz"


def test_nonfinite_rgb_report_remains_strict_json_serializable(tmp_path: Path) -> None:
    frames = _textured_frames(1).astype(np.float32)
    frames[0, 0, 0, 0] = np.nan
    episode = _episode(tmp_path, "episode_000030", success=True, rgb=frames)

    report = audit_sources([parse_source_spec(str(tmp_path))])

    assert report["invalid_samples"][0]["reason"] == "nonfinite_rgb"
    json.dumps(report, allow_nan=False)
