from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.export_balanced_lerobot_vla_images_v044 import (
    action_target_semantics,
    build_training_metadata,
    inspect_episode_rgb_quality,
    parse_args,
    select_splits,
    validate_action_target_configuration,
)


def _checkerboard_rgb(frames: int = 4) -> np.ndarray:
    pattern = (np.indices((8, 8)).sum(axis=0) % 2 * 255).astype(np.uint8)
    frame = np.repeat(pattern[..., None], 3, axis=-1)
    return np.repeat(frame[None, ...], frames, axis=0)


def _write_episode(
    root: Path,
    episode_id: int,
    rgb: np.ndarray,
    *,
    success: bool = True,
) -> Path:
    episode_dir = root / f"episode_{episode_id:06d}"
    episode_dir.mkdir(parents=True)
    (episode_dir / "metadata.json").write_text(
        json.dumps({"success": success, "camera_frames": len(rgb)}),
        encoding="utf-8",
    )
    np.savez(episode_dir / "trajectory.npz", placeholder=np.zeros(1, dtype=np.float32))
    np.savez(episode_dir / "camera.npz", rgb=rgb)
    return episode_dir


def test_parse_args_accepts_repeated_and_single_sources() -> None:
    args = parse_args(
        [
            "--easy-source",
            "easy_a",
            "--easy-source",
            "easy_b",
            "--medium-source",
            "medium",
            "--hard-source",
            "hard",
            "--output-root",
            "output",
            "--repo-prefix",
            "local/test",
        ]
    )

    assert args.easy_source == [Path("easy_a"), Path("easy_b")]
    assert args.medium_source == [Path("medium")]
    assert args.hard_source == [Path("hard")]
    assert args.minimum_rgb_std == 5.0
    assert args.minimum_rgb_range == 20.0
    assert args.rgb_quality_frames == 3
    assert args.action_target_mode == "recorded_waypoint"
    assert args.maximum_position_step_m == 0.04
    assert args.lift_loss_multiplier == 2.0
    assert args.hard_loss_multiplier == 1.5
    validate_action_target_configuration(args)
    assert action_target_semantics(args)["mode"] == "recorded_waypoint"


def test_visual_fsm_mode_requires_cache_and_describes_semantics() -> None:
    args = Namespace(
        action_target_mode="visual_fsm_bounded",
        maximum_position_step_m=0.04,
        visual_state_cache=None,
    )
    with pytest.raises(ValueError, match="requires --visual-state-cache"):
        validate_action_target_configuration(args)

    args.visual_state_cache = Path("visual-cache")
    validate_action_target_configuration(args)
    semantics = action_target_semantics(args)
    assert semantics["controller_phase_source"] == (
        "trajectory.controller_state[state_trajectory_indices]"
    )
    assert semantics["maximum_position_step_m"] == 0.04
    assert semantics["approach_height_m"] == 0.12
    assert semantics["lift_height_m"] == 0.18


def test_visual_fsm_delta_mode_describes_online_decode() -> None:
    args = Namespace(
        action_target_mode="visual_fsm_bounded_delta",
        maximum_position_step_m=0.04,
        visual_state_cache=Path("visual-cache"),
    )

    validate_action_target_configuration(args)
    semantics = action_target_semantics(args)

    assert semantics["mode"] == "visual_fsm_bounded_delta"
    assert semantics["position_target_mode"] == "delta"
    assert "current_eef_xyz_at_execution" in semantics["online_position_decode"]


def test_delta_training_metadata_multiplies_lift_and_hard_weights() -> None:
    easy_approach = build_training_metadata("easy", 1)
    easy_lift = build_training_metadata("easy", 4)
    hard_approach = build_training_metadata("hard", 1)
    hard_lift = build_training_metadata("hard", 4)

    np.testing.assert_array_equal(easy_approach["training.difficulty_id"], [0.0])
    np.testing.assert_array_equal(hard_lift["training.difficulty_id"], [2.0])
    np.testing.assert_array_equal(hard_lift["training.controller_phase"], [4.0])
    assert float(easy_approach["training.loss_weight"][0]) == 1.0
    assert float(easy_lift["training.loss_weight"][0]) == 2.0
    assert float(hard_approach["training.loss_weight"][0]) == 1.5
    assert float(hard_lift["training.loss_weight"][0]) == 3.0


def test_rgb_quality_gate_checks_only_requested_leading_frames(tmp_path: Path) -> None:
    rgb = _checkerboard_rgb()
    rgb[3].fill(127)
    episode_dir = _write_episode(tmp_path, 0, rgb)

    first_three = inspect_episode_rgb_quality(
        episode_dir,
        minimum_rgb_std=5.0,
        minimum_rgb_range=20.0,
        maximum_frames=3,
    )
    first_four = inspect_episode_rgb_quality(
        episode_dir,
        minimum_rgb_std=5.0,
        minimum_rgb_range=20.0,
        maximum_frames=4,
    )

    assert first_three["passed"] is True
    assert first_three["frames_checked"] == 3
    assert first_four["passed"] is False
    assert first_four["frames_checked"] == 4


def test_select_splits_filters_bad_rgb_across_multiple_sources(tmp_path: Path) -> None:
    source_map: dict[str, list[Path]] = {}
    for level in ("easy", "medium", "hard"):
        good_root = tmp_path / f"{level}_good"
        mixed_root = tmp_path / f"{level}_mixed"
        _write_episode(good_root, 0, _checkerboard_rgb())
        _write_episode(mixed_root, 0, np.full((4, 8, 8, 3), 127, dtype=np.uint8))
        _write_episode(mixed_root, 1, _checkerboard_rgb(), success=False)
        source_map[level] = [good_root, mixed_root]

    args = Namespace(
        easy_source=source_map["easy"],
        medium_source=source_map["medium"],
        hard_source=source_map["hard"],
        train_per_level=1,
        validation_per_level=0,
        test_per_level=0,
        episodes_per_level=1,
        seed=123,
        minimum_rgb_std=5.0,
        minimum_rgb_range=20.0,
        rgb_quality_frames=3,
    )

    splits, counts = select_splits(args)

    assert len(splits["train"]) == 3
    assert splits["validation"] == []
    assert splits["test"] == []
    for level in ("easy", "medium", "hard"):
        assert counts[level]["candidate_successes"] == 2
        assert counts[level]["quality_passed"] == 1
        assert counts[level]["quality_filtered"] == 1
        assert counts[level]["selected"] == 1
        assert len(counts[level]["sources"]) == 2
