"""Export balanced Easy/Medium/Hard image-backed LeRobot splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaac_lab_data_engine.vla.dataset import (  # noqa: E402
    ACTION_NAMES,
    DELTA_ACTION_NAMES,
    STATE_NAMES,
    VISUAL_CONDITIONED_STATE_NAMES,
    VISUAL_FSM_CONDITIONED_STATE_NAMES,
    align_episode,
    build_visual_fsm_bounded_actions,
    discover_demonstrations,
)


LEVELS = ("easy", "medium", "hard")
DIFFICULTY_IDS = {"easy": 0.0, "medium": 1.0, "hard": 2.0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for level in LEVELS:
        parser.add_argument(
            f"--{level}-source",
            type=Path,
            action="append",
            required=True,
            help=(
                f"Source directory for {level} episodes. Repeat the option to "
                "combine multiple source directories."
            ),
        )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--episodes-per-level", type=int, default=168)
    parser.add_argument("--train-per-level", type=int, default=134)
    parser.add_argument("--validation-per-level", type=int, default=17)
    parser.add_argument("--test-per-level", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--action-offset-steps",
        type=int,
        default=1,
        help=(
            "Trajectory-action offset from each post-step camera observation. "
            "Use 1 for causal observation(t)->action(t+1) behavior cloning."
        ),
    )
    parser.add_argument(
        "--action-target-mode",
        choices=(
            "recorded_waypoint",
            "visual_fsm_bounded",
            "visual_fsm_bounded_delta",
        ),
        default="recorded_waypoint",
        help=(
            "recorded_waypoint preserves the collected absolute-IK labels; "
            "visual_fsm_bounded rebuilds a phase-aware visual waypoint and "
            "limits each absolute xyz target relative to the current EEF; "
            "visual_fsm_bounded_delta stores the same bounded xyz as a "
            "current-EEF-relative displacement."
        ),
    )
    parser.add_argument(
        "--maximum-position-step-m",
        type=float,
        default=0.04,
        help=(
            "Maximum current-EEF-to-target xyz distance for "
            "visual_fsm_bounded labels (default: 0.04 m)."
        ),
    )
    parser.add_argument(
        "--lift-loss-multiplier",
        type=float,
        default=2.0,
        help=(
            "Frame weight multiplier for phase 4 (LIFT) in delta-mode export. "
            "Stored as training.loss_weight; the trainer must consume it."
        ),
    )
    parser.add_argument(
        "--hard-loss-multiplier",
        type=float,
        default=1.5,
        help=(
            "Frame weight multiplier for Hard samples in delta-mode export. "
            "It is multiplied with the lift weight for Hard/LIFT frames."
        ),
    )
    parser.add_argument(
        "--minimum-rgb-std",
        type=float,
        default=5.0,
        help="Minimum per-frame RGB pixel standard deviation in uint8 intensity units.",
    )
    parser.add_argument(
        "--minimum-rgb-range",
        type=float,
        default=20.0,
        help="Minimum per-frame RGB max-minus-min range in uint8 intensity units.",
    )
    parser.add_argument(
        "--rgb-quality-frames",
        type=int,
        default=3,
        help="Maximum number of leading RGB frames checked in each episode.",
    )
    parser.add_argument("--task", default="Pick up the red cube and lift it 10 centimeters.")
    parser.add_argument(
        "--visual-state-cache",
        type=Path,
        help=(
            "Optional cache from precompute_vla_visual_state_cache.py. Appends "
            "RGB-D target xyz and validity to the 25-D proprioceptive state."
        ),
    )
    return parser.parse_args(argv)


def validate_action_target_configuration(args: argparse.Namespace) -> None:
    """Validate label semantics before creating an output directory."""

    if args.maximum_position_step_m <= 0.0:
        raise ValueError("--maximum-position-step-m must be positive")
    if float(getattr(args, "lift_loss_multiplier", 2.0)) <= 0.0:
        raise ValueError("--lift-loss-multiplier must be positive")
    if float(getattr(args, "hard_loss_multiplier", 1.5)) <= 0.0:
        raise ValueError("--hard-loss-multiplier must be positive")
    if args.action_target_mode in {
        "visual_fsm_bounded",
        "visual_fsm_bounded_delta",
    } and args.visual_state_cache is None:
        raise ValueError(
            f"--action-target-mode {args.action_target_mode} requires "
            "--visual-state-cache"
        )


def action_target_semantics(args: argparse.Namespace) -> dict[str, object]:
    """Return a self-describing action/state contract for export manifests."""

    if args.action_target_mode == "recorded_waypoint":
        return {
            "mode": "recorded_waypoint",
            "position": (
                "trajectory.action xyz at state_trajectory_index + "
                "action_offset_steps; absolute robot-root IK waypoint"
            ),
            "orientation_and_gripper": "copied from the recorded trajectory action",
            "controller_phase_in_state": False,
            "controller_phase_source": None,
            "maximum_position_step_m": None,
            "approach_height_m": None,
            "lift_height_m": None,
        }
    delta_position = args.action_target_mode == "visual_fsm_bounded_delta"
    return {
        "mode": args.action_target_mode,
        "position": (
            "phase waypoint reconstructed from cached visual xyz in robot-root "
            "coordinates, then directionally limited from current EEF; output "
            + (
                "is delta_xyz = bounded_target_xyz - current_eef_xyz"
                if delta_position
                else "remains absolute xyz"
            )
        ),
        "position_target_mode": "delta" if delta_position else "absolute",
        "online_position_decode": (
            "absolute_target_xyz = current_eef_xyz_at_execution + predicted_delta_xyz"
            if delta_position
            else "not required"
        ),
        "controller_phase_in_state": True,
        "controller_phase_state_encoding": "one raw float32 scalar with ids 0..4",
        "controller_phase_source": (
            "trajectory.controller_state[state_trajectory_indices]"
        ),
        "phase_waypoints": {
            "REST_0": "current EEF xyz",
            "APPROACH_ABOVE_OBJECT_1": "visual xyz + [0, 0, 0.12] m",
            "APPROACH_OBJECT_2": "visual xyz",
            "GRASP_OBJECT_3": "visual xyz",
            "LIFT_OBJECT_4": "visual xyz + [0, 0, 0.18] m",
        },
        "maximum_position_step_m": float(args.maximum_position_step_m),
        "orientation": (
            "REST preserves current EEF wxyz; phases 1..4 use [0, 1, 0, 0]"
        ),
        "gripper": "phases 0..2 use +1 open; phases 3..4 use -1 closed",
        "approach_height_m": 0.12,
        "lift_height_m": 0.18,
    }


def build_training_metadata(
    level: str,
    controller_phase: int,
    *,
    lift_loss_multiplier: float = 2.0,
    hard_loss_multiplier: float = 1.5,
) -> dict[str, np.ndarray]:
    """Build non-policy-input metadata for weighted delta-action training."""

    if level not in DIFFICULTY_IDS:
        raise ValueError(f"Unknown difficulty level: {level!r}")
    phase = int(controller_phase)
    if phase < 0 or phase > 4:
        raise ValueError("controller_phase must be within [0, 4]")
    if lift_loss_multiplier <= 0.0 or hard_loss_multiplier <= 0.0:
        raise ValueError("Training loss multipliers must be positive")
    loss_weight = 1.0
    if phase == 4:
        loss_weight *= float(lift_loss_multiplier)
    if level == "hard":
        loss_weight *= float(hard_loss_multiplier)
    return {
        "training.controller_phase": np.asarray([phase], dtype=np.float32),
        "training.difficulty_id": np.asarray(
            [DIFFICULTY_IDS[level]], dtype=np.float32
        ),
        "training.loss_weight": np.asarray([loss_weight], dtype=np.float32),
    }


def _source_list(value: Path | str | list[Path] | tuple[Path, ...]) -> list[Path]:
    """Normalize both legacy single-source and repeatable CLI values."""

    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(path) for path in value]


def inspect_episode_rgb_quality(
    episode_dir: Path,
    *,
    minimum_rgb_std: float,
    minimum_rgb_range: float,
    maximum_frames: int,
) -> dict[str, object]:
    """Check the leading RGB frames without loading a whole trajectory.

    Every inspected frame must satisfy both thresholds. This rejects blank or
    partially initialized captures instead of letting later frames hide them.
    """

    if minimum_rgb_std < 0.0 or minimum_rgb_range < 0.0:
        raise ValueError("RGB quality thresholds must be non-negative")
    if maximum_frames < 1:
        raise ValueError("--rgb-quality-frames must be at least 1")

    try:
        with np.load(episode_dir / "camera.npz") as camera:
            rgb = np.asarray(camera["rgb"])
    except (KeyError, OSError, ValueError) as error:
        return {
            "passed": False,
            "reason": f"camera_read_error:{type(error).__name__}",
            "frames_checked": 0,
        }

    if rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) == 0:
        return {
            "passed": False,
            "reason": "invalid_rgb_shape",
            "frames_checked": 0,
            "rgb_shape": list(rgb.shape),
        }

    frame_count = min(maximum_frames, len(rgb))
    flattened = rgb[:frame_count].astype(np.float32, copy=False).reshape(frame_count, -1)
    frame_std = flattened.std(axis=1)
    frame_range = np.ptp(flattened, axis=1)
    finite = np.isfinite(frame_std).all() and np.isfinite(frame_range).all()
    passed = bool(
        finite
        and np.all(frame_std >= minimum_rgb_std)
        and np.all(frame_range >= minimum_rgb_range)
    )
    return {
        "passed": passed,
        "reason": "passed" if passed else "below_rgb_quality_threshold",
        "frames_checked": frame_count,
        "minimum_observed_rgb_std": float(frame_std.min()),
        "minimum_observed_rgb_range": float(frame_range.min()),
    }


def discover_quality_demonstrations(
    sources: Path | str | list[Path] | tuple[Path, ...],
    *,
    minimum_rgb_std: float,
    minimum_rgb_range: float,
    maximum_frames: int,
) -> tuple[list[Path], dict[str, object]]:
    """Return successful, unique episodes that pass the RGB quality gate."""

    source_paths = _source_list(sources)
    discovered = discover_demonstrations(source_paths, successes_only=True)
    candidates = sorted(set(discovered))
    qualified: list[Path] = []
    rejection_reasons: dict[str, int] = {}
    for episode_dir in candidates:
        quality = inspect_episode_rgb_quality(
            episode_dir,
            minimum_rgb_std=minimum_rgb_std,
            minimum_rgb_range=minimum_rgb_range,
            maximum_frames=maximum_frames,
        )
        if bool(quality["passed"]):
            qualified.append(episode_dir)
            continue
        reason = str(quality["reason"])
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    counts: dict[str, object] = {
        "sources": [str(path.expanduser().resolve()) for path in source_paths],
        "available_successes": len(candidates),
        "candidate_successes": len(candidates),
        "quality_passed": len(qualified),
        "quality_filtered": len(candidates) - len(qualified),
        "quality_rejection_reasons": rejection_reasons,
    }
    return qualified, counts


def select_splits(args: argparse.Namespace):
    requested = args.train_per_level + args.validation_per_level + args.test_per_level
    if requested != args.episodes_per_level:
        raise ValueError(
            "train/validation/test per-level counts must sum to --episodes-per-level"
        )
    rng = np.random.default_rng(args.seed)
    split_map = {"train": [], "validation": [], "test": []}
    source_map = {
        "easy": args.easy_source,
        "medium": args.medium_source,
        "hard": args.hard_source,
    }
    level_counts = {}
    for level in LEVELS:
        episodes, counts = discover_quality_demonstrations(
            source_map[level],
            minimum_rgb_std=args.minimum_rgb_std,
            minimum_rgb_range=args.minimum_rgb_range,
            maximum_frames=args.rgb_quality_frames,
        )
        if len(episodes) < args.episodes_per_level:
            raise RuntimeError(
                f"{level} has {len(episodes)} successful RGB-qualified episodes; "
                f"need {args.episodes_per_level}"
            )
        order = rng.permutation(len(episodes))[: args.episodes_per_level]
        selected = [episodes[int(index)] for index in order]
        train_end = args.train_per_level
        validation_end = train_end + args.validation_per_level
        split_map["train"].extend((level, path) for path in selected[:train_end])
        split_map["validation"].extend(
            (level, path) for path in selected[train_end:validation_end]
        )
        split_map["test"].extend((level, path) for path in selected[validation_end:])
        level_counts[level] = counts | {"selected": len(selected)}
    return split_map, level_counts


def export_split(args: argparse.Namespace, split: str, records):
    validate_action_target_configuration(args)
    output = args.output_root / split
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    aligned_records = [
        (
            level,
            align_episode(
                path,
                task=args.task,
                action_offset_steps=args.action_offset_steps,
            ),
        )
        for level, path in records
    ]
    shapes = sorted({tuple(episode.rgb.shape[1:]) for _, episode in aligned_records})
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent RGB shapes in {split}: {shapes}")
    height, width, channels = shapes[0]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"{args.repo_prefix}-{split}"
    phase_conditioned_mode = args.action_target_mode in {
        "visual_fsm_bounded",
        "visual_fsm_bounded_delta",
    }
    if phase_conditioned_mode:
        state_names = VISUAL_FSM_CONDITIONED_STATE_NAMES
    elif args.visual_state_cache:
        state_names = VISUAL_CONDITIONED_STATE_NAMES
    else:
        state_names = STATE_NAMES
    action_names = (
        DELTA_ACTION_NAMES
        if args.action_target_mode == "visual_fsm_bounded_delta"
        else ACTION_NAMES
    )
    weighted_delta_mode = args.action_target_mode == "visual_fsm_bounded_delta"
    features = {
        "observation.images.table": {
            "dtype": "image",
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": list(state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": list(action_names),
        },
    }
    if weighted_delta_mode:
        features.update(
            {
                "training.controller_phase": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["controller_phase"],
                },
                "training.difficulty_id": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["difficulty_id"],
                },
                "training.loss_weight": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["loss_weight"],
                },
            }
        )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=args.fps,
        root=output,
        robot_type="franka_isaac_lab_absolute_ik",
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )
    manifest_records = []
    controller_phase_frame_counts = {str(phase): 0 for phase in range(5)}
    try:
        for episode_index, (level, episode) in enumerate(aligned_records):
            episode_state = episode.state
            episode_action = episode.action
            controller_phase = None
            if args.visual_state_cache:
                cache_path = (
                    args.visual_state_cache
                    / split
                    / f"episode_{episode_index:06d}.npz"
                )
                if not cache_path.is_file():
                    raise FileNotFoundError(cache_path)
                with np.load(cache_path) as cache:
                    cached_source = str(np.asarray(cache["source"]).item())
                    visual_position = np.asarray(
                        cache["visual_position_b"], dtype=np.float32
                    )
                    visual_validity = np.asarray(cache["validity"], dtype=np.float32)
                if Path(cached_source).resolve() != episode.episode_dir.resolve():
                    raise ValueError(
                        f"Visual cache source mismatch: {cache_path} names "
                        f"{cached_source}, expected {episode.episode_dir}"
                    )
                if visual_position.shape != (episode.frames, 3):
                    raise ValueError(
                        f"Expected visual position [{episode.frames},3] in "
                        f"{cache_path}, got {visual_position.shape}"
                    )
                if visual_validity.shape == (episode.frames,):
                    visual_validity = visual_validity[:, None]
                if visual_validity.shape != (episode.frames, 1):
                    raise ValueError(
                        f"Expected validity [{episode.frames},1] in {cache_path}, "
                        f"got {visual_validity.shape}"
                    )
                episode_state = np.concatenate(
                    (episode.state, visual_position, visual_validity), axis=-1
                ).astype(np.float32, copy=False)
                if phase_conditioned_mode:
                    with np.load(
                        episode.episode_dir / "trajectory.npz", allow_pickle=False
                    ) as trajectory:
                        controller_phase = np.asarray(
                            trajectory["controller_state"], dtype=np.int64
                        )[episode.state_trajectory_indices]
                    if controller_phase.shape != (episode.frames,):
                        raise ValueError(
                            f"Expected controller phase [{episode.frames}] for "
                            f"{episode.episode_dir}, got {controller_phase.shape}"
                        )
                    episode_state = np.concatenate(
                        (episode_state, controller_phase[:, None].astype(np.float32)),
                        axis=-1,
                    ).astype(np.float32, copy=False)
                    episode_action = build_visual_fsm_bounded_actions(
                        episode.state[:, 18:25],
                        visual_position,
                        controller_phase,
                        maximum_position_step_m=args.maximum_position_step_m,
                        approach_height_m=0.12,
                        lift_height_m=0.18,
                        position_target_mode=(
                            "delta"
                            if args.action_target_mode == "visual_fsm_bounded_delta"
                            else "absolute"
                        ),
                    )
                    for phase in range(5):
                        controller_phase_frame_counts[str(phase)] += int(
                            np.count_nonzero(controller_phase == phase)
                        )
            for index in range(episode.frames):
                frame = {
                    "observation.images.table": episode.rgb[index],
                    "observation.state": episode_state[index],
                    "action": episode_action[index],
                    "task": episode.task,
                }
                if weighted_delta_mode:
                    frame.update(
                        build_training_metadata(
                            level,
                            int(controller_phase[index]),
                            lift_loss_multiplier=float(args.lift_loss_multiplier),
                            hard_loss_multiplier=float(args.hard_loss_multiplier),
                        )
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
            manifest_records.append(
                {
                    "level": level,
                    "source": str(episode.episode_dir),
                    "frames": episode.frames,
                }
            )
    finally:
        dataset.finalize()

    level_episode_counts = {
        level: sum(record["level"] == level for record in manifest_records)
        for level in LEVELS
    }
    level_frame_counts = {
        level: sum(
            int(record["frames"])
            for record in manifest_records
            if record["level"] == level
        )
        for level in LEVELS
    }
    summary = {
        "split": split,
        "repo_id": repo_id,
        "output": str(output.resolve()),
        "episodes": len(manifest_records),
        "frames": sum(int(record["frames"]) for record in manifest_records),
        "level_episode_counts": level_episode_counts,
        "level_frame_counts": level_frame_counts,
        "rgb_shape": [height, width, channels],
        "state_dim": len(state_names),
        "action_dim": len(action_names),
        "action_names": list(action_names),
        "privileged_object_state": False,
        "visual_pose_conditioning": bool(args.visual_state_cache),
        "visual_state_cache": (
            str(args.visual_state_cache.resolve()) if args.visual_state_cache else None
        ),
        "action_offset_steps": args.action_offset_steps,
        "action_target_mode": args.action_target_mode,
        "action_target_semantics": action_target_semantics(args),
        "controller_phase_frame_counts": (
            controller_phase_frame_counts
            if phase_conditioned_mode
            else None
        ),
        "training_metadata": (
            {
                "policy_input": False,
                "features": [
                    "training.controller_phase",
                    "training.difficulty_id",
                    "training.loss_weight",
                ],
                "difficulty_ids": DIFFICULTY_IDS,
                "base_loss_weight": 1.0,
                "lift_phase": 4,
                "lift_loss_multiplier": float(args.lift_loss_multiplier),
                "hard_loss_multiplier": float(args.hard_loss_multiplier),
                "combination": "multiplicative",
                "trainer_must_consume_loss_weight": True,
            }
            if weighted_delta_mode
            else None
        ),
        "records": manifest_records,
    }
    (output / "balanced_split_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BALANCED_SPLIT=" + json.dumps(summary | {"records": "omitted"}, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    validate_action_target_configuration(args)
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    split_map, level_counts = select_splits(args)
    summaries = {
        split: export_split(args, split, split_map[split])
        for split in ("train", "validation", "test")
    }
    source_sets = {
        split: {record["source"] for record in summary["records"]}
        for split, summary in summaries.items()
    }
    overlaps = {
        "train_validation": len(source_sets["train"] & source_sets["validation"]),
        "train_test": len(source_sets["train"] & source_sets["test"]),
        "validation_test": len(source_sets["validation"] & source_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Episode leakage detected: {overlaps}")
    top_summary = {
        "status": "BALANCED_VLA_EXPORT_OK",
        "seed": args.seed,
        "action_offset_steps": args.action_offset_steps,
        "action_target_mode": args.action_target_mode,
        "action_target_semantics": action_target_semantics(args),
        "episodes_per_level": args.episodes_per_level,
        "total_selected_episodes": args.episodes_per_level * len(LEVELS),
        "visual_quality_gate": {
            "minimum_rgb_std": args.minimum_rgb_std,
            "minimum_rgb_range": args.minimum_rgb_range,
            "maximum_leading_frames_checked": args.rgb_quality_frames,
            "all_checked_frames_must_pass": True,
            "candidate_successes": sum(
                int(counts["candidate_successes"]) for counts in level_counts.values()
            ),
            "quality_filtered": sum(
                int(counts["quality_filtered"]) for counts in level_counts.values()
            ),
        },
        "source_counts": level_counts,
        "split_counts": {
            split: {
                "episodes": summary["episodes"],
                "frames": summary["frames"],
                "levels": summary["level_episode_counts"],
            }
            for split, summary in summaries.items()
        },
        "episode_overlap": overlaps,
    }
    (args.output_root / "balanced_export_summary.json").write_text(
        json.dumps(top_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BALANCED_VLA_EXPORT=" + json.dumps(top_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
