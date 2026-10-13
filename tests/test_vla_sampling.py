from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from isaac_lab_data_engine.vla.sampling import (
    FrameLossWeightDataset,
    build_frame_sampling_weights,
    deterministic_sampling_preview,
    load_sampling_plan,
    make_weighted_random_sampler,
    save_sampling_plan,
    summarize_sampling_plan,
)


def test_weights_multiply_hard_and_lift_priorities() -> None:
    levels = np.asarray(["easy", "hard", "hard", "medium"])
    phases = np.asarray([0, 0, 4, 3])
    weights = build_frame_sampling_weights(levels, phases, normalize_mean=False)

    np.testing.assert_allclose(weights, [1.0, 2.0, 6.0, 1.5])


def test_weighted_distribution_boosts_hard_and_lift() -> None:
    levels = np.asarray(["easy", "medium", "hard"] * 5)
    phases = np.tile(np.arange(5), 3)
    weights = build_frame_sampling_weights(levels, phases)
    preview = deterministic_sampling_preview(weights, num_samples=20_000, seed=7)
    summary = summarize_sampling_plan(
        levels, phases, weights, num_samples=len(preview), preview_indices=preview
    )

    source = summary["source_distribution"]
    expected = summary["expected_sampled_distribution"]
    actual = summary["deterministic_numpy_preview_distribution"]
    assert expected["by_level"]["hard"]["fraction"] > source["by_level"]["hard"]["fraction"]
    assert expected["by_phase"]["4"]["fraction"] > source["by_phase"]["4"]["fraction"]
    assert actual["by_level"]["hard"]["fraction"] == pytest.approx(
        expected["by_level"]["hard"]["fraction"], abs=0.015
    )
    assert actual["by_phase"]["4"]["fraction"] == pytest.approx(
        expected["by_phase"]["4"]["fraction"], abs=0.015
    )


def test_invalid_levels_phases_and_multipliers_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown difficulty"):
        build_frame_sampling_weights(["extreme"], [0])
    with pytest.raises(ValueError, match="unknown controller"):
        build_frame_sampling_weights(["hard"], [9])
    with pytest.raises(ValueError, match="finite and positive"):
        build_frame_sampling_weights(
            ["hard"],
            [4],
            level_multipliers={"easy": 1.0, "medium": 1.0, "hard": 0.0},
        )


def test_plan_round_trip_checks_length_hash_and_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "balanced_split_manifest.json"
    manifest.write_text('{"split":"train"}', encoding="utf-8")
    weights = np.asarray([0.5, 1.0, 2.0], dtype=np.float64)
    import hashlib

    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    plan = tmp_path / "sampling_plan.json"
    save_sampling_plan(
        plan,
        weights,
        {
            "dataset_manifest_sha256": manifest_hash,
            "seed": 123,
            "num_samples": 3,
            "replacement": True,
        },
    )

    loaded, payload = load_sampling_plan(
        plan, dataset_length=3, dataset_manifest_path=manifest
    )
    np.testing.assert_array_equal(loaded, weights)
    assert payload["seed"] == 123

    with pytest.raises(ValueError, match="length mismatch"):
        load_sampling_plan(plan, dataset_length=4)

    manifest.write_text('{"split":"validation"}', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        load_sampling_plan(plan, dataset_length=3, dataset_manifest_path=manifest)


def test_plan_refuses_overwrite_and_detects_weight_tampering(tmp_path: Path) -> None:
    plan = tmp_path / "sampling_plan.json"
    save_sampling_plan(plan, [1.0, 2.0], {"seed": 1, "num_samples": 2})
    with pytest.raises(FileExistsError):
        save_sampling_plan(plan, [1.0, 2.0], {"seed": 1, "num_samples": 2})

    weights_path = tmp_path / "sampling_plan.weights.npy"
    np.save(weights_path, np.asarray([2.0, 1.0], dtype=np.float64), allow_pickle=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_sampling_plan(plan, dataset_length=2)


def test_torch_sampler_is_reproducible_and_loss_proxy_is_transparent() -> None:
    weights = np.asarray([1.0, 2.0, 6.0], dtype=np.float64)
    first = list(make_weighted_random_sampler(weights, num_samples=100, seed=41))
    second = list(make_weighted_random_sampler(weights, num_samples=100, seed=41))
    assert first == second
    assert first.count(2) > first.count(0)

    class TinyDataset:
        num_frames = 3
        meta = "metadata"

        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict[str, int]:
            return {"frame": index}

    wrapped = FrameLossWeightDataset(TinyDataset(), [0.5, 1.0, 1.5])
    assert len(wrapped) == 3
    assert wrapped.num_frames == 3
    assert wrapped.meta == "metadata"
    assert wrapped[2]["frame"] == 2
    assert wrapped[2]["training.loss_weight"] == pytest.approx(1.5)

    with pytest.raises(ValueError, match="length mismatch"):
        FrameLossWeightDataset(TinyDataset(), [1.0, 2.0])
