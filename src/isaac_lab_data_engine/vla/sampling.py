"""Deterministic frame-weight plans for VLA training.

The stock LeRobot 0.4.4 training CLI does not expose a configurable
``WeightedRandomSampler``.  This module keeps the sampling policy outside the
dataset so validation/test data remain untouched, while giving the Windows
training launcher a small, strictly validated artifact to consume.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


LEVELS = ("easy", "medium", "hard")
CONTROLLER_PHASES = (0, 1, 2, 3, 4)
CONTROLLER_PHASE_NAMES = {
    0: "rest",
    1: "approach_above_object",
    2: "approach_object",
    3: "grasp_object",
    4: "lift_object",
}

# Multiplicative defaults deliberately favor Hard and the two contact/lift
# stages without removing coverage of the approach stages.
DEFAULT_LEVEL_MULTIPLIERS = {"easy": 1.0, "medium": 1.0, "hard": 2.0}
DEFAULT_PHASE_MULTIPLIERS = {0: 1.0, 1: 1.0, 2: 1.25, 3: 1.5, 4: 3.0}


def _validate_multiplier_map(
    values: Mapping[Any, float],
    expected_keys: Sequence[Any],
    *,
    name: str,
) -> dict[Any, float]:
    if set(values) != set(expected_keys):
        raise ValueError(
            f"{name} keys must be exactly {list(expected_keys)}, got {list(values)}"
        )
    normalized = {key: float(values[key]) for key in expected_keys}
    if not all(np.isfinite(value) and value > 0.0 for value in normalized.values()):
        raise ValueError(f"{name} values must all be finite and positive")
    return normalized


def build_frame_sampling_weights(
    levels: Sequence[str] | np.ndarray,
    controller_phases: Sequence[int] | np.ndarray,
    *,
    level_multipliers: Mapping[str, float] | None = None,
    phase_multipliers: Mapping[int, float] | None = None,
    normalize_mean: bool = True,
) -> np.ndarray:
    """Build one multiplicative sampling weight per original dataset frame.

    A Hard/LIFT frame receives both multipliers.  The returned weights are
    normalized to mean one by default; this does not alter sampling
    probabilities but keeps diagnostics and downstream serialization stable.
    """

    level_array = np.asarray(levels, dtype=object).reshape(-1)
    phase_array = np.asarray(controller_phases).reshape(-1)
    if level_array.shape != phase_array.shape:
        raise ValueError(
            "levels and controller_phases must have the same number of frames"
        )
    if len(level_array) == 0:
        raise ValueError("at least one frame is required")

    if not np.issubdtype(phase_array.dtype, np.integer):
        if not np.isfinite(phase_array.astype(np.float64)).all() or not np.allclose(
            phase_array, np.rint(phase_array), atol=0.0, rtol=0.0
        ):
            raise ValueError("controller phases must be integer-valued")
    phase_array = phase_array.astype(np.int64, copy=False)

    unknown_levels = sorted(set(str(value) for value in level_array) - set(LEVELS))
    if unknown_levels:
        raise ValueError(f"unknown difficulty levels: {unknown_levels}")
    unknown_phases = sorted(set(int(value) for value in phase_array) - set(CONTROLLER_PHASES))
    if unknown_phases:
        raise ValueError(f"unknown controller phases: {unknown_phases}")

    level_weights = _validate_multiplier_map(
        level_multipliers or DEFAULT_LEVEL_MULTIPLIERS,
        LEVELS,
        name="level_multipliers",
    )
    phase_weights = _validate_multiplier_map(
        phase_multipliers or DEFAULT_PHASE_MULTIPLIERS,
        CONTROLLER_PHASES,
        name="phase_multipliers",
    )
    weights = np.asarray(
        [
            level_weights[str(level)] * phase_weights[int(phase)]
            for level, phase in zip(level_array, phase_array, strict=True)
        ],
        dtype=np.float64,
    )
    if normalize_mean:
        weights /= weights.mean()
    return weights


def deterministic_sampling_preview(
    weights: Sequence[float] | np.ndarray,
    *,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Draw a deterministic audit preview from the planned distribution.

    This preview is diagnostic.  Runtime training should use the serialized
    weights, seed, replacement flag, and sample count with PyTorch's
    ``WeightedRandomSampler``.
    """

    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    _validate_weights(weights_array)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    probabilities = weights_array / weights_array.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(
        len(weights_array), size=int(num_samples), replace=True, p=probabilities
    ).astype(np.int64, copy=False)


def make_weighted_random_sampler(
    weights: Sequence[float] | np.ndarray,
    *,
    num_samples: int,
    seed: int,
):
    """Create the intended PyTorch sampler with an isolated fixed generator."""

    weights_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    _validate_weights(weights_array)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights_array.copy()),
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


class FrameLossWeightDataset:
    """Transparent dataset proxy that injects one scalar loss weight per frame.

    The proxy does not alter dataset length or frame identity, so it can be used
    together with a WeightedRandomSampler over the same weight vector. A
    separate, milder loss-weight vector is recommended when both mechanisms are
    enabled, because using identical weights would square the effective bias.
    """

    def __init__(
        self,
        dataset: Any,
        loss_weights: Sequence[float] | np.ndarray,
        *,
        key: str = "training.loss_weight",
    ) -> None:
        if not key:
            raise ValueError("loss-weight key must not be empty")
        values = np.asarray(loss_weights, dtype=np.float32).reshape(-1)
        _validate_weights(values)
        if len(values) != len(dataset):
            raise ValueError(
                "loss-weight dataset length mismatch: "
                f"weights={len(values)}, dataset={len(dataset)}"
            )
        self.dataset = dataset
        self.loss_weights = values
        self.loss_weight_key = key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dataset item dict, got {type(sample).__name__}")
        result = dict(sample)
        result[self.loss_weight_key] = self.loss_weights[int(index)]
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _group_distribution(
    levels: np.ndarray,
    phases: np.ndarray,
    masses: np.ndarray,
) -> dict[str, object]:
    total = float(masses.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("distribution mass must be finite and positive")

    def item(mask: np.ndarray) -> dict[str, float | int]:
        mass = float(masses[mask].sum())
        return {"mass": mass, "fraction": mass / total}

    by_level = {level: item(levels == level) for level in LEVELS}
    by_phase = {
        str(phase): {
            "name": CONTROLLER_PHASE_NAMES[phase],
            **item(phases == phase),
        }
        for phase in CONTROLLER_PHASES
    }
    by_joint = {
        level: {
            str(phase): item((levels == level) & (phases == phase))
            for phase in CONTROLLER_PHASES
        }
        for level in LEVELS
    }
    return {"by_level": by_level, "by_phase": by_phase, "by_level_and_phase": by_joint}


def summarize_sampling_plan(
    levels: Sequence[str] | np.ndarray,
    controller_phases: Sequence[int] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    *,
    num_samples: int,
    preview_indices: Sequence[int] | np.ndarray | None = None,
) -> dict[str, object]:
    """Summarize source, expected, and optional deterministic preview mixes."""

    level_array = np.asarray(levels, dtype=str).reshape(-1)
    phase_array = np.asarray(controller_phases, dtype=np.int64).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (len(level_array) == len(phase_array) == len(weight_array)):
        raise ValueError("levels, phases, and weights must have equal lengths")
    _validate_weights(weight_array)
    if num_samples < 1:
        raise ValueError("num_samples must be positive")

    source = _group_distribution(
        level_array, phase_array, np.ones(len(weight_array), dtype=np.float64)
    )
    expected = _group_distribution(level_array, phase_array, weight_array)

    preview = None
    if preview_indices is not None:
        indices = np.asarray(preview_indices, dtype=np.int64).reshape(-1)
        if len(indices) != num_samples:
            raise ValueError("preview_indices length must equal num_samples")
        if np.any((indices < 0) | (indices >= len(weight_array))):
            raise ValueError("preview_indices contain an out-of-range frame index")
        preview = _group_distribution(
            level_array[indices],
            phase_array[indices],
            np.ones(len(indices), dtype=np.float64),
        )

    effective_sample_size = float(weight_array.sum() ** 2 / np.square(weight_array).sum())
    return {
        "num_source_frames": len(weight_array),
        "num_samples_per_sampler_epoch": int(num_samples),
        "replacement": True,
        "weight_min": float(weight_array.min()),
        "weight_max": float(weight_array.max()),
        "effective_sample_size": effective_sample_size,
        "source_distribution": source,
        "expected_sampled_distribution": expected,
        "deterministic_numpy_preview_distribution": preview,
    }


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_weights(weights: np.ndarray) -> None:
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("sampling weights must be a non-empty one-dimensional array")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("sampling weights must all be finite and positive")


def save_sampling_plan(
    plan_path: Path,
    weights: Sequence[float] | np.ndarray,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Write a JSON contract plus a compact NumPy weight vector."""

    plan_path = Path(plan_path)
    if plan_path.exists():
        raise FileExistsError(f"Refusing to overwrite {plan_path}")
    weights_path = plan_path.with_suffix(".weights.npy")
    if weights_path.exists():
        raise FileExistsError(f"Refusing to overwrite {weights_path}")

    weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
    _validate_weights(weight_array)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(weights_path, weight_array, allow_pickle=False)

    payload = {
        "format_version": 1,
        "strategy": "torch_weighted_random_sampler",
        "weights_file": weights_path.name,
        "weights_dtype": "float64",
        "num_frames": len(weight_array),
        "weights_sha256": array_sha256(weight_array),
        **dict(metadata),
    }
    plan_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def load_sampling_plan(
    plan_path: Path,
    *,
    dataset_length: int,
    dataset_manifest_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load and strictly verify a frame-sampling plan before training."""

    plan_path = Path(plan_path)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported sampling-plan version: {payload.get('format_version')}")
    if payload.get("strategy") != "torch_weighted_random_sampler":
        raise ValueError(f"Unsupported sampling strategy: {payload.get('strategy')}")
    if int(payload.get("num_frames", -1)) != int(dataset_length):
        raise ValueError(
            "Sampling-plan dataset length mismatch: "
            f"plan={payload.get('num_frames')}, dataset={dataset_length}"
        )

    weights_path = plan_path.parent / str(payload["weights_file"])
    weights = np.load(weights_path, allow_pickle=False)
    if weights.dtype != np.float64:
        raise ValueError(f"Sampling weights must be float64, got {weights.dtype}")
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    _validate_weights(weights)
    if len(weights) != dataset_length:
        raise ValueError(
            f"Sampling weight count mismatch: weights={len(weights)}, dataset={dataset_length}"
        )
    actual_hash = array_sha256(weights)
    if actual_hash != payload.get("weights_sha256"):
        raise ValueError("Sampling weight SHA-256 mismatch")

    expected_manifest_hash = payload.get("dataset_manifest_sha256")
    if dataset_manifest_path is not None and expected_manifest_hash is not None:
        actual_manifest_hash = file_sha256(Path(dataset_manifest_path))
        if actual_manifest_hash != expected_manifest_hash:
            raise ValueError("Sampling-plan dataset manifest SHA-256 mismatch")
    return weights, payload
