"""CPU-only tests for deterministic, resumable SmolVLA checkpoint scans."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.scan_smolvla_checkpoints import (
    atomic_write_json,
    build_report,
    deterministic_action_noise,
    discover_peft_checkpoints,
    evaluate_checkpoint,
    is_cuda_oom,
    iter_index_batches,
    make_scan_config,
    validate_resume_report,
)
from isaac_lab_data_engine.vla.checkpoint import (
    checkpoint_chunk_size,
    ensure_policy_config_file,
    read_policy_config_payload,
)


def _sample(index: int) -> dict[str, object]:
    state = np.linspace(0.0, 1.0, 25, dtype=np.float32)
    state[0] = float(index)
    state[18:21] = [0.45 + index * 0.001, -0.02, 0.12]
    action = np.asarray(
        [0.50, 0.01, 0.10, 1.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return {
        "observation.images.table": np.full(
            (4, 5, 3), index % 255, dtype=np.uint8
        ),
        "observation.state": state,
        "action": action,
        "episode_index": np.asarray(index % 3, dtype=np.int64),
        "task": "lift the cube",
    }


class _FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            chunk_size=2,
            max_action_dim=8,
            n_obs_steps=1,
        )
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def predict_action_chunk(self, batch, noise):
        state_term = batch["observation.state"][:, None, :8] * 0.01
        return noise[:, :, :8] * 0.02 + state_term


def _evaluate(
    batch_size: int,
    *,
    progress_every: int = 0,
    position_target_mode: str = "absolute",
):
    dataset = [_sample(index) for index in range(5)]
    policy = _FakePolicy()
    result = evaluate_checkpoint(
        torch=torch,
        checkpoint_step=500,
        policy=policy,
        preprocess=lambda batch: batch,
        postprocess=lambda actions: actions,
        dataset=dataset,
        selected_indices=[0, 1, 2, 3, 4],
        cached_samples=None,
        episode_levels={0: "easy", 1: "medium", 2: "hard"},
        seed=1234,
        batch_size=batch_size,
        progress_every=progress_every,
        checkpoint_started=time.perf_counter(),
        position_target_mode=position_target_mode,
    )
    return result, policy


def test_index_batches_keep_order_and_short_final_batch() -> None:
    assert list(iter_index_batches([5, 2, 9, 4, 1], 2)) == [
        [5, 2],
        [9, 4],
        [1],
    ]
    with pytest.raises(ValueError, match="positive"):
        list(iter_index_batches([1], 0))


def test_four_dimensional_scan_uses_gripper_index_three_and_no_quaternion():
    sample = _sample(0)
    sample["action"] = np.asarray([.01, .02, -.03, -1.], dtype=np.float32)
    sample["training.controller_phase"] = np.asarray([2.], dtype=np.float32)
    policy = _FakePolicy()
    policy.predict_action_chunk = lambda batch, noise: torch.tensor(
        [[sample["action"].tolist()]*2], dtype=torch.float32)
    result = evaluate_checkpoint(torch=torch, checkpoint_step=1, policy=policy,
        preprocess=lambda x:x, postprocess=lambda x:x, dataset=[sample], selected_indices=[0],
        cached_samples=None, episode_levels={0:"easy"}, seed=1, batch_size=1,
        progress_every=0, checkpoint_started=time.perf_counter(), position_target_mode="delta")
    assert result["position_mae_cm"] == 0
    assert result["gripper_sign_accuracy"] == 1
    assert result["orientation_mae_deg"] is None
    assert result["phase_metrics"]["2"]["samples"] == 1


def test_seeded_noise_matches_legacy_and_is_batch_partition_invariant() -> None:
    seed = 20261008
    indices = [3, 9, 12]
    rng_state = torch.random.get_rng_state().clone()
    together = deterministic_action_noise(
        torch,
        indices,
        seed=seed,
        chunk_size=4,
        max_action_dim=8,
        device="cpu",
    )
    split = torch.cat(
        [
            deterministic_action_noise(
                torch,
                part,
                seed=seed,
                chunk_size=4,
                max_action_dim=8,
                device="cpu",
            )
            for part in ([3], [9, 12])
        ],
        dim=0,
    )
    assert torch.equal(together, split)
    assert torch.equal(torch.random.get_rng_state(), rng_state)

    for offset, sample_index in enumerate(indices):
        with torch.random.fork_rng():
            torch.manual_seed(seed + sample_index)
            legacy = torch.normal(
                mean=0.0,
                std=1.0,
                size=(1, 4, 8),
                dtype=torch.float32,
                device="cpu",
            )
        assert torch.equal(together[offset : offset + 1], legacy)


def test_batched_fake_evaluation_matches_singleton_metrics() -> None:
    singleton, singleton_policy = _evaluate(1)
    batched, batched_policy = _evaluate(2)

    comparable_fields = [
        "samples",
        "position_mae_cm",
        "position_p90_cm",
        "position_axis_mae_cm",
        "current_eef_baseline_mae_cm",
        "position_baseline_mae_cm",
        "position_mae_cm_by_level",
        "orientation_mae_deg",
        "gripper_mae",
        "gripper_sign_accuracy",
        "finite_rate",
    ]
    for field in comparable_fields:
        if isinstance(singleton[field], (float, list, dict)):
            assert batched[field] == pytest.approx(singleton[field])
        else:
            assert batched[field] == singleton[field]
    assert singleton_policy.reset_count == 5
    assert batched_policy.reset_count == 3
    assert batched["batch_size_effective"] == 2


def test_delta_evaluation_uses_zero_action_baseline() -> None:
    result, _ = _evaluate(2, position_target_mode="delta")

    dataset = [_sample(index) for index in range(5)]
    expected_cm = np.mean(
        [np.linalg.norm(np.asarray(sample["action"])[:3]) for sample in dataset]
    ) * 100.0
    assert result["position_target_mode"] == "delta"
    assert result["position_baseline_mode"] == "zero_delta"
    assert result["zero_delta_baseline_mae_cm"] == pytest.approx(expected_cm)
    assert result["position_baseline_mae_cm"] == pytest.approx(expected_cm)
    assert result["current_eef_baseline_mae_cm"] is None


def test_evaluation_reports_each_non_padding_action_horizon() -> None:
    samples = [_sample(index) for index in range(2)]
    for sample in samples:
        action = np.asarray(sample["action"], dtype=np.float32)
        sample["action"] = np.stack((action, action + 0.01), axis=0)
        sample["action_is_pad"] = np.asarray([False, False])
    samples[1]["action_is_pad"] = np.asarray([False, True])
    policy = _FakePolicy()
    result = evaluate_checkpoint(
        torch=torch,
        checkpoint_step=500,
        policy=policy,
        preprocess=lambda batch: batch,
        postprocess=lambda actions: actions,
        dataset=samples,
        selected_indices=[0, 1],
        cached_samples=None,
        episode_levels={0: "easy", 1: "medium"},
        seed=1234,
        batch_size=2,
        progress_every=0,
        checkpoint_started=time.perf_counter(),
        position_target_mode="delta",
    )

    horizons = result["action_horizon_metrics"]
    assert [item["horizon_index"] for item in horizons] == [0, 1]
    assert [item["valid_samples"] for item in horizons] == [2, 1]
    assert horizons[0]["position_mae_cm"] == pytest.approx(
        result["position_mae_cm"]
    )


def test_progress_reports_intervals_and_short_final_batch(capsys) -> None:
    _evaluate(2, progress_every=2)
    lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("CHECKPOINT_PROGRESS=")
    ]
    payloads = [json.loads(line.split("=", 1)[1]) for line in lines]
    assert [item["processed"] for item in payloads] == [2, 4, 5]
    assert all(item["step"] == 500 for item in payloads)
    assert payloads[-1]["percent"] == 100.0
    assert payloads[-1]["eta_s"] == 0.0


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint_root=tmp_path / "checkpoints",
        dataset_root=tmp_path / "validation",
        repo_id="local/test-validation",
        samples=0,
        seed=123,
        device="cpu",
        batch_size=2,
        position_target_mode="absolute",
        threshold_position_mae_cm=2.0,
    )


def _result(step: int, position_mae_cm: float) -> dict[str, object]:
    return {
        "step": step,
        "samples": 2,
        "position_mae_cm": position_mae_cm,
    }


def test_incremental_report_roundtrip_and_resume_validation(tmp_path: Path) -> None:
    args = _args(tmp_path)
    config = make_scan_config(args, [500, 1000], [0, 1], dataset_length=2)
    partial = build_report(args, config, [_result(500, 1.5)])
    output = tmp_path / "scan.json"
    atomic_write_json(output, partial)

    loaded = json.loads(output.read_text(encoding="utf-8"))
    restored = validate_resume_report(loaded, config)
    assert [item["step"] for item in restored] == [500]
    assert loaded["status"] == "VLA_CHECKPOINT_SCAN_PARTIAL"
    assert loaded["completed_steps"] == [500]
    assert loaded["pending_steps"] == [1000]

    complete = build_report(
        args,
        config,
        [_result(500, 1.5), _result(1000, 1.0)],
    )
    atomic_write_json(output, complete)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == (
        "VLA_CHECKPOINT_SCAN_COMPLETE"
    )

    mismatched = dict(config)
    mismatched["seed"] = 999
    with pytest.raises(ValueError, match="seed"):
        validate_resume_report(complete, mismatched)


def test_atomic_write_keeps_previous_report_when_replace_fails(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "scan.json"
    output.write_text('{"generation": 1}', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(
        "scripts.scan_smolvla_checkpoints.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="simulated"):
        atomic_write_json(output, {"generation": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"generation": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rejects_legacy_duplicate_and_unknown_steps(tmp_path: Path) -> None:
    args = _args(tmp_path)
    config = make_scan_config(args, [500, 1000], [0, 1], dataset_length=2)
    with pytest.raises(ValueError, match="older scanner"):
        validate_resume_report({"results": []}, config)
    with pytest.raises(ValueError, match="duplicate"):
        validate_resume_report(
            {
                "scan_config": config,
                "results": [_result(500, 1.0), _result(500, 1.1)],
            },
            config,
        )
    with pytest.raises(ValueError, match="unrequested"):
        validate_resume_report(
            {"scan_config": config, "results": [_result(1500, 1.0)]},
            config,
        )


def test_cuda_oom_detection_does_not_swallow_other_runtime_errors() -> None:
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not is_cuda_oom(RuntimeError("tensor shape mismatch"))


def test_checkpoint_discovery_requires_every_explicit_step(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    for step in (2500, 5000):
        model_dir = checkpoint_root / f"{step:06d}" / "pretrained_model"
        model_dir.mkdir(parents=True)
        (model_dir / "adapter_model.safetensors").write_bytes(b"adapter")

    discovered = discover_peft_checkpoints(checkpoint_root, [5000, 2500])
    assert [path.parent.name for path in discovered] == ["002500", "005000"]

    with pytest.raises(RuntimeError, match=r"7500"):
        discover_peft_checkpoints(checkpoint_root, [2500, 7500])
    with pytest.raises(ValueError, match="duplicate"):
        discover_peft_checkpoints(checkpoint_root, [2500, 2500])


def test_adapter_only_checkpoint_uses_policy_from_train_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    policy = {"type": "smolvla", "chunk_size": 4, "max_action_dim": 32}
    (checkpoint / "train_config.json").write_text(
        json.dumps({"policy": policy}),
        encoding="utf-8",
    )

    assert read_policy_config_payload(checkpoint) == policy
    assert checkpoint_chunk_size(checkpoint) == 4
    assert ensure_policy_config_file(checkpoint)
    assert json.loads((checkpoint / "config.json").read_text(encoding="utf-8")) == policy
    assert not ensure_policy_config_file(checkpoint)


def test_full_checkpoint_config_takes_precedence(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"type": "smolvla", "chunk_size": 8}),
        encoding="utf-8",
    )
    (checkpoint / "train_config.json").write_text(
        json.dumps({"policy": {"type": "smolvla", "chunk_size": 4}}),
        encoding="utf-8",
    )

    assert checkpoint_chunk_size(checkpoint) == 8
    assert not ensure_policy_config_file(checkpoint)
