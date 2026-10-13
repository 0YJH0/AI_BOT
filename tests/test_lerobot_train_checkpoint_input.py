"""Regression coverage for adapter-only inputs to the Windows trainer."""

import json
from pathlib import Path

import pytest

from scripts.lerobot_train_windows import _prepare_local_policy_config


@pytest.mark.parametrize("separate_argument", [False, True])
def test_prepare_adapter_config_preserves_weights_and_cli(tmp_path: Path, separate_argument: bool) -> None:
    checkpoint = tmp_path / "checkpoint with spaces"
    checkpoint.mkdir()
    policy = {"type": "smolvla", "chunk_size": 4, "optimizer_lr": 1e-5}
    (checkpoint / "train_config.json").write_text(json.dumps({"policy": policy}), encoding="utf-8")
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"original adapter weights")
    argv = (["--policy.path", str(checkpoint)] if separate_argument else [f"--policy.path={checkpoint}"])
    argv.append("--policy.optimizer_lr=5e-6")
    original_argv = list(argv)

    assert _prepare_local_policy_config(argv) == checkpoint
    assert json.loads((checkpoint / "config.json").read_text()) == policy
    assert weights.read_bytes() == b"original adapter weights"
    assert argv == original_argv
    assert _prepare_local_policy_config(argv) == checkpoint


def test_prepare_preserves_existing_config_and_ignores_hub_id(tmp_path: Path) -> None:
    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    config = '{"type":"smolvla","chunk_size":8}\n'
    (checkpoint / "config.json").write_text(config, encoding="utf-8")
    (checkpoint / "train_config.json").write_text('{"policy":{"type":"smolvla","chunk_size":4}}')
    assert _prepare_local_policy_config([f"--policy.path={checkpoint}"]) == checkpoint
    assert (checkpoint / "config.json").read_text() == config
    assert _prepare_local_policy_config(["--policy.path=local/nonexistent-hub-policy"]) is None


def test_prepare_rejects_missing_policy_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "broken"
    checkpoint.mkdir()
    (checkpoint / "train_config.json").write_text('{"policy":{}}')
    with pytest.raises(ValueError, match="no policy type"):
        _prepare_local_policy_config([f"--policy.path={checkpoint}"])
    assert not (checkpoint / "config.json").exists()
