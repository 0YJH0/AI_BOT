"""Compatibility helpers for LeRobot PEFT-only policy checkpoints."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_policy_config_payload(checkpoint: str | Path) -> dict[str, Any]:
    """Read a policy config from a full or adapter-only LeRobot checkpoint.

    Fresh PEFT runs contain ``config.json``. When LeRobot 0.4.4 fine-tunes an
    already wrapped ``PeftModel``, PEFT saves only the adapter while LeRobot
    keeps the complete policy config under ``train_config.json["policy"]``.
    Both layouts describe the same runtime policy and are accepted here.
    """

    checkpoint = Path(checkpoint)
    config_path = checkpoint / "config.json"
    train_config_path = checkpoint / "train_config.json"

    if config_path.is_file():
        source_path = config_path
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    elif train_config_path.is_file():
        source_path = train_config_path
        train_payload = json.loads(train_config_path.read_text(encoding="utf-8"))
        payload = train_payload.get("policy") if isinstance(train_payload, dict) else None
    else:
        raise FileNotFoundError(
            f"Neither config.json nor train_config.json exists in {checkpoint}"
        )

    if not isinstance(payload, dict):
        raise ValueError(f"Policy config in {source_path} must be a JSON object")
    if not payload.get("type"):
        raise ValueError(f"Policy config in {source_path} has no policy type")
    return dict(payload)


def checkpoint_chunk_size(checkpoint: str | Path) -> int:
    """Return and validate the action chunk size without importing LeRobot."""

    payload = read_policy_config_payload(checkpoint)
    try:
        chunk_size = int(payload["chunk_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid chunk_size in policy config under {checkpoint}") from error
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive under {checkpoint}")
    return chunk_size


def load_lerobot_policy_config(checkpoint: str | Path):
    """Load ``PreTrainedConfig`` from either supported checkpoint layout."""

    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(checkpoint)
    if (checkpoint / "config.json").is_file():
        return PreTrainedConfig.from_pretrained(str(checkpoint))

    payload = read_policy_config_payload(checkpoint)
    with tempfile.TemporaryDirectory(prefix="lerobot-policy-config-") as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "config.json").write_text(
            json.dumps(payload, indent=4),
            encoding="utf-8",
        )
        return PreTrainedConfig.from_pretrained(str(temp_path))


def ensure_policy_config_file(checkpoint: str | Path) -> bool:
    """Materialize missing ``config.json`` metadata after a PEFT-only save.

    Returns ``True`` only when a new file was written. Existing config files
    are never overwritten.
    """

    checkpoint = Path(checkpoint)
    destination = checkpoint / "config.json"
    if destination.is_file():
        return False

    payload = read_policy_config_payload(checkpoint)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return True
