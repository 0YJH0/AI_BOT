"""Domain randomization utilities."""

from pathlib import Path

import yaml

from .domain_randomizer import DomainRandomizer


def create_domain_randomizer(config_path: str | Path) -> DomainRandomizer:
    """Create the configured uniform or failure-aware randomizer."""
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("strategy", {}).get("type") == "failure_replay_jitter":
        # Local import avoids a module cycle during package initialization.
        from isaac_lab_data_engine.data_engine import FailureAwareDomainRandomizer

        return FailureAwareDomainRandomizer(path)
    return DomainRandomizer(path)


__all__ = ["DomainRandomizer", "create_domain_randomizer"]
