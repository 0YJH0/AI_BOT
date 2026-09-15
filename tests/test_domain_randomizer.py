"""Simulator-free tests for deterministic Phase 4 parameter sampling."""

from __future__ import annotations

from pathlib import Path

from isaac_lab_data_engine.randomization import DomainRandomizer


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "randomization.yaml"


def test_samples_are_reproducible_and_in_range():
    first = DomainRandomizer(CONFIG_PATH)
    second = DomainRandomizer(CONFIG_PATH)
    samples_a = [first.sample() for _ in range(3)]
    samples_b = [second.sample() for _ in range(3)]

    assert samples_a == samples_b
    assert len({tuple(sample["object"]["position_m"]) for sample in samples_a}) == 3

    for sample in samples_a:
        obj = sample["object"]
        assert 0.48 <= obj["position_m"][0] <= 0.62
        assert -0.12 <= obj["position_m"][1] <= 0.12
        assert 0.05 <= obj["mass_kg"] <= 0.20
        assert 0.45 <= obj["static_friction"] <= 0.95
        assert 0.0 <= obj["dynamic_friction"] <= obj["static_friction"]
        assert -180.0 <= obj["yaw_deg"] <= 180.0
        assert 1800.0 <= sample["lighting"]["intensity"] <= 3200.0


def test_batch_has_independent_objects_and_explicitly_shared_lighting():
    samples = DomainRandomizer(CONFIG_PATH).sample_batch(4)

    assert [sample["sample_index"] for sample in samples] == [1, 2, 3, 4]
    assert len({tuple(sample["object"]["position_m"]) for sample in samples}) == 4
    assert all(sample["lighting"] == samples[0]["lighting"] for sample in samples)
    assert samples[0]["lighting"]["shared_across_batch"] is True
