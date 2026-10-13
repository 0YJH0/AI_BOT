"""Guard against presenting mismatched VLA runs as a fixed-scene comparison."""

from copy import deepcopy

import pytest

from scripts.summarize_vla_gpu_steps import compare_runs


def paired_runs():
    shared_config = {key: "same" for key in (
        "config_sha256", "controller_mode", "scripted_ik_fallback_enabled",
        "condition_vla_on_visual_pose", "condition_vla_on_controller_phase",
        "vla_position_target_mode", "execution_horizon", "action_repeat",
        "success_criteria", "vision_checkpoint_sha256", "levels",
        "camera_ready_timeout_s", "visual_initialization")}
    shared_episode = {key: "same" for key in (
        "sample_key", "object_x_m", "object_y_m", "object_yaw_deg", "object_mass_kg",
        "static_friction", "dynamic_friction", "restitution", "camera_position_m",
        "camera_target_m", "lighting_intensity")}
    first = {"server": {"checkpoint": "same", "seed": 123, "device": "cuda", "gpu_memory_fraction": 0.35},
             "adapter_sha256": "same", "config": dict(shared_config, max_steps=300),
             "episode": shared_episode}
    second = deepcopy(first)
    second["config"]["max_steps"] = 600
    return [first, second]


def test_paired_budgets_pass_and_are_sorted():
    result = compare_runs(paired_runs()[::-1])
    assert all(result["invariants"].values())
    assert [r["config"]["max_steps"] for r in result["runs"]] == [300, 600]


@pytest.mark.parametrize("section,key,value", [
    ("server", "seed", 124),
    ("episode", "object_x_m", "changed"),
    ("config", "action_repeat", 10),
])
def test_mismatched_runs_are_rejected(section, key, value):
    runs = paired_runs()
    runs[1][section][key] = value
    with pytest.raises(ValueError, match="invariants failed"):
        compare_runs(runs)
