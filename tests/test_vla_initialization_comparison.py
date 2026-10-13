import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "comparison", Path(__file__).parents[1] / "scripts/compare_smolvla_initializations.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def scan():
    config = dict(dataset_root="validation", repo_id="local", dataset_length=3, indices=[0, 2],
                  seed=1, device="cuda", batch_size_requested=2, checkpoint_steps=[6000],
                  position_target_mode="delta", action_chunk_size=4, threshold_position_mae_cm=2)
    return dict(status="VLA_CHECKPOINT_SCAN_COMPLETE", dataset_split="validation",
                scan_config=config, indices=[0, 2], best_checkpoint={"step": 6000},
                results=[dict(step=6000, samples=2, finite_rate=1, batch_size_effective=2,
                              zero_delta_baseline_mae_cm=3, position_mae_cm=1.5)])


def test_same_final_checkpoint_comparison():
    c, f = scan(), scan()
    f["results"][0]["position_mae_cm"] = 1.2
    report = comparison.compare_scans(c, f)
    assert report["both_final_checkpoints_pass_2cm_gate"]
    assert report["primary_comparison"]["fresh_minus_continued_position_cm"] == pytest.approx(-.3)


@pytest.mark.parametrize("bad", ["indices", "seed", "batch", "baseline", "finite", "split"])
def test_rejects_unfair_validation(bad):
    c, f = scan(), scan()
    if bad == "indices": f["scan_config"]["indices"] = [1, 2]
    if bad == "seed": f["scan_config"]["seed"] = 2
    if bad == "batch": f["results"][0]["batch_size_effective"] = 1
    if bad == "baseline": f["results"][0]["zero_delta_baseline_mae_cm"] = 1
    if bad == "finite": f["results"][0]["finite_rate"] = .9
    if bad == "split": f["dataset_split"] = "train"
    with pytest.raises(ValueError):
        comparison.compare_scans(c, f)


def test_failed_final_gate_not_replaced_by_best_checkpoint():
    c, f = scan(), scan()
    f["results"][0]["position_mae_cm"] = 2.1
    assert not comparison.compare_scans(c, f)["both_final_checkpoints_pass_2cm_gate"]


def test_training_only_initialization_may_differ():
    config = dict(dataset={}, seed=1000, steps=6000, batch_size=1, num_workers=0,
                  optimizer={}, scheduler={}, resume=False, save_freq=500, tolerance_s=.0001,
                  policy={"pretrained_path": "old", "use_peft": True})
    fresh = copy.deepcopy(config)
    fresh["policy"]["pretrained_path"] = "base"
    comparison.verify_training_parity(config, fresh, {}, {})
    fresh["seed"] = 1
    with pytest.raises(ValueError):
        comparison.verify_training_parity(config, fresh, {}, {})


def test_closed_loop_rejects_different_protocol():
    c = ({"config": {"max_steps": 600}}, [], {})
    f = ({"config": {"max_steps": 300}}, [], {})
    with pytest.raises(ValueError):
        comparison.compare_benchmarks(c, f)


def test_failed_gate_requires_explicit_diagnostic_override():
    report = {"both_final_checkpoints_pass_2cm_gate": False}
    with pytest.raises(ValueError):
        comparison.verify_gate_override(report, False, "")
    with pytest.raises(ValueError):
        comparison.verify_gate_override(report, True, "")
    override = comparison.verify_gate_override(report, True, "explicit user request")
    assert override["used"]
    assert report["both_final_checkpoints_pass_2cm_gate"] is False


def test_override_does_not_claim_it_was_used_when_gate_passed():
    report = {"both_final_checkpoints_pass_2cm_gate": True}
    assert not comparison.verify_gate_override(report, True, "diagnostic")["used"]


def test_closed_loop_rejects_different_inference_seed():
    c = ({"config": {}}, [], {"inference_seed": 1})
    f = ({"config": {}}, [], {"inference_seed": 2})
    with pytest.raises(ValueError, match="Inference seed"):
        comparison.compare_benchmarks(c, f)


def test_distinct_service_ports_do_not_change_scene_protocol():
    scene_fields = ("sample_key", "level_seed", "sample_index", "object_x_m", "object_y_m",
                    "object_yaw_deg", "object_mass_kg", "static_friction", "dynamic_friction",
                    "restitution", "camera_position_m", "camera_target_m", "lighting_intensity")
    metric_fields = ("success", "failure_reason", "maximum_lift_cm",
                     "minimum_ee_object_distance_m", "steps", "final_controller_state",
                     "mean_vla_inference_ms")
    row = {key: "fixed" for key in scene_fields + metric_fields}
    c_report = {"config": {"vla_server": "127.0.0.1:8768", "max_steps": 600},
                "overall": {}, "levels": []}
    f_report = copy.deepcopy(c_report)
    f_report["config"]["vla_server"] = "127.0.0.1:8769"
    report = comparison.compare_benchmarks(
        (c_report, [row], {"inference_seed": 1}),
        (f_report, [row], {"inference_seed": 1}),
    )
    assert report["episodes_per_arm"] == 1
