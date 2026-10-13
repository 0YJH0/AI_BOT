import copy
import numpy as np
import pytest

from isaac_lab_data_engine.vla.full_trajectory import FORMAT, validate_full_trajectory


def example():
    n = 20
    state = np.zeros((n, 30), dtype=np.float32)
    state[:, 29] = np.repeat(np.arange(5), 4)
    action = np.zeros((n, 8), dtype=np.float32)
    action[:, 4] = 1
    action[:, 7] = np.where(state[:, 29] < 3, 1, -1)
    rgb = np.tile(np.asarray([0, 255, 128], dtype=np.uint8), (n, 8, 8, 1))
    arrays = dict(state=state, action=action, controller_phase=state[:, 29].astype(int),
                  control_step=1+5*np.arange(n), rgb=rgb, depth=np.ones((n, 8, 8)),
                  physics_action=np.zeros((100, 8)), applied_absolute_action=np.zeros((n, 8)))
    metadata = dict(format=FORMAT, success=True, maximum_lift_cm=12,
                    sustained_success_steps=10, model_requests=0, scripted_fallback_actions=0)
    return arrays, metadata


def test_full_trajectory_requires_causal_applied_actions():
    arrays, metadata = example()
    assert validate_full_trajectory(arrays, metadata) == 20
    arrays["physics_action"][1, 0] = 0.1
    with pytest.raises(ValueError, match="actually applied"):
        validate_full_trajectory(arrays, metadata)


@pytest.mark.parametrize("bad", ["blank", "lift", "phase", "cadence", "delta", "fallback"])
def test_rejects_incomplete_or_invalid_demonstration(bad):
    arrays, metadata = example()
    if bad == "blank": arrays["rgb"][:] = 0
    if bad == "lift": metadata["maximum_lift_cm"] = 9.9
    if bad == "phase": arrays["state"][:, 29] = 0
    if bad == "cadence": arrays["control_step"][1] += 1
    if bad == "delta": arrays["action"][0, 0] = 0.05
    if bad == "fallback": metadata["scripted_fallback_actions"] = 1
    with pytest.raises(ValueError):
        validate_full_trajectory(arrays, metadata)
