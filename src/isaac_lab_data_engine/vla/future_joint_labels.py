"""Construct position-control waypoint labels from future measured arm states."""
import numpy as np


def future_joint_labels(states, cartesian_actions, control_steps, horizon=1):
    states=np.asarray(states,np.float32)
    cartesian_actions=np.asarray(cartesian_actions,np.float32)
    steps=np.asarray(control_steps)
    if states.ndim!=2 or states.shape[1]!=30 or cartesian_actions.shape!=(len(states),8):
        raise ValueError('Expected one 30-state / 8-action original episode')
    if (not isinstance(horizon, (int, np.integer)) or horizon<1 or horizon>=len(states)
            or steps.shape != (len(states),) or not np.isfinite(steps).all()
            or not np.all(np.diff(steps)==5)):
        raise ValueError('Need contiguous 10 Hz frames and a valid future horizon')
    if not np.isfinite(states).all() or not np.isfinite(cartesian_actions).all():
        raise ValueError('Nonfinite episode')
    # Future observations are targets only, never student inputs. Do not cross
    # episode boundaries or synthesize a target for the last horizon frames.
    actions=np.concatenate((states[horizon:,:7],cartesian_actions[:-horizon,7:8]),axis=1)
    return states[:-horizon].copy(),actions.astype(np.float32)
