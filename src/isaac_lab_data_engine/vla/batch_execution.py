"""Synchronous chunk scheduling, with a fresh delta anchor at each action boundary."""
import numpy as np

from .hybrid_controller import decode_vla_position_target


class BatchChunkExecutor:
    def __init__(self, num_envs, horizon, repeat=5):
        if num_envs < 1 or horizon not in (1, 2, 4) or repeat < 1:
            raise ValueError('Invalid batch execution settings')
        self.num_envs, self.horizon, self.repeat = num_envs, horizon, repeat
        self.chunk = None
        self.index = 0
        self.remaining = 0
        self.held = None
        self.raw = None

    @property
    def needs_prediction(self):
        return self.remaining == 0 and (self.chunk is None or self.index == self.horizon)

    @property
    def starts_action(self):
        return self.remaining == 0

    def load(self, actions):
        if not self.needs_prediction:
            raise ValueError('Cannot replace an unexecuted chunk')
        actions = np.asarray(actions, dtype=np.float32)
        if (actions.ndim != 3 or actions.shape[0] != self.num_envs or
                actions.shape[1] < self.horizon or actions.shape[2] != 8 or
                not np.isfinite(actions).all()):
            raise ValueError('Invalid or too short batched action chunk')
        self.chunk = actions[:, :self.horizon].copy()
        self.index = 0

    def advance(self, eef):
        if self.needs_prediction:
            raise ValueError('Prediction required before advancing')
        if self.remaining == 0:
            eef = np.asarray(eef)
            if eef.shape != (self.num_envs, 7):
                raise ValueError('Expected one current 7-D end-effector pose per slot')
            self.raw = self.chunk[:, self.index].copy()
            self.held = np.stack([decode_vla_position_target(self.raw[i], eef[i],
                position_target_mode='delta') for i in range(self.num_envs)])
            self.index += 1
            self.remaining = self.repeat
        self.remaining -= 1
        return self.held.copy()
