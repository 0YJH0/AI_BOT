"""Strict consecutive-frame readiness, independent of Isaac/renderer polling."""
from collections import deque
import numpy as np


class ConsecutiveVisualWindow:
    def __init__(self, size=5, max_spread_m=.025):
        if size < 1 or max_spread_m <= 0:
            raise ValueError("Invalid visual readiness settings")
        self.samples = deque(maxlen=size)
        self.size, self.max_spread_m = size, max_spread_m
        self.last_frame = None
        self.valid_frames = self.invalid_frames = self.duplicate_frames = 0
        self.consecutive = self.maximum_consecutive = 0
        self.spread_m = None
        self.median = None

    def observe(self, frame, position, *, quality_valid, pose_valid):
        # Re-reading one sensor buffer must neither fill nor break the window.
        if frame == self.last_frame:
            self.duplicate_frames += 1
            return False
        self.last_frame = frame
        position = np.asarray(position, dtype=np.float32)
        if not (quality_valid and pose_valid and position.shape == (3,)
                and np.isfinite(position).all()):
            self.invalid_frames += 1
            self.consecutive = 0
            self.samples.clear()
            self.spread_m = self.median = None
            return False
        self.valid_frames += 1
        self.consecutive += 1
        self.maximum_consecutive = max(self.maximum_consecutive, self.consecutive)
        self.samples.append(position.copy())
        values = np.asarray(self.samples)
        self.median = np.median(values, axis=0).astype(np.float32)
        self.spread_m = float(np.linalg.norm(values - self.median, axis=1).max())
        return len(values) == self.size and self.spread_m <= self.max_spread_m
