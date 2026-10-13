"""Exact saved delta labels versus the same rule evaluated on live states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import build_visual_fsm_bounded_actions


class TrainingLabelReplay:
    def __init__(self, bundle_path: Path, level: str, mode: str):
        if mode not in {"stored", "online"}:
            raise ValueError("Label mode must be stored or online")
        self.mode = mode
        self.cursor = 0
        with np.load(bundle_path, allow_pickle=False) as bundle:
            self.states = bundle[f"{level}_states"].copy()
            self.actions = bundle[f"{level}_actions"].copy()
            self.domain = json.loads(str(bundle[f"{level}_domain_json"].item()))
            self.source = str(bundle[f"{level}_source"].item())
            self.episode_id = int(bundle[f"{level}_episode_id"].item())
            self.frame_indices = bundle[f"{level}_frame_indices"].copy()
            self.semantics = json.loads(str(bundle["semantics_json"].item()))
        if self.states.shape != (len(self.actions), 30) or self.actions.shape[1:] != (8,):
            raise ValueError("Expected 30-D states and 8-D saved actions")

    def next_action(self, eef_pose: np.ndarray, visual_position: np.ndarray, phase: int):
        """None means the saved sequence ended; caller holds the last absolute target."""
        if self.mode == "stored":
            if self.cursor >= len(self.actions):
                return None
            action = self.actions[self.cursor].copy()
        else:
            action = build_visual_fsm_bounded_actions(
                np.asarray(eef_pose, dtype=np.float32).reshape(1, 7),
                np.asarray(visual_position, dtype=np.float32).reshape(1, 3),
                np.asarray([phase]),
                maximum_position_step_m=self.semantics["maximum_position_step_m"],
                approach_height_m=self.semantics["approach_height_m"],
                lift_height_m=self.semantics["lift_height_m"],
                position_target_mode="delta")[0]
        self.cursor += 1
        return action
