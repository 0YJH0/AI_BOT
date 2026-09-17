"""Frame-level PyTorch dataset over the project's sharded RGB-D HDF5 format."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from .rgbd_pose import RGBDPoseModelCfg, preprocess_rgbd


@dataclass(frozen=True)
class RGBDFrameRef:
    episode_id: int
    shard_path: Path
    group_path: str
    frame_index: int


class RGBDFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Read individual RGB-D frames with camera-frame object-position labels."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        config: RGBDPoseModelCfg | None = None,
        augment: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split = split
        self.config = config or RGBDPoseModelCfg()
        self.augment = augment
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if not (self.dataset_dir / "_SUCCESS").is_file():
            raise ValueError(f"Dataset is not marked complete: {self.dataset_dir}")
        with (self.dataset_dir / "index.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if row["split"] == split and row.get("camera_data") == "True"
            ]
        self.frames: list[RGBDFrameRef] = []
        for row in rows:
            frame_count = int(row["camera_frames"])
            self.frames.extend(
                RGBDFrameRef(
                    episode_id=int(row["episode_id"]),
                    shard_path=self.dataset_dir / row["shard"],
                    group_path=row["hdf5_group"] + "/camera",
                    frame_index=frame_index,
                )
                for frame_index in range(frame_count)
            )
        self._handles: dict[Path, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", swmr=True)
        return self._handles[path]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame = self.frames[index]
        group = self._handle(frame.shard_path)[frame.group_path]
        rgb = torch.from_numpy(group["rgb"][frame.frame_index])
        depth = torch.from_numpy(group["depth"][frame.frame_index])
        target = torch.from_numpy(group["object_pose_camera"][frame.frame_index, :3]).float()
        intrinsics = torch.from_numpy(group["camera_intrinsics"][frame.frame_index]).float()
        raw_height, raw_width = rgb.shape[:2]
        resized_intrinsics = intrinsics.clone()
        resized_intrinsics[0, :] *= self.config.image_width / raw_width
        resized_intrinsics[1, :] *= self.config.image_height / raw_height
        target_pixel = torch.stack(
            (
                resized_intrinsics[0, 0] * target[0] / target[2] + resized_intrinsics[0, 2],
                resized_intrinsics[1, 1] * target[1] / target[2] + resized_intrinsics[1, 2],
            )
        )
        inputs = preprocess_rgbd(
            rgb,
            depth,
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            depth_clip_m=self.config.depth_clip_m,
        ).squeeze(0)
        grid = torch.stack(
            (
                target_pixel[0] / max(self.config.image_width - 1, 1) * 2.0 - 1.0,
                target_pixel[1] / max(self.config.image_height - 1, 1) * 2.0 - 1.0,
            )
        ).clamp(-1.0, 1.0).reshape(1, 1, 1, 2)
        normalized_surface_depth = functional.grid_sample(
            inputs[None, 3:4], grid, mode="bilinear", align_corners=True
        ).reshape(())
        surface_depth = (
            (normalized_surface_depth + 1.0) * 0.5 * self.config.depth_clip_m
        )
        if self.augment:
            rgb_scale = torch.empty(1).uniform_(0.85, 1.15)
            inputs[:3] = (inputs[:3] * rgb_scale).clamp_(-1.0, 1.0)
            inputs[:3] = (inputs[:3] + torch.randn_like(inputs[:3]) * 0.015).clamp_(
                -1.0, 1.0
            )
            inputs[3:] = (inputs[3:] + torch.randn_like(inputs[3:]) * 0.004).clamp_(
                -1.0, 1.0
            )
        return {
            "input": inputs,
            "target_position_camera": target,
            "camera_intrinsics": intrinsics,
            "resized_intrinsics": resized_intrinsics,
            "target_pixel": target_pixel,
            "surface_depth_at_target": surface_depth,
            "episode_id": torch.tensor(frame.episode_id, dtype=torch.int64),
            "frame_index": torch.tensor(frame.frame_index, dtype=torch.int64),
        }

    def target_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        targets: list[np.ndarray] = []
        for frame in self.frames:
            group = self._handle(frame.shard_path)[frame.group_path]
            targets.append(group["object_pose_camera"][frame.frame_index, :3])
        if not targets:
            raise ValueError(f"No RGB-D frames in split {self.split}")
        values = torch.from_numpy(np.asarray(targets, dtype=np.float32))
        return values.mean(dim=0), values.std(dim=0).clamp_min(1.0e-4)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        self.close()
