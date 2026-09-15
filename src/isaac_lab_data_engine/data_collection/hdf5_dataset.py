"""Split-aware reader for Phase 6 sharded HDF5 datasets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .episode_writer import CAMERA_TENSOR_SCHEMA, TENSOR_SCHEMA


def _read_json_dataset(dataset: h5py.Dataset) -> dict[str, Any]:
    value = dataset[()]
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return json.loads(text)


class HDF5EpisodeDataset:
    """Random-access episode reader suitable for worker-process data loading.

    A shard is opened only for the duration of ``__getitem__``. This is slower
    than a process-local cache but remains safe when instances are copied into
    PyTorch DataLoader workers.
    """

    def __init__(self, dataset_dir: str | Path, split: str | None = None) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        if not (self.dataset_dir / "_SUCCESS").is_file():
            raise ValueError(f"Dataset is not marked complete: {self.dataset_dir}")
        if split not in (None, "train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        with (self.dataset_dir / "index.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.rows = rows if split is None else [row for row in rows if row["split"] == split]
        self.split = split
        self._row_by_episode_id = {int(row["episode_id"]): row for row in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def episode_ids(self) -> list[int]:
        return [int(row["episode_id"]) for row in self.rows]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._load_row(self.rows[index])

    def get_episode(self, episode_id: int) -> dict[str, Any]:
        try:
            row = self._row_by_episode_id[int(episode_id)]
        except KeyError as exc:
            raise KeyError(f"Episode {episode_id} is not present in this dataset view") from exc
        return self._load_row(row)

    def _load_row(self, row: dict[str, str]) -> dict[str, Any]:
        with h5py.File(self.dataset_dir / row["shard"], "r", swmr=True) as shard:
            group = shard[row["hdf5_group"]]
            trajectory = {name: group[name][...] for name in TENSOR_SCHEMA}
            metadata = _read_json_dataset(group["metadata_json"])
            domain_params = (
                _read_json_dataset(group["domain_params_json"])
                if "domain_params_json" in group
                else None
            )
            camera = (
                {name: group["camera"][name][...] for name in CAMERA_TENSOR_SCHEMA}
                if "camera" in group
                else None
            )
        return {
            "episode_id": int(row["episode_id"]),
            "split": row["split"],
            "trajectory": trajectory,
            "metadata": metadata,
            "domain_params": domain_params,
            "camera": camera,
            "index": dict(row),
        }

    def trajectory_lengths(self) -> np.ndarray:
        return np.asarray([int(row["steps"]) for row in self.rows], dtype=np.int64)
