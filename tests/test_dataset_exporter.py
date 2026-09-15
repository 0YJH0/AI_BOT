"""Round-trip tests for the Phase 6 sharded HDF5 exporter."""

from __future__ import annotations

import csv
import json

import h5py
import numpy as np
import pytest

from isaac_lab_data_engine.data_collection import CameraFrameBuffer, EpisodeBuffer, EpisodeWriter
from isaac_lab_data_engine.data_collection import dataset_exporter
from isaac_lab_data_engine.data_collection.dataset_exporter import export_dataset
from isaac_lab_data_engine.data_collection.hdf5_dataset import HDF5EpisodeDataset


def _buffer(episode_id: int) -> EpisodeBuffer:
    buffer = EpisodeBuffer()
    for step in range(2 + episode_id % 3):
        buffer.append(
            q=np.full(9, episode_id + step, dtype=np.float32),
            dq=np.zeros(9, dtype=np.float32),
            eef_pose=np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            object_pose=np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            action=np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32),
            reward=float(step),
            timestamp=(step + 1) * 0.02,
            controller_state=step,
        )
    return buffer


def _make_raw_dataset(path, count: int = 10) -> None:
    writer = EpisodeWriter(path)
    for episode_id in range(1, count + 1):
        buffer = _buffer(episode_id)
        steps = len(buffer)
        success = episode_id % 2 == 0
        writer.write(
            episode_id,
            buffer,
            {
                "success": success,
                "failure_reason": "none" if success else "test_failure",
                "steps": steps,
                "completion_time_s": steps * 0.02,
                "total_reward": float(buffer.as_arrays()["reward"].sum()),
                "parallel_batch_index": 0,
                "parallel_env_index": episode_id - 1,
                "parallel_num_envs": count,
                "domain_randomization": True,
                "domain_params_file": "domain_params.json",
            },
            sidecar_json={
                "domain_params.json": {
                    "object": {
                        "position_m": [0.5 + episode_id * 0.001, 0.0, 0.48],
                        "yaw_deg": float(episode_id),
                        "mass_kg": 0.1,
                        "static_friction": 0.5,
                        "dynamic_friction": 0.4,
                    },
                    "camera": {"applied": False},
                    "lighting": {"intensity": 2000.0},
                    "applied_values": {},
                }
            },
        )


def test_hdf5_export_round_trip_and_resume(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "export"
    _make_raw_dataset(source)

    summary = export_dataset(
        source,
        output,
        episodes_per_shard=3,
        split_ratios=(0.6, 0.2, 0.2),
        split_seed=7,
    )

    assert summary["status"] == "PHASE6_EXPORT_OK"
    assert summary["num_episodes"] == 10
    assert summary["num_shards"] == 4
    assert summary["split_counts"] == {"train": 6, "validation": 2, "test": 2}
    assert (output / "_SUCCESS").is_file()

    with (output / "index.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["split"] for row in rows} == {"train", "validation", "test"}

    first_row = rows[0]
    with h5py.File(output / first_row["shard"], "r") as shard:
        exported_q = shard[first_row["hdf5_group"]]["q"][...]
    with np.load(source / first_row["source_episode_path"] / "trajectory.npz") as raw:
        assert np.array_equal(exported_q, raw["q"])

    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "DATASET_EXPORT_VALIDATION_OK"
    assert report["source_value_comparison"] is True

    train_dataset = HDF5EpisodeDataset(output, split="train")
    assert len(train_dataset) == 6
    sample = train_dataset[0]
    assert sample["episode_id"] in train_dataset.episode_ids
    assert sample["trajectory"]["q"].shape[0] == sample["metadata"]["steps"]
    assert sample["domain_params"]["object"]["mass_kg"] == 0.1
    assert train_dataset.get_episode(sample["episode_id"])["split"] == "train"
    assert train_dataset.trajectory_lengths().shape == (6,)

    resumed = export_dataset(source, output, resume=True)
    assert resumed["status"] == "DATASET_EXPORT_VALIDATION_OK"
    assert resumed["resumed_existing_export"] is True


def test_export_refuses_existing_output_without_resume(tmp_path):
    source = tmp_path / "raw"
    output = tmp_path / "export"
    _make_raw_dataset(source, count=2)
    export_dataset(source, output, episodes_per_shard=1)

    with pytest.raises(FileExistsError):
        export_dataset(source, output)


def test_incomplete_staging_export_can_resume(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    output = tmp_path / "export"
    _make_raw_dataset(source, count=5)
    original_write_shard = dataset_exporter._write_shard
    calls = 0

    def fail_on_second_shard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original_write_shard(*args, **kwargs)

    monkeypatch.setattr(dataset_exporter, "_write_shard", fail_on_second_shard)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        export_dataset(source, output, episodes_per_shard=2)

    staging = output.parent / f".{output.name}.staging"
    state = json.loads((staging / "export_state.json").read_text(encoding="utf-8"))
    assert state["completed_shards"] == [0]
    assert not output.exists()

    monkeypatch.setattr(dataset_exporter, "_write_shard", original_write_shard)
    summary = export_dataset(source, output, episodes_per_shard=2, resume=True)
    assert summary["status"] == "PHASE6_EXPORT_OK"
    assert summary["num_shards"] == 3
    assert (output / "_SUCCESS").is_file()


def test_rgbd_camera_data_exports_and_reads_back(tmp_path):
    source = tmp_path / "raw_rgbd"
    output = tmp_path / "export_rgbd"
    trajectory = _buffer(1)
    camera = CameraFrameBuffer()
    camera.append(
        frame_step=1,
        timestamp=0.02,
        rgb=np.full((8, 10, 3), 128, dtype=np.uint8),
        depth=np.full((8, 10), 1.25, dtype=np.float32),
        camera_intrinsics=np.array([[10, 0, 5], [0, 10, 4], [0, 0, 1]]),
        camera_pose_w=np.array([1, 0, 1, 1, 0, 0, 0]),
        object_pose_w=np.array([0.5, 0, 0.5, 1, 0, 0, 0]),
        eef_pose_w=np.array([0.5, 0, 0.7, 1, 0, 0, 0]),
    )
    EpisodeWriter(source).write(
        1,
        trajectory,
        {
            "success": True,
            "failure_reason": "none",
            "steps": len(trajectory),
            "completion_time_s": len(trajectory) * 0.02,
            "total_reward": 1.0,
        },
        camera_buffer=camera,
    )

    summary = export_dataset(source, output, episodes_per_shard=1)
    assert summary["validation"] == "DATASET_EXPORT_VALIDATION_OK"
    row = next(csv.DictReader((output / "index.csv").open(encoding="utf-8")))
    assert row["camera_data"] == "True"
    assert row["camera_frames"] == "1"
    sample = HDF5EpisodeDataset(output)[0]
    assert sample["camera"]["rgb"].shape == (1, 8, 10, 3)
    assert sample["camera"]["depth"].dtype == np.float32
    assert np.array_equal(sample["camera"]["rgb"][0], np.full((8, 10, 3), 128))
