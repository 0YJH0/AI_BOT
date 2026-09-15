"""Training-ready, resumable HDF5 export for raw episode datasets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from .episode_writer import FORMAT_VERSION as RAW_FORMAT_VERSION
from .episode_writer import CAMERA_TENSOR_SCHEMA, TENSOR_SCHEMA, validate_episode


EXPORT_FORMAT_VERSION = "1.0.0"
INDEX_FIELDS = [
    "episode_id",
    "split",
    "shard",
    "hdf5_group",
    "steps",
    "success",
    "failure_reason",
    "completion_time_s",
    "total_reward",
    "parallel_batch_index",
    "parallel_env_index",
    "parallel_num_envs",
    "domain_randomization",
    "object_x_m",
    "object_y_m",
    "object_yaw_deg",
    "object_mass_kg",
    "static_friction",
    "dynamic_friction",
    "camera_applied",
    "camera_data",
    "camera_frames",
    "source_camera_sha256",
    "source_episode_path",
    "source_trajectory_sha256",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _package_version(name: str) -> str | None:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class RawEpisode:
    episode_id: int
    path: Path
    metadata: dict[str, Any]
    domain_params: dict[str, Any] | None
    trajectory_sha256: str
    camera_sha256: str | None


def _load_raw_episodes(source_dir: Path) -> tuple[dict[str, Any], list[RawEpisode]]:
    manifest_path = source_dir / "dataset_metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Source manifest does not contain any episodes")

    records: list[RawEpisode] = []
    seen_ids: set[int] = set()
    for entry in sorted(entries, key=lambda item: int(item["episode_id"])):
        episode_id = int(entry["episode_id"])
        if episode_id in seen_ids:
            raise ValueError(f"Duplicate source episode_id: {episode_id}")
        seen_ids.add(episode_id)
        episode_dir = source_dir / entry["path"]
        validation = validate_episode(episode_dir)
        if int(validation["episode_id"]) != episode_id:
            raise ValueError(f"Manifest/path episode mismatch for {episode_dir}")
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        domain_params = None
        domain_file = metadata.get("domain_params_file")
        if domain_file:
            domain_params = json.loads((episode_dir / domain_file).read_text(encoding="utf-8"))
        trajectory_path = episode_dir / "trajectory.npz"
        camera_file = metadata.get("camera_file")
        camera_path = episode_dir / camera_file if camera_file else None
        records.append(
            RawEpisode(
                episode_id=episode_id,
                path=episode_dir,
                metadata=metadata,
                domain_params=domain_params,
                trajectory_sha256=_sha256(trajectory_path),
                camera_sha256=_sha256(camera_path) if camera_path is not None else None,
            )
        )
    return manifest, records


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def make_splits(
    records: Iterable[RawEpisode],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Create reproducible, success-stratified train/validation/test splits."""

    if len(ratios) != 3 or any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must contain three non-negative values")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}")

    rng = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    records = list(records)
    for label in (False, True):
        ids = np.asarray(
            [record.episode_id for record in records if bool(record.metadata["success"]) is label],
            dtype=np.int64,
        )
        rng.shuffle(ids)
        counts = _allocate_counts(len(ids), ratios)
        start = 0
        for split_name, count in zip(splits, counts, strict=True):
            splits[split_name].extend(int(value) for value in ids[start : start + count])
            start += count
    for values in splits.values():
        values.sort()
    return splits


def _split_lookup(splits: dict[str, list[int]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for split_name, episode_ids in splits.items():
        for episode_id in episode_ids:
            if episode_id in lookup:
                raise ValueError(f"Episode {episode_id} appears in multiple splits")
            lookup[episode_id] = split_name
    return lookup


def _compression_options(compression: str, compression_level: int) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        if not 0 <= compression_level <= 9:
            raise ValueError("gzip compression level must be between 0 and 9")
        return {"compression": "gzip", "compression_opts": compression_level, "shuffle": True}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise ValueError(f"Unsupported compression: {compression}")


def _write_shard(
    path: Path,
    shard_index: int,
    records: list[RawEpisode],
    source_dir: Path,
    source_manifest: dict[str, Any],
    compression: str,
    compression_level: int,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    compression_kwargs = _compression_options(compression, compression_level)
    with h5py.File(temporary, "w", libver="latest") as shard:
        shard.attrs["export_format_version"] = EXPORT_FORMAT_VERSION
        shard.attrs["raw_format_version"] = source_manifest.get("format_version", RAW_FORMAT_VERSION)
        shard.attrs["shard_index"] = shard_index
        shard.attrs["source_dataset"] = str(source_dir)
        shard.attrs["created_at"] = _utc_now()
        shard.attrs["episode_ids_json"] = json.dumps([record.episode_id for record in records])
        shard.attrs["tensor_schema_json"] = json.dumps(
            source_manifest.get("tensor_schema", TENSOR_SCHEMA), ensure_ascii=False
        )
        episodes_group = shard.create_group("episodes", track_order=True)
        for record in records:
            group_name = f"episode_{record.episode_id:06d}"
            group = episodes_group.create_group(group_name)
            group.attrs["episode_id"] = record.episode_id
            group.attrs["success"] = bool(record.metadata["success"])
            group.attrs["steps"] = int(record.metadata["steps"])
            group.attrs["source_trajectory_sha256"] = record.trajectory_sha256
            with np.load(record.path / "trajectory.npz") as trajectory:
                for tensor_name in TENSOR_SCHEMA:
                    group.create_dataset(
                        tensor_name,
                        data=trajectory[tensor_name],
                        chunks=True,
                        **compression_kwargs,
                    )
            camera_file = record.metadata.get("camera_file")
            if camera_file:
                camera_group = group.create_group("camera")
                with np.load(record.path / camera_file) as camera_data:
                    for tensor_name in CAMERA_TENSOR_SCHEMA:
                        camera_group.create_dataset(
                            tensor_name,
                            data=camera_data[tensor_name],
                            chunks=True,
                            **compression_kwargs,
                        )
                camera_group.attrs["source_camera_sha256"] = record.camera_sha256
            group.create_dataset(
                "metadata_json",
                data=json.dumps(record.metadata, ensure_ascii=False),
                dtype=string_dtype,
            )
            if record.domain_params is not None:
                group.create_dataset(
                    "domain_params_json",
                    data=json.dumps(record.domain_params, ensure_ascii=False),
                    dtype=string_dtype,
                )
        shard.flush()
    os.replace(temporary, path)


def _index_row(
    record: RawEpisode,
    split: str,
    shard_name: str,
    source_dir: Path,
) -> dict[str, Any]:
    domain = record.domain_params or {}
    obj = domain.get("object", {})
    camera = domain.get("camera", {})
    position = obj.get("position_m", [None, None, None])
    metadata = record.metadata
    return {
        "episode_id": record.episode_id,
        "split": split,
        "shard": f"shards/{shard_name}",
        "hdf5_group": f"/episodes/episode_{record.episode_id:06d}",
        "steps": int(metadata["steps"]),
        "success": bool(metadata["success"]),
        "failure_reason": metadata["failure_reason"],
        "completion_time_s": float(metadata["completion_time_s"]),
        "total_reward": float(metadata["total_reward"]),
        "parallel_batch_index": metadata.get("parallel_batch_index"),
        "parallel_env_index": metadata.get("parallel_env_index"),
        "parallel_num_envs": metadata.get("parallel_num_envs", 1),
        "domain_randomization": bool(metadata.get("domain_randomization", False)),
        "object_x_m": position[0],
        "object_y_m": position[1],
        "object_yaw_deg": obj.get("yaw_deg"),
        "object_mass_kg": obj.get("mass_kg"),
        "static_friction": obj.get("static_friction"),
        "dynamic_friction": obj.get("dynamic_friction"),
        "camera_applied": camera.get("applied"),
        "camera_data": bool(metadata.get("camera_data", False)),
        "camera_frames": int(metadata.get("camera_frames", 0)),
        "source_camera_sha256": record.camera_sha256,
        "source_episode_path": str(record.path.relative_to(source_dir)),
        "source_trajectory_sha256": record.trajectory_sha256,
    }


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(path, stream.getvalue())


def _read_utf8_dataset(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def validate_export(
    export_dir: str | Path,
    source_dir: str | Path | None = None,
    compare_source: bool = True,
) -> dict[str, Any]:
    """Validate shard structure, splits, finiteness, and optional source equality."""

    started = time.perf_counter()
    export_dir = Path(export_dir).resolve()
    source_path = Path(source_dir).resolve() if source_dir is not None else None
    with (export_dir / "index.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    splits = json.loads((export_dir / "splits.json").read_text(encoding="utf-8"))
    split_ids = {name: {int(value) for value in values} for name, values in splits["episodes"].items()}
    all_split_ids = set().union(*split_ids.values())
    if sum(len(values) for values in split_ids.values()) != len(all_split_ids):
        raise ValueError("Train/validation/test splits overlap")

    index_ids = {int(row["episode_id"]) for row in rows}
    if len(index_ids) != len(rows):
        raise ValueError("index.csv contains duplicate episode IDs")
    if index_ids != all_split_ids:
        raise ValueError("Split coverage does not match index.csv")

    total_transitions = 0
    checked_shards: set[str] = set()
    for row in rows:
        episode_id = int(row["episode_id"])
        if episode_id not in split_ids[row["split"]]:
            raise ValueError(f"Episode {episode_id} has inconsistent split metadata")
        shard_path = export_dir / row["shard"]
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        checked_shards.add(row["shard"])
        raw_arrays: dict[str, np.ndarray] | None = None
        raw_camera_arrays: dict[str, np.ndarray] | None = None
        source_episode_path: Path | None = None
        if compare_source:
            if source_path is None:
                raise ValueError("source_dir is required when compare_source=True")
            source_episode_path = source_path / row["source_episode_path"]
            if _sha256(source_episode_path / "trajectory.npz") != row["source_trajectory_sha256"]:
                raise ValueError(f"Episode {episode_id}: source trajectory checksum changed")
            with np.load(source_episode_path / "trajectory.npz") as raw:
                raw_arrays = {name: raw[name] for name in TENSOR_SCHEMA}
            if row.get("camera_data", "False").lower() == "true":
                source_camera_path = source_episode_path / "camera.npz"
                if _sha256(source_camera_path) != row["source_camera_sha256"]:
                    raise ValueError(f"Episode {episode_id}: source camera checksum changed")
                with np.load(source_camera_path) as raw_camera:
                    raw_camera_arrays = {
                        name: raw_camera[name] for name in CAMERA_TENSOR_SCHEMA
                    }
        with h5py.File(shard_path, "r", swmr=True) as shard:
            group = shard[row["hdf5_group"]]
            metadata = json.loads(_read_utf8_dataset(group["metadata_json"]))
            steps = int(row["steps"])
            if int(metadata["steps"]) != steps:
                raise ValueError(f"Episode {episode_id}: metadata/index step mismatch")
            total_transitions += steps
            for tensor_name, schema in TENSOR_SCHEMA.items():
                if tensor_name not in group:
                    raise ValueError(f"Episode {episode_id}: missing tensor {tensor_name}")
                array = group[tensor_name][...]
                if array.shape[0] != steps:
                    raise ValueError(f"Episode {episode_id}: {tensor_name} length mismatch")
                if str(array.dtype) != schema["dtype"]:
                    raise ValueError(f"Episode {episode_id}: {tensor_name} dtype mismatch")
                if not np.isfinite(array).all():
                    raise ValueError(f"Episode {episode_id}: {tensor_name} contains non-finite values")
                if raw_arrays is not None and not np.array_equal(array, raw_arrays[tensor_name]):
                    raise ValueError(f"Episode {episode_id}: {tensor_name} differs from source")
            timestamps = group["timestamp"][...]
            if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.0):
                raise ValueError(f"Episode {episode_id}: timestamp is not strictly increasing")
            if bool(metadata.get("domain_randomization", False)):
                if "domain_params_json" not in group:
                    raise ValueError(f"Episode {episode_id}: missing domain_params_json")
                exported_domain = json.loads(_read_utf8_dataset(group["domain_params_json"]))
            else:
                exported_domain = None
            camera_expected = bool(metadata.get("camera_data", False))
            if camera_expected:
                if "camera" not in group:
                    raise ValueError(f"Episode {episode_id}: missing camera group")
                camera_group = group["camera"]
                expected_frames = int(metadata["camera_frames"])
                for tensor_name, schema in CAMERA_TENSOR_SCHEMA.items():
                    if tensor_name not in camera_group:
                        raise ValueError(
                            f"Episode {episode_id}: missing camera tensor {tensor_name}"
                        )
                    array = camera_group[tensor_name][...]
                    if array.shape[0] != expected_frames:
                        raise ValueError(
                            f"Episode {episode_id}: camera {tensor_name} length mismatch"
                        )
                    if str(array.dtype) != schema["dtype"]:
                        raise ValueError(
                            f"Episode {episode_id}: camera {tensor_name} dtype mismatch"
                        )
                    if raw_camera_arrays is not None and not np.array_equal(
                        array, raw_camera_arrays[tensor_name]
                    ):
                        raise ValueError(
                            f"Episode {episode_id}: camera {tensor_name} differs from source"
                        )
            elif "camera" in group:
                raise ValueError(f"Episode {episode_id}: unexpected camera group")
            if source_episode_path is not None:
                source_metadata = json.loads(
                    (source_episode_path / "metadata.json").read_text(encoding="utf-8")
                )
                if metadata != source_metadata:
                    raise ValueError(f"Episode {episode_id}: metadata differs from source")
                domain_file = source_metadata.get("domain_params_file")
                source_domain = (
                    json.loads((source_episode_path / domain_file).read_text(encoding="utf-8"))
                    if domain_file
                    else None
                )
                if exported_domain != source_domain:
                    raise ValueError(f"Episode {episode_id}: domain parameters differ from source")

    dataset_info = json.loads((export_dir / "dataset_info.json").read_text(encoding="utf-8"))
    shard_episode_ids: set[int] = set()
    for shard_info in dataset_info["shards"]:
        shard_path = export_dir / shard_info["path"]
        if _sha256(shard_path) != shard_info["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {shard_info['path']}")
        with h5py.File(shard_path, "r", swmr=True) as shard:
            for group in shard["episodes"].values():
                episode_id = int(group.attrs["episode_id"])
                if episode_id in shard_episode_ids:
                    raise ValueError(f"Episode {episode_id} is duplicated across shards")
                shard_episode_ids.add(episode_id)
    if shard_episode_ids != index_ids:
        raise ValueError("HDF5 shard episode coverage does not match index.csv")

    return {
        "status": "DATASET_EXPORT_VALIDATION_OK",
        "validated_at": _utc_now(),
        "num_episodes": len(rows),
        "num_shards": len(checked_shards),
        "total_transitions": total_transitions,
        "source_value_comparison": bool(compare_source),
        "split_counts": {name: len(values) for name, values in split_ids.items()},
        "validation_wall_time_s": time.perf_counter() - started,
    }


def export_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    episodes_per_shard: int = 64,
    compression: str = "gzip",
    compression_level: int = 4,
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    split_seed: int = 42,
    resume: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """Export a raw dataset through a resumable staging directory."""

    started = time.perf_counter()
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if episodes_per_shard < 1:
        raise ValueError("episodes_per_shard must be at least 1")
    if source_dir == output_dir:
        raise ValueError("Source and output directories must differ")

    if output_dir.exists():
        if resume and (output_dir / "_SUCCESS").is_file():
            report = validate_export(output_dir, source_dir, compare_source=validate)
            report["resumed_existing_export"] = True
            return report
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    manifest, records = _load_raw_episodes(source_dir)
    source_manifest_path = source_dir / "dataset_metadata.json"
    splits = make_splits(records, split_ratios, split_seed)
    split_lookup = _split_lookup(splits)
    if set(split_lookup) != {record.episode_id for record in records}:
        raise ValueError("Split generation did not cover every source episode")

    staging_dir = output_dir.parent / f".{output_dir.name}.staging"
    state_path = staging_dir / "export_state.json"
    settings = {
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "output_dir": str(output_dir),
        "episodes_per_shard": episodes_per_shard,
        "compression": compression,
        "compression_level": compression_level,
        "split_ratios": list(split_ratios),
        "split_seed": split_seed,
    }
    if staging_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Incomplete staging export exists: {staging_dir}. Re-run with --resume."
            )
        if not state_path.is_file():
            raise ValueError(f"Staging directory has no export_state.json: {staging_dir}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["settings"] != settings:
            raise ValueError("Resume settings or source manifest differ from the staged export")
    else:
        staging_dir.mkdir(parents=True)
        (staging_dir / "shards").mkdir()
        state = {
            "status": "in_progress",
            "created_at": _utc_now(),
            "settings": settings,
            "completed_shards": [],
        }
        _write_json(state_path, state)

    num_shards = math.ceil(len(records) / episodes_per_shard)
    for shard_index in range(num_shards):
        shard_name = f"shard_{shard_index:05d}.h5"
        shard_records = records[
            shard_index * episodes_per_shard : (shard_index + 1) * episodes_per_shard
        ]
        if shard_index in state["completed_shards"]:
            if not (staging_dir / "shards" / shard_name).is_file():
                raise FileNotFoundError(f"Completed shard is missing: {shard_name}")
            continue
        _write_shard(
            staging_dir / "shards" / shard_name,
            shard_index,
            shard_records,
            source_dir,
            manifest,
            compression,
            compression_level,
        )
        state["completed_shards"].append(shard_index)
        _write_json(state_path, state)

    index_rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        shard_name = f"shard_{record_index // episodes_per_shard:05d}.h5"
        index_rows.append(_index_row(record, split_lookup[record.episode_id], shard_name, source_dir))
    _write_index(staging_dir / "index.csv", index_rows)
    split_payload = {
        "seed": split_seed,
        "ratios": {name: ratio for name, ratio in zip(splits, split_ratios, strict=True)},
        "counts": {name: len(values) for name, values in splits.items()},
        "stratified_by": "success",
        "episodes": splits,
    }
    _write_json(staging_dir / "splits.json", split_payload)

    lengths = np.asarray([int(record.metadata["steps"]) for record in records])
    successes = np.asarray([bool(record.metadata["success"]) for record in records])
    source_size = sum(
        path.stat().st_size
        for record in records
        for path in record.path.iterdir()
        if path.is_file()
    )
    shard_paths = sorted((staging_dir / "shards").glob("shard_*.h5"))
    shard_checksums = [
        {"path": f"shards/{path.name}", "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in shard_paths
    ]
    exported_size = sum(item["size_bytes"] for item in shard_checksums)
    domain_records = [record.domain_params for record in records if record.domain_params is not None]
    observed_domain: dict[str, Any] | None = None
    if domain_records:
        def observed_range(path: tuple[str, ...]) -> list[float]:
            values: list[float] = []
            for item in domain_records:
                value: Any = item
                for key in path:
                    value = value[key]
                values.append(float(value))
            return [min(values), max(values)]

        observed_domain = {
            "object_x_m": observed_range(("object", "position_m", 0)),
            "object_y_m": observed_range(("object", "position_m", 1)),
            "object_yaw_deg": observed_range(("object", "yaw_deg")),
            "object_mass_kg": observed_range(("object", "mass_kg")),
            "static_friction": observed_range(("object", "static_friction")),
            "dynamic_friction": observed_range(("object", "dynamic_friction")),
        }

    dataset_info = {
        "export_format_version": EXPORT_FORMAT_VERSION,
        "raw_format_version": manifest.get("format_version", RAW_FORMAT_VERSION),
        "created_at": _utc_now(),
        "source_dataset": str(source_dir),
        "source_manifest_sha256": settings["source_manifest_sha256"],
        "storage": {
            "format": "HDF5",
            "episodes_per_shard": episodes_per_shard,
            "compression": compression,
            "compression_level": compression_level if compression == "gzip" else None,
            "num_shards": len(shard_paths),
            "source_size_bytes": source_size,
            "exported_shard_size_bytes": exported_size,
            "shard_to_source_size_ratio": exported_size / source_size if source_size else None,
        },
        "statistics": {
            "num_episodes": len(records),
            "num_success": int(successes.sum()),
            "num_failure": int((~successes).sum()),
            "success_rate": float(successes.mean()),
            "total_transitions": int(lengths.sum()),
            "episode_steps_min": int(lengths.min()),
            "episode_steps_max": int(lengths.max()),
            "episode_steps_mean": float(lengths.mean()),
            "num_camera_episodes": sum(
                bool(record.metadata.get("camera_data", False)) for record in records
            ),
            "total_camera_frames": sum(
                int(record.metadata.get("camera_frames", 0)) for record in records
            ),
        },
        "splits": split_payload["counts"],
        "tensor_schema": manifest.get("tensor_schema", TENSOR_SCHEMA),
        "camera_tensor_schema": manifest.get("camera_tensor_schema", CAMERA_TENSOR_SCHEMA),
        "coordinate_conventions": manifest.get("coordinate_conventions", {}),
        "observed_domain_parameters": observed_domain,
        "software_versions": {
            "isaaclab": _package_version("isaaclab"),
            "isaacsim": _package_version("isaacsim"),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "h5py": _package_version("h5py"),
        },
        "shards": shard_checksums,
    }
    _write_json(staging_dir / "dataset_info.json", dataset_info)

    if validate:
        validation_report = validate_export(staging_dir, source_dir, compare_source=True)
    else:
        validation_report = {
            "status": "DATASET_EXPORT_VALIDATION_SKIPPED",
            "num_episodes": len(records),
            "num_shards": len(shard_paths),
        }
    validation_report["export_wall_time_s"] = time.perf_counter() - started
    _write_json(staging_dir / "validation_report.json", validation_report)
    state["status"] = "complete"
    state["completed_at"] = _utc_now()
    _write_json(state_path, state)
    _atomic_write_text(staging_dir / "_SUCCESS", "DATASET_EXPORT_OK\n")
    os.replace(staging_dir, output_dir)

    return {
        "status": "PHASE6_EXPORT_OK",
        "output_dir": str(output_dir),
        "num_episodes": len(records),
        "num_shards": len(shard_paths),
        "total_transitions": int(lengths.sum()),
        "success_rate": float(successes.mean()),
        "split_counts": split_payload["counts"],
        "source_size_bytes": source_size,
        "exported_shard_size_bytes": exported_size,
        "validation": validation_report["status"],
        "export_wall_time_s": validation_report["export_wall_time_s"],
    }
