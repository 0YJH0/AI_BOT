"""Versioned, self-describing episode serialization for collected episodes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from isaac_lab_data_engine.geometry import invert_transform, pose_to_matrix, relative_pose


FORMAT_VERSION = "0.2.0"
TENSOR_SCHEMA: dict[str, dict[str, Any]] = {
    "q": {"shape": ["T", 9], "dtype": "float32", "description": "robot joint position"},
    "dq": {"shape": ["T", 9], "dtype": "float32", "description": "robot joint velocity"},
    "eef_pose": {
        "shape": ["T", 7],
        "dtype": "float32",
        "description": "end-effector pose in robot root frame; xyz + quaternion wxyz",
    },
    "object_pose": {
        "shape": ["T", 7],
        "dtype": "float32",
        "description": "object pose in robot root frame; xyz + quaternion wxyz",
    },
    "action": {
        "shape": ["T", 8],
        "dtype": "float32",
        "description": "absolute IK pose target (xyz + quaternion wxyz) + gripper command",
    },
    "reward": {"shape": ["T"], "dtype": "float32", "description": "manager environment reward"},
    "timestamp": {"shape": ["T"], "dtype": "float64", "description": "simulation time in seconds"},
    "controller_state": {
        "shape": ["T"],
        "dtype": "int64",
        "description": "scripted pick-and-lift state enum",
    },
}

CAMERA_TENSOR_SCHEMA: dict[str, dict[str, Any]] = {
    "frame_step": {"shape": ["F"], "dtype": "int64", "description": "1-based control step"},
    "timestamp": {"shape": ["F"], "dtype": "float64", "description": "simulation time in seconds"},
    "rgb": {"shape": ["F", "H", "W", 3], "dtype": "uint8", "description": "RGB image"},
    "depth": {"shape": ["F", "H", "W"], "dtype": "float32", "description": "distance to image plane in meters"},
    "camera_intrinsics": {"shape": ["F", 3, 3], "dtype": "float32", "description": "pinhole intrinsic matrix"},
    "camera_pose_w": {"shape": ["F", 7], "dtype": "float32", "description": "ROS optical camera pose in world; xyz + quaternion wxyz"},
    "camera_extrinsics_w2c": {"shape": ["F", 4, 4], "dtype": "float32", "description": "world-to-ROS-camera transform"},
    "object_pose_w": {"shape": ["F", 7], "dtype": "float32", "description": "object pose in world"},
    "object_pose_camera": {"shape": ["F", 7], "dtype": "float32", "description": "object pose in ROS optical camera frame"},
    "eef_pose_w": {"shape": ["F", 7], "dtype": "float32", "description": "end-effector pose in world"},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeBuffer:
    """Accumulate one single-environment trajectory in host memory."""

    def __init__(self) -> None:
        self._samples: dict[str, list[np.ndarray | float | int]] = {name: [] for name in TENSOR_SCHEMA}

    def __len__(self) -> int:
        return len(self._samples["timestamp"])

    def append(
        self,
        *,
        q: Any,
        dq: Any,
        eef_pose: Any,
        object_pose: Any,
        action: Any,
        reward: float,
        timestamp: float,
        controller_state: int,
    ) -> None:
        """Append one post-step state and the action that produced it."""

        self._samples["q"].append(np.asarray(q, dtype=np.float32).copy())
        self._samples["dq"].append(np.asarray(dq, dtype=np.float32).copy())
        self._samples["eef_pose"].append(np.asarray(eef_pose, dtype=np.float32).copy())
        self._samples["object_pose"].append(np.asarray(object_pose, dtype=np.float32).copy())
        self._samples["action"].append(np.asarray(action, dtype=np.float32).copy())
        self._samples["reward"].append(float(reward))
        self._samples["timestamp"].append(float(timestamp))
        self._samples["controller_state"].append(int(controller_state))

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Stack samples and enforce the public on-disk dtypes."""

        if not self:
            raise ValueError("Cannot serialize an empty episode.")
        return {
            "q": np.stack(self._samples["q"]).astype(np.float32, copy=False),
            "dq": np.stack(self._samples["dq"]).astype(np.float32, copy=False),
            "eef_pose": np.stack(self._samples["eef_pose"]).astype(np.float32, copy=False),
            "object_pose": np.stack(self._samples["object_pose"]).astype(np.float32, copy=False),
            "action": np.stack(self._samples["action"]).astype(np.float32, copy=False),
            "reward": np.asarray(self._samples["reward"], dtype=np.float32),
            "timestamp": np.asarray(self._samples["timestamp"], dtype=np.float64),
            "controller_state": np.asarray(self._samples["controller_state"], dtype=np.int64),
        }


class CameraFrameBuffer:
    """Accumulate synchronized RGB-D, calibration, and 6DoF labels."""

    def __init__(self) -> None:
        self._samples: dict[str, list[np.ndarray | float | int]] = {
            name: [] for name in CAMERA_TENSOR_SCHEMA
        }

    def __len__(self) -> int:
        return len(self._samples["frame_step"])

    def append(
        self,
        *,
        frame_step: int,
        timestamp: float,
        rgb: Any,
        depth: Any,
        camera_intrinsics: Any,
        camera_pose_w: Any,
        object_pose_w: Any,
        eef_pose_w: Any,
    ) -> None:
        rgb_array = np.asarray(rgb, dtype=np.uint8)
        depth_array = np.asarray(depth, dtype=np.float32)
        if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
            depth_array = depth_array[..., 0]
        if rgb_array.ndim != 3 or rgb_array.shape[-1] != 3:
            raise ValueError(f"Expected RGB [H,W,3], got {rgb_array.shape}")
        if depth_array.shape != rgb_array.shape[:2]:
            raise ValueError(f"Depth/RGB resolution mismatch: {depth_array.shape}, {rgb_array.shape}")
        camera_pose = np.asarray(camera_pose_w, dtype=np.float32)
        object_pose = np.asarray(object_pose_w, dtype=np.float32)
        transform_camera_world = invert_transform(pose_to_matrix(camera_pose))
        object_pose_camera = relative_pose(camera_pose, object_pose)

        self._samples["frame_step"].append(int(frame_step))
        self._samples["timestamp"].append(float(timestamp))
        self._samples["rgb"].append(rgb_array.copy())
        self._samples["depth"].append(depth_array.copy())
        self._samples["camera_intrinsics"].append(
            np.asarray(camera_intrinsics, dtype=np.float32).copy()
        )
        self._samples["camera_pose_w"].append(camera_pose.copy())
        self._samples["camera_extrinsics_w2c"].append(
            transform_camera_world.astype(np.float32)
        )
        self._samples["object_pose_w"].append(object_pose.copy())
        self._samples["object_pose_camera"].append(object_pose_camera.astype(np.float32))
        self._samples["eef_pose_w"].append(np.asarray(eef_pose_w, dtype=np.float32).copy())

    def as_arrays(self) -> dict[str, np.ndarray]:
        if not self:
            raise ValueError("Cannot serialize an empty camera buffer")
        arrays = {
            name: np.asarray(values, dtype=np.dtype(schema["dtype"]))
            for (name, values), schema in zip(
                self._samples.items(), CAMERA_TENSOR_SCHEMA.values(), strict=True
            )
        }
        return arrays


def _save_camera_previews(episode_dir: Path, arrays: dict[str, np.ndarray]) -> list[str]:
    from PIL import Image

    rgb_path = episode_dir / "camera_rgb_preview.png"
    depth_path = episode_dir / "camera_depth_preview.png"
    Image.fromarray(arrays["rgb"][-1]).save(rgb_path)
    depth = arrays["depth"][-1]
    finite = np.isfinite(depth)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        low, high = np.percentile(depth[finite], [1.0, 99.0])
        normalized = 1.0 - np.clip((depth[finite] - low) / max(high - low, 1.0e-6), 0.0, 1.0)
        preview[finite] = np.round(normalized * 255.0).astype(np.uint8)
    Image.fromarray(preview).save(depth_path)
    return [rgb_path.name, depth_path.name]


class EpisodeWriter:
    """Write episode folders and maintain the dataset-level manifest."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def next_episode_id(self) -> int:
        ids: list[int] = []
        for path in self.output_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"):
            try:
                ids.append(int(path.name.removeprefix("episode_")))
            except ValueError:
                continue
        return max(ids, default=0) + 1

    def write(
        self,
        episode_id: int,
        buffer: EpisodeBuffer,
        metadata: dict[str, Any],
        sidecar_json: dict[str, dict[str, Any]] | None = None,
        camera_buffer: CameraFrameBuffer | None = None,
    ) -> Path:
        """Persist a new episode without overwriting an existing one."""

        episode_dir = self.output_dir / f"episode_{episode_id:06d}"
        episode_dir.mkdir(parents=False, exist_ok=False)

        arrays = buffer.as_arrays()
        trajectory_path = episode_dir / "trajectory.npz"
        np.savez_compressed(trajectory_path, **arrays)

        camera_arrays: dict[str, np.ndarray] | None = None
        if camera_buffer is not None:
            camera_arrays = camera_buffer.as_arrays()
            np.savez_compressed(episode_dir / "camera.npz", **camera_arrays)
            preview_files = _save_camera_previews(episode_dir, camera_arrays)
            metadata = {
                **metadata,
                "camera_data": True,
                "camera_file": "camera.npz",
                "camera_frames": len(camera_buffer),
                "camera_resolution": [
                    int(camera_arrays["rgb"].shape[1]),
                    int(camera_arrays["rgb"].shape[2]),
                ],
                "camera_preview_files": preview_files,
            }

        episode_metadata = {
            "format_version": FORMAT_VERSION,
            "episode_id": episode_id,
            "created_at": _utc_now(),
            **metadata,
            "tensor_shapes": {name: list(array.shape) for name, array in arrays.items()},
            "tensor_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        }
        if camera_arrays is not None:
            episode_metadata["camera_tensor_shapes"] = {
                name: list(array.shape) for name, array in camera_arrays.items()
            }
            episode_metadata["camera_tensor_dtypes"] = {
                name: str(array.dtype) for name, array in camera_arrays.items()
            }
        metadata_path = episode_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(episode_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for file_name, payload in (sidecar_json or {}).items():
            if Path(file_name).name != file_name or not file_name.endswith(".json"):
                raise ValueError(f"Invalid sidecar JSON filename: {file_name}")
            (episode_dir / file_name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self._update_manifest(episode_dir, episode_metadata)
        return episode_dir

    def _update_manifest(self, episode_dir: Path, metadata: dict[str, Any]) -> None:
        manifest_path = self.output_dir / "dataset_metadata.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {
                "format_version": FORMAT_VERSION,
                "created_at": _utc_now(),
                "tensor_schema": TENSOR_SCHEMA,
                "camera_tensor_schema": CAMERA_TENSOR_SCHEMA,
                "coordinate_conventions": {
                    "pose_order": "x, y, z, qw, qx, qy, qz",
                    "eef_pose_frame": "robot_root",
                    "object_pose_frame": "robot_root",
                    "camera_pose_frame": "world",
                    "camera_quaternion_convention": "ROS optical axes; wxyz",
                    "camera_extrinsics": "world_to_camera",
                },
                "episodes": [],
            }

        manifest["episodes"].append(
            {
                "episode_id": metadata["episode_id"],
                "path": episode_dir.name,
                "success": metadata["success"],
                "failure_reason": metadata["failure_reason"],
                "steps": metadata["steps"],
                "completion_time_s": metadata["completion_time_s"],
                "total_reward": metadata["total_reward"],
                "parallel_batch_index": metadata.get("parallel_batch_index"),
                "parallel_env_index": metadata.get("parallel_env_index"),
                "parallel_num_envs": metadata.get("parallel_num_envs", 1),
                "domain_randomization": bool(metadata.get("domain_randomization", False)),
                "domain_params_file": metadata.get("domain_params_file"),
                "camera_data": bool(metadata.get("camera_data", False)),
                "camera_file": metadata.get("camera_file"),
                "camera_frames": metadata.get("camera_frames", 0),
            }
        )
        manifest["episodes"].sort(key=lambda item: item["episode_id"])
        manifest["num_episodes"] = len(manifest["episodes"])
        manifest["num_success"] = sum(bool(item["success"]) for item in manifest["episodes"])
        manifest["num_failure"] = manifest["num_episodes"] - manifest["num_success"]
        manifest["updated_at"] = _utc_now()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_episode(episode_dir: str | Path) -> dict[str, Any]:
    """Load an episode and validate keys, shapes, dtypes, finiteness, and time order."""

    episode_dir = Path(episode_dir)
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "trajectory.npz") as data:
        missing = set(TENSOR_SCHEMA) - set(data.files)
        if missing:
            raise ValueError(f"Missing trajectory tensors: {sorted(missing)}")
        arrays = {name: data[name] for name in TENSOR_SCHEMA}

    steps = int(metadata["steps"])
    expected_tail_shapes = {
        "q": (9,),
        "dq": (9,),
        "eef_pose": (7,),
        "object_pose": (7,),
        "action": (8,),
        "reward": (),
        "timestamp": (),
        "controller_state": (),
    }
    for name, array in arrays.items():
        expected_shape = (steps, *expected_tail_shapes[name])
        if array.shape != expected_shape:
            raise ValueError(f"{name}: expected shape {expected_shape}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        if str(array.dtype) != TENSOR_SCHEMA[name]["dtype"]:
            raise ValueError(f"{name}: expected dtype {TENSOR_SCHEMA[name]['dtype']}, got {array.dtype}")

    if steps > 1 and not np.all(np.diff(arrays["timestamp"]) > 0.0):
        raise ValueError("timestamp must be strictly increasing")

    domain_params_file = metadata.get("domain_params_file")
    if metadata.get("domain_randomization", False):
        if not domain_params_file:
            raise ValueError("Randomized episode does not declare domain_params_file")
        domain_params_path = episode_dir / domain_params_file
        if not domain_params_path.is_file():
            raise ValueError(f"Missing domain randomization sidecar: {domain_params_file}")
        domain_params = json.loads(domain_params_path.read_text(encoding="utf-8"))
        for section in ("object", "camera", "lighting", "applied_values"):
            if section not in domain_params:
                raise ValueError(f"domain_params.json is missing section: {section}")

    camera_shapes = None
    finite_depth_fraction = None
    if metadata.get("camera_data", False):
        camera_file = metadata.get("camera_file")
        if not camera_file or Path(camera_file).name != camera_file:
            raise ValueError("Camera episode does not declare a safe camera_file")
        camera_path = episode_dir / camera_file
        if not camera_path.is_file():
            raise FileNotFoundError(f"Missing camera data: {camera_file}")
        with np.load(camera_path) as camera_data:
            missing = set(CAMERA_TENSOR_SCHEMA) - set(camera_data.files)
            if missing:
                raise ValueError(f"Missing camera tensors: {sorted(missing)}")
            camera_arrays = {name: camera_data[name] for name in CAMERA_TENSOR_SCHEMA}
        frames = int(metadata["camera_frames"])
        height, width = [int(value) for value in metadata["camera_resolution"]]
        expected_shapes = {
            "frame_step": (frames,),
            "timestamp": (frames,),
            "rgb": (frames, height, width, 3),
            "depth": (frames, height, width),
            "camera_intrinsics": (frames, 3, 3),
            "camera_pose_w": (frames, 7),
            "camera_extrinsics_w2c": (frames, 4, 4),
            "object_pose_w": (frames, 7),
            "object_pose_camera": (frames, 7),
            "eef_pose_w": (frames, 7),
        }
        for name, array in camera_arrays.items():
            if array.shape != expected_shapes[name]:
                raise ValueError(f"{name}: expected shape {expected_shapes[name]}, got {array.shape}")
            if str(array.dtype) != CAMERA_TENSOR_SCHEMA[name]["dtype"]:
                raise ValueError(
                    f"{name}: expected dtype {CAMERA_TENSOR_SCHEMA[name]['dtype']}, got {array.dtype}"
                )
        if frames < 1:
            raise ValueError("Camera episode contains no frames")
        if not np.all(np.diff(camera_arrays["frame_step"]) > 0):
            raise ValueError("Camera frame_step must be strictly increasing")
        if camera_arrays["frame_step"][-1] > steps:
            raise ValueError("Camera frame_step exceeds trajectory length")
        if frames > 1 and not np.all(np.diff(camera_arrays["timestamp"]) > 0.0):
            raise ValueError("Camera timestamp must be strictly increasing")
        for name in (
            "camera_intrinsics",
            "camera_pose_w",
            "camera_extrinsics_w2c",
            "object_pose_w",
            "object_pose_camera",
            "eef_pose_w",
        ):
            if not np.isfinite(camera_arrays[name]).all():
                raise ValueError(f"{name} contains non-finite values")
        depth = camera_arrays["depth"]
        if np.isnan(depth).any() or (depth[np.isfinite(depth)] <= 0.0).any():
            raise ValueError("Depth must contain positive finite meters or +inf background")
        finite_depth_fraction = float(np.isfinite(depth).mean())
        if finite_depth_fraction == 0.0:
            raise ValueError("Depth has no finite pixels")
        for index in range(frames):
            expected_extrinsic = invert_transform(
                pose_to_matrix(camera_arrays["camera_pose_w"][index])
            )
            if not np.allclose(
                camera_arrays["camera_extrinsics_w2c"][index],
                expected_extrinsic,
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise ValueError("camera_extrinsics_w2c is inconsistent with camera_pose_w")
            expected_object = relative_pose(
                camera_arrays["camera_pose_w"][index],
                camera_arrays["object_pose_w"][index],
            )
            if not np.allclose(
                pose_to_matrix(camera_arrays["object_pose_camera"][index]),
                pose_to_matrix(expected_object),
                atol=1.0e-5,
                rtol=0.0,
            ):
                raise ValueError("object_pose_camera is inconsistent with world poses")
        camera_shapes = {name: list(array.shape) for name, array in camera_arrays.items()}

    return {
        "status": "EPISODE_VALIDATION_OK",
        "episode_id": metadata["episode_id"],
        "steps": steps,
        "success": metadata["success"],
        "domain_randomization": bool(metadata.get("domain_randomization", False)),
        "tensor_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "tensor_dtypes": {name: str(array.dtype) for name, array in arrays.items()},
        "camera_data": bool(metadata.get("camera_data", False)),
        "camera_tensor_shapes": camera_shapes,
        "finite_depth_fraction": finite_depth_fraction,
    }
