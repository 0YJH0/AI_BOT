"""Reproducible single-environment domain randomization for Phase 4."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _range(config: dict[str, Any], key: str) -> tuple[float, float]:
    value = config[key]
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be a two-element YAML list")
    low, high = float(value[0]), float(value[1])
    if high < low:
        raise ValueError(f"{key} has descending range: {value}")
    return low, high


class DomainRandomizer:
    """Sample, apply, and read back domain parameters for one environment."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.seed = int(self.config["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.sample_index = 0
        self._validate_config()

    def _validate_config(self) -> None:
        for section in ("object", "camera", "lighting"):
            if section not in self.config:
                raise ValueError(f"Missing randomization section: {section}")

        object_cfg = self.config["object"]
        for key in (
            "position_x_m",
            "position_y_m",
            "position_z_m",
            "yaw_deg",
            "mass_kg",
            "static_friction",
            "dynamic_friction_ratio",
            "restitution",
        ):
            _range(object_cfg, key)
        if _range(object_cfg, "mass_kg")[0] <= 0.0:
            raise ValueError("object.mass_kg must be positive")

        camera_cfg = self.config["camera"]
        for key in ("nominal_position_m", "nominal_target_m"):
            if len(camera_cfg[key]) != 3:
                raise ValueError(f"camera.{key} must have three elements")
        for key in (
            "position_offset_x_m",
            "position_offset_y_m",
            "position_offset_z_m",
            "target_offset_x_m",
            "target_offset_y_m",
            "target_offset_z_m",
        ):
            _range(camera_cfg, key)

        lighting_cfg = self.config["lighting"]
        for key in ("intensity", "color_r", "color_g", "color_b"):
            _range(lighting_cfg, key)

    def _uniform(self, section: str, key: str) -> float:
        low, high = _range(self.config[section], key)
        return float(self.rng.uniform(low, high)) if high > low else low

    def sample(self) -> dict[str, Any]:
        """Return the next JSON-serializable parameter sample."""

        self.sample_index += 1
        static_friction = self._uniform("object", "static_friction")
        dynamic_ratio = self._uniform("object", "dynamic_friction_ratio")
        camera_cfg = self.config["camera"]
        position = [
            float(camera_cfg["nominal_position_m"][0]) + self._uniform("camera", "position_offset_x_m"),
            float(camera_cfg["nominal_position_m"][1]) + self._uniform("camera", "position_offset_y_m"),
            float(camera_cfg["nominal_position_m"][2]) + self._uniform("camera", "position_offset_z_m"),
        ]
        target = [
            float(camera_cfg["nominal_target_m"][0]) + self._uniform("camera", "target_offset_x_m"),
            float(camera_cfg["nominal_target_m"][1]) + self._uniform("camera", "target_offset_y_m"),
            float(camera_cfg["nominal_target_m"][2]) + self._uniform("camera", "target_offset_z_m"),
        ]
        return {
            "randomization_seed": self.seed,
            "sample_index": self.sample_index,
            "config_file": str(self.config_path),
            "object": {
                "position_m": [
                    self._uniform("object", "position_x_m"),
                    self._uniform("object", "position_y_m"),
                    self._uniform("object", "position_z_m"),
                ],
                "yaw_deg": self._uniform("object", "yaw_deg"),
                "mass_kg": self._uniform("object", "mass_kg"),
                "static_friction": static_friction,
                "dynamic_friction": static_friction * dynamic_ratio,
                "restitution": self._uniform("object", "restitution"),
            },
            "camera": {"position_m": position, "target_m": target},
            "lighting": {
                "intensity": self._uniform("lighting", "intensity"),
                "color_rgb": [
                    self._uniform("lighting", "color_r"),
                    self._uniform("lighting", "color_g"),
                    self._uniform("lighting", "color_b"),
                ],
            },
        }

    def sample_batch(self, num_envs: int) -> list[dict[str, Any]]:
        """Sample per-environment parameters for one vectorized rollout batch.

        Object and camera parameters are independent.  The scene has one
        global DomeLight, so lighting is sampled once and shared explicitly.
        """

        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        samples = [self.sample() for _ in range(num_envs)]
        shared_lighting = deepcopy(samples[0]["lighting"])
        shared_lighting["shared_across_batch"] = True
        for sample in samples:
            sample["lighting"] = deepcopy(shared_lighting)
        return samples

    def apply(self, env: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Apply one sample to a single-environment simulation."""

        if env.num_envs != 1:
            raise ValueError("apply() requires one environment; use apply_batch() for vectorized scenes")
        return self.apply_batch(env, [params])[0]

    def apply_batch(self, env: Any, params: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply one sample per environment and attach PhysX/USD read-backs."""

        import torch
        from pxr import Gf, UsdGeom, UsdLux

        if len(params) != env.num_envs:
            raise ValueError(f"Expected {env.num_envs} parameter samples, got {len(params)}")

        applied = deepcopy(params)
        env_ids_cpu = torch.arange(env.num_envs, dtype=torch.int32, device="cpu")
        target_object = env.scene["object"]
        object_cfgs = [sample["object"] for sample in params]

        yaw = torch.deg2rad(
            torch.tensor([cfg["yaw_deg"] for cfg in object_cfgs], dtype=torch.float32, device=env.device)
        )
        position = torch.tensor(
            [cfg["position_m"] for cfg in object_cfgs], dtype=torch.float32, device=env.device
        )
        position += env.scene.env_origins
        orientation = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
        orientation[:, 0] = torch.cos(yaw / 2.0)
        orientation[:, 3] = torch.sin(yaw / 2.0)
        target_object.write_root_pose_to_sim(torch.cat((position, orientation), dim=-1))
        target_object.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))

        masses = target_object.root_physx_view.get_masses()
        default_mass = target_object.data.default_mass.clone()
        masses[:, 0] = torch.tensor(
            [cfg["mass_kg"] for cfg in object_cfgs], dtype=masses.dtype, device=masses.device
        )
        target_object.root_physx_view.set_masses(masses, env_ids_cpu)
        inertias = target_object.root_physx_view.get_inertias()
        mass_ratio = masses / default_mass
        inertias[:] = target_object.data.default_inertia * mass_ratio
        target_object.root_physx_view.set_inertias(inertias, env_ids_cpu)

        materials = target_object.root_physx_view.get_material_properties()
        materials[:, :, 0] = torch.tensor(
            [cfg["static_friction"] for cfg in object_cfgs],
            dtype=materials.dtype,
            device=materials.device,
        ).unsqueeze(-1)
        materials[:, :, 1] = torch.tensor(
            [cfg["dynamic_friction"] for cfg in object_cfgs],
            dtype=materials.dtype,
            device=materials.device,
        ).unsqueeze(-1)
        materials[:, :, 2] = torch.tensor(
            [cfg["restitution"] for cfg in object_cfgs],
            dtype=materials.dtype,
            device=materials.device,
        ).unsqueeze(-1)
        target_object.root_physx_view.set_material_properties(materials, env_ids_cpu)

        camera_applied = "table_camera" in env.scene.sensors
        if camera_applied:
            camera = env.scene["table_camera"]
            eye = torch.tensor(
                [sample["camera"]["position_m"] for sample in params], device=env.device
            )
            target = torch.tensor(
                [sample["camera"]["target_m"] for sample in params], device=env.device
            )
            camera.set_world_poses_from_view(env.scene.env_origins + eye, env.scene.env_origins + target)

        dome_light = UsdLux.DomeLight.Get(env.sim.stage, "/World/Light")
        if not dome_light:
            raise RuntimeError("Expected DomeLight at /World/Light")
        light_cfg = params[0]["lighting"]
        dome_light.GetIntensityAttr().Set(float(light_cfg["intensity"]))
        dome_light.GetColorAttr().Set(Gf.Vec3f(*[float(value) for value in light_cfg["color_rgb"]]))

        env.sim.forward()
        if camera_applied:
            # Camera.reset() explicitly recomputes the pose buffer after pose
            # randomization; a regular update only advances sensor timestamps.
            camera.reset()
        object_positions = target_object.data.root_pos_w.detach().cpu()
        object_quaternions = target_object.data.root_quat_w.detach().cpu()
        applied_masses = target_object.root_physx_view.get_masses().detach().cpu()
        applied_materials = target_object.root_physx_view.get_material_properties().detach().cpu()
        light_intensity = float(dome_light.GetIntensityAttr().Get())
        light_color = [float(value) for value in dome_light.GetColorAttr().Get()]
        for env_index, sample in enumerate(applied):
            sample["camera"]["applied"] = camera_applied
            sample["applied_values"] = {
                "object_position_w_m": object_positions[env_index].tolist(),
                "object_quaternion_wxyz": object_quaternions[env_index].tolist(),
                "object_mass_kg": float(applied_masses[env_index, 0].item()),
                "object_material": applied_materials[env_index, 0].tolist(),
                "light_intensity": light_intensity,
                "light_color_rgb": light_color,
            }
        if camera_applied:
            xform_cache = UsdGeom.XformCache()
            origins_cpu = env.scene.env_origins.detach().cpu().numpy()
            for env_index, sample in enumerate(applied):
                prim_path = f"/World/envs/env_{env_index}/TableCamera"
                camera_prim = env.sim.stage.GetPrimAtPath(prim_path)
                if not camera_prim or not camera_prim.IsA(UsdGeom.Camera):
                    raise RuntimeError(f"Expected a Camera prim at {prim_path}")
                camera_transform = xform_cache.GetLocalToWorldTransform(camera_prim)
                camera_translation = camera_transform.ExtractTranslation()
                expected_translation = origins_cpu[env_index] + np.asarray(
                    params[env_index]["camera"]["position_m"], dtype=np.float64
                )
                actual_translation = np.asarray(camera_translation, dtype=np.float64)
                if not np.allclose(actual_translation, expected_translation, atol=1.0e-5, rtol=0.0):
                    raise RuntimeError(
                        "Camera USD pose read-back does not match the randomized command: "
                        f"expected {expected_translation.tolist()}, got {actual_translation.tolist()}"
                    )
                sample["applied_values"]["camera_position_w_m"] = actual_translation.tolist()
                sample["applied_values"]["camera_transform_w"] = [
                    [float(camera_transform[row][column]) for column in range(4)] for row in range(4)
                ]
                sample["applied_values"]["camera_readback_source"] = "USD local-to-world transform"
        return applied
