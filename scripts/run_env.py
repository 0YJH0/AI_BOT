"""Run and validate the Phase 1 Franka manipulation scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the Phase 1 Franka/table/object/RGB-D scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of scene copies.")
parser.add_argument(
    "--steps",
    type=int,
    default=-1,
    help="Physics steps before exit. Defaults to 120 in headless mode and unlimited in GUI mode.",
)
parser.add_argument(
    "--save_camera",
    action="store_true",
    help="Save one RGB image and one normalized depth preview under results/phase1.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# The Phase 1 acceptance criteria include an RGB-D sensor, including headless runs.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from isaac_lab_data_engine.envs.phase1_scene import Phase1SceneCfg, TABLE_TOP_Z


def _shape(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def _first(tensor: torch.Tensor) -> list[float]:
    return tensor[0].detach().cpu().tolist()


def _validate_and_summarize(scene: InteractiveScene, steps: int) -> dict[str, object]:
    """Validate required Phase 1 state/sensor channels and return a compact summary."""

    robot = scene.articulations["robot"]
    target_object = scene.rigid_objects["object"]
    eef_sensor = scene.sensors["end_effector_frame"]
    camera = scene.sensors["table_camera"]

    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    object_pose = torch.cat((target_object.data.root_pos_w, target_object.data.root_quat_w), dim=-1)
    eef_pose = torch.cat(
        (eef_sensor.data.target_pos_w[:, 0, :], eef_sensor.data.target_quat_w[:, 0, :]), dim=-1
    )
    rgb = camera.data.output["rgb"]
    depth = camera.data.output["distance_to_image_plane"]
    intrinsics = camera.data.intrinsic_matrices
    camera_pose = torch.cat((camera.data.pos_w, camera.data.quat_w_ros), dim=-1)

    num_envs = scene.num_envs
    assert joint_pos.shape == (num_envs, 9), f"Unexpected joint position shape: {joint_pos.shape}"
    assert joint_vel.shape == (num_envs, 9), f"Unexpected joint velocity shape: {joint_vel.shape}"
    assert object_pose.shape == (num_envs, 7), f"Unexpected object pose shape: {object_pose.shape}"
    assert eef_pose.shape == (num_envs, 7), f"Unexpected end-effector pose shape: {eef_pose.shape}"
    assert rgb.shape == (num_envs, 240, 320, 3), f"Unexpected RGB shape: {rgb.shape}"
    assert depth.shape == (num_envs, 240, 320, 1), f"Unexpected depth shape: {depth.shape}"
    assert intrinsics.shape == (num_envs, 3, 3), f"Unexpected intrinsics shape: {intrinsics.shape}"
    assert camera_pose.shape == (num_envs, 7), f"Unexpected camera pose shape: {camera_pose.shape}"

    assert torch.isfinite(joint_pos).all(), "Joint positions contain non-finite values."
    assert torch.isfinite(joint_vel).all(), "Joint velocities contain non-finite values."
    assert torch.isfinite(object_pose).all(), "Object pose contains non-finite values."
    assert torch.isfinite(eef_pose).all(), "End-effector pose contains non-finite values."
    assert torch.isfinite(intrinsics).all(), "Camera intrinsics contain non-finite values."
    assert torch.isfinite(depth).any(), "Depth image contains no finite pixels."
    assert target_object.data.root_pos_w[:, 2].min() >= TABLE_TOP_Z - 0.01, "Object fell through the table."

    return {
        "status": "PHASE1_VALIDATION_OK",
        "num_envs": num_envs,
        "steps": steps,
        "device": str(joint_pos.device),
        "joint_position_shape": _shape(joint_pos),
        "joint_velocity_shape": _shape(joint_vel),
        "end_effector_pose_shape": _shape(eef_pose),
        "object_pose_shape": _shape(object_pose),
        "rgb_shape": _shape(rgb),
        "depth_shape": _shape(depth),
        "camera_intrinsics_shape": _shape(intrinsics),
        "camera_pose_shape": _shape(camera_pose),
        "object_pose_env0": _first(object_pose),
        "end_effector_pose_env0": _first(eef_pose),
        "camera_intrinsics_env0": intrinsics[0].detach().cpu().tolist(),
        "finite_depth_fraction": float(torch.isfinite(depth).float().mean().item()),
        "rgb_mean": float(rgb.float().mean().item()),
    }


def _save_camera_preview(scene: InteractiveScene) -> list[str]:
    """Save one debug RGB frame and a normalized depth visualization."""

    from PIL import Image

    output_dir = Path(__file__).resolve().parents[1] / "results" / "phase1"
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = scene.sensors["table_camera"]
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].detach().cpu()

    rgb_path = output_dir / "camera_rgb.png"
    depth_path = output_dir / "camera_depth_preview.png"
    Image.fromarray(rgb).save(rgb_path)

    finite_mask = torch.isfinite(depth)
    finite_depth = depth[finite_mask]
    depth_preview = torch.zeros_like(depth)
    if finite_depth.numel() > 0:
        depth_min = finite_depth.min()
        depth_max = finite_depth.max()
        depth_preview[finite_mask] = 1.0 - (depth[finite_mask] - depth_min) / (depth_max - depth_min + 1.0e-6)
    Image.fromarray((depth_preview.clamp(0.0, 1.0) * 255).byte().numpy()).save(depth_path)

    return [str(rgb_path), str(depth_path)]


def main() -> None:
    """Create the scene, keep the robot at its default pose, and validate sensor data."""

    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1")

    max_steps = args_cli.steps
    if max_steps < 0 and args_cli.headless:
        max_steps = 120

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.6, -1.6, 1.4), target=(0.45, 0.0, 0.45))

    scene_cfg = Phase1SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, lazy_sensor_update=False)
    scene = InteractiveScene(scene_cfg)

    assert set(scene.articulations) == {"robot"}
    assert set(scene.rigid_objects) == {"object"}
    assert {"end_effector_frame", "table_camera"}.issubset(scene.sensors)
    assert {"ground", "table", "light"}.issubset(scene.extras)

    sim.reset()
    scene.reset()

    camera = scene.sensors["table_camera"]
    camera_eye_offset = torch.tensor((1.60, -2.00, 1.65), device=scene.env_origins.device).repeat(
        args_cli.num_envs, 1
    )
    camera_target_offset = torch.tensor((0.25, 0.0, 0.85), device=scene.env_origins.device).repeat(
        args_cli.num_envs, 1
    )
    camera.set_world_poses_from_view(
        scene.env_origins + camera_eye_offset,
        scene.env_origins + camera_target_offset,
    )

    robot = scene.articulations["robot"]
    joint_targets = robot.data.default_joint_pos.clone()
    sim_dt = sim.get_physics_dt()
    step_count = 0

    try:
        while simulation_app.is_running() and (max_steps < 0 or step_count < max_steps):
            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            step_count += 1

        summary = _validate_and_summarize(scene, step_count)
        if args_cli.save_camera:
            summary["camera_preview_files"] = _save_camera_preview(scene)
        print("PHASE1_SUMMARY=" + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
