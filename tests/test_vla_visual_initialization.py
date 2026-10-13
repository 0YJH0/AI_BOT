import numpy as np
import ast
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from isaac_lab_data_engine.vla.visual_initialization import ConsecutiveVisualWindow


def test_duplicate_frame_cannot_fill_or_break_window():
    w = ConsecutiveVisualWindow(3)
    assert not w.observe(1, [0, 0, 0], quality_valid=True, pose_valid=True)
    assert not w.observe(1, [0, 0, 0], quality_valid=False, pose_valid=False)
    assert w.consecutive == 1 and w.duplicate_frames == 1
    assert not w.observe(2, [0, 0, 0], quality_valid=True, pose_valid=True)
    assert w.observe(3, [0, 0, 0], quality_valid=True, pose_valid=True)


def test_invalid_new_frame_resets_real_consecutive_window():
    w = ConsecutiveVisualWindow(3)
    for i in range(20):
        assert not w.observe(i, [0, 0, 0], quality_valid=i % 2 == 0, pose_valid=True)
    assert w.valid_frames == 10 and w.maximum_consecutive == 1
    for i in (20, 21):
        assert not w.observe(i, [0, 0, 0], quality_valid=True, pose_valid=True)
    assert w.observe(22, [0, 0, 0], quality_valid=True, pose_valid=True)


def test_spread_and_nan_not_accepted():
    w = ConsecutiveVisualWindow(3, .01)
    for i, p in enumerate(([0, 0, 0], [.1, 0, 0], [0, 0, 0])):
        assert not w.observe(i, p, quality_valid=True, pose_valid=True)
    assert not w.observe(3, [np.nan, 0, 0], quality_valid=True, pose_valid=True)
    assert w.consecutive == 0


def test_initializer_keeps_real_physics_and_never_polls_zero_dt_buffers():
    # Extract only simulator-independent functions, avoiding AppLauncher import.
    path = Path(__file__).parents[1] / "scripts/run_vla_benchmark.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    selected = [n for n in parsed.body if isinstance(n, ast.FunctionDef)
                and n.name in ("_camera_snapshot", "_initialize_visual_position")]
    args = SimpleNamespace(visual_init_window=5, visual_init_max_spread_m=.025,
        camera_warmup_min_steps=10, camera_warmup_max_steps=30, camera_ready_timeout_s=3.,
        camera_refresh_renders=3, camera_render_yield_s=0., camera_min_rgb_std=30.,
        camera_min_rgb_range=128., camera_min_depth_fraction=.4, visual_invalid_pose_timeout_s=0.,
        proprio_only=False)
    pixels = torch.tensor(np.tile(np.array([0, 255], np.uint8), (1, 8, 4)))
    rgb = pixels[..., None].expand(1, 8, 8, 3).clone()
    data = SimpleNamespace(output={"rgb": rgb.clone(), "distance_to_image_plane": torch.ones(1, 8, 8)},
        intrinsic_matrices=torch.eye(3)[None], pos_w=torch.zeros(1, 3), quat_w_ros=torch.tensor([[1.,0,0,0]]))
    camera = SimpleNamespace(data=data, frame=torch.zeros(1, dtype=torch.int64), reset=lambda: None)
    state = {"steps": 0, "updates": 0}
    def update(dt, force_recompute):
        assert dt > 0  # Old renderer polling fails here.
        state["updates"] += 1
        camera.frame += 1
        data.output["rgb"] = rgb.clone() if state["steps"] >= 35 else torch.zeros_like(rgb)
    camera.update = update
    def step(action):
        state["steps"] += 1
        return None, None, torch.tensor([False]), torch.tensor([False]), None
    env = SimpleNamespace(device="cpu", step_dt=.02, physics_dt=.01,
        scene={"table_camera":camera}, sim=SimpleNamespace(forward=lambda:None,render=lambda:None),
        episode_length_buf=torch.zeros(1), step=step)
    namespace = {"torch":torch, "np":np, "time":time, "hashlib":hashlib, "Any":object,
        "args_cli":args, "ManagerBasedRLEnv":object, "RGBDPoseEstimator":object,
        "_poses_in_robot_frame":lambda e:(torch.tensor([[.4,0,.4,1,0,0,0]]),None),
        "_visual_position_b":lambda e,m,s:(torch.tensor([[.55,0,.03]]),torch.tensor([True]))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    position, valid, diagnostic = namespace["_initialize_visual_position"](env, None)
    assert valid.item() and diagnostic["stable"]
    assert diagnostic["steps"] == 39 and diagnostic["max_consecutive_valid_frames"] == 5
    assert diagnostic["invalid_frames"] == 25
    assert state["updates"] == state["steps"]  # Exactly one capture per control step.
