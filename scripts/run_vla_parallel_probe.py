"""Synchronous tiled-camera VLA capacity probe; shared weights, independent slot state.

Reuses the established benchmark's pose transforms, visual estimator, phase FSM,
guard and success rules. This is a capacity test, not a paired success-rate claim.
Two reset waves exercise full-length control; completed slots hold their last
target while the full batch continues inference to measure sustained capacity.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

probe_parser = argparse.ArgumentParser(add_help=False)
probe_parser.add_argument('--parallel-envs', type=int, required=True)
probe_parser.add_argument('--waves', type=int, default=2)
probe_parser.add_argument('--strict-benchmark', action='store_true',
    help='Abort on camera failure rather than reporting a misleading grasp benchmark')
probe_parser.add_argument('--camera-mode', choices=('tiled', 'standard'), default='tiled')
probe_args, rest = probe_parser.parse_known_args()
if probe_args.parallel_envs not in (1, 2, 4, 8, 16) or probe_args.waves < 1:
    raise ValueError('Invalid capacity probe settings')
sys.argv = [sys.argv[0]] + rest
import run_vla_benchmark as base  # Launch Isaac once, reuse audited helpers.
import numpy as np
import torch
from isaaclab.sensors import TiledCameraCfg
from isaac_lab_data_engine.vla.rpc import VLABatchRequest
from isaac_lab_data_engine.vla.visual_initialization import ConsecutiveVisualWindow
from isaac_lab_data_engine.vla.batch_execution import BatchChunkExecutor


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def quality(snapshot):
    rgb = snapshot['rgb'][..., :3].float().flatten(1)
    depth = snapshot['depth'].flatten(1)
    std = rgb.std(dim=1, unbiased=False)
    span = rgb.max(dim=1).values - rgb.min(dim=1).values
    fraction = (torch.isfinite(depth) & (depth > 0)).float().mean(dim=1)
    valid = (std >= 30) & (span >= 128) & (fraction >= .4)
    return valid.cpu().numpy(), std.cpu().tolist(), fraction.cpu().tolist()


def main():
    args = base.args_cli
    n = probe_args.parallel_envs
    if (args.execution_horizon not in (1, 2, 4) or args.action_repeat != 5 or args.proprio_only
            or args.joint_target_actions or args.vla_input_interface != 'legacy'):
        raise ValueError('Probe supports the legacy 30-D delta interface, horizon=1/2/4 repeat=5')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    level = next(l for l in base.benchmark_cfg.levels if l.name == 'medium')
    randomizer = base.create_domain_randomizer(level.randomization_config)
    # Pre-sample independently of completion order. Same numbered samples at all scales.
    samples = [randomizer.sample() for _ in range(n * probe_args.waves)]
    save(out/'scene_manifest.json', dict(difficulty='medium', seed=randomizer.seed, samples=samples))
    cfg = base.PickLiftEnvCfg()
    cfg.scene.num_envs = n
    # 2.5 m puts a neighbouring table into the camera frustum. Separate scenes
    # without changing any robot-relative scene parameters or camera optics.
    cfg.scene.env_spacing = 6.0
    cfg.scene.lazy_sensor_update = True
    cfg.sim.device = args.device
    cfg.episode_length_s = (args.max_steps + 100) * cfg.sim.dt * cfg.decimation
    # Avoid manager auto-reset erasing terminal state; this probe owns wave resets.
    cfg.terminations.object_dropping = None
    camera = cfg.scene.table_camera
    if probe_args.camera_mode == 'tiled':
        cfg.scene.table_camera = TiledCameraCfg(prim_path=camera.prim_path,
            update_period=camera.update_period, height=camera.height, width=camera.width,
            data_types=list(camera.data_types), spawn=camera.spawn, offset=camera.offset,
            update_latest_camera_pose=True)
    base.configure_parallel_physx_capacity(cfg, n)
    env = base.ManagerBasedRLEnv(cfg)
    estimator = base.RGBDPoseEstimator.load(args.vision_checkpoint, env.device)
    client = base.VLAClient(args.vla_host, args.vla_port, args.vla_timeout)
    all_rows, wave_reports, latencies = [], [], []
    peak_allocated, peak_reserved = 0., 0.
    started = time.perf_counter()
    try:
        for wave in range(probe_args.waves):
            t0 = time.perf_counter()
            np.random.seed(level.seed)
            torch.manual_seed(level.seed)
            env.reset()
            domains = randomizer.apply_batch(env, samples[wave*n:(wave+1)*n])
            save(out / f'domains_wave{wave}.json', domains)
            ee, _ = base._poses_in_robot_frame(env)
            hold = torch.cat((ee, torch.ones((n, 1), device=env.device)), -1)
            windows = [ConsecutiveVisualWindow(5, .025) for _ in range(n)]
            ready = np.zeros(n, dtype=bool)
            visual = torch.tensor([[.55, 0, .03]]*n, device=env.device)
            ready_steps = [None]*n
            warmup_started = time.perf_counter()
            step = 0
            while not ready.all() and (step < 30 or time.perf_counter()-warmup_started < 180):
                env.episode_length_buf.zero_()
                env.step(hold)
                step += 1
                for _ in range(3): env.sim.render()
                time.sleep(.02)
                env.scene['table_camera'].update(env.step_dt, force_recompute=True)
                snap = base._camera_snapshot(env)
                if step < 10: continue
                qvalid, std, fractions = quality(snap)
                position, pvalid = base._visual_position_b(env, estimator, snap)
                frame_ids = env.scene['table_camera'].frame.cpu().tolist()
                for i in range(n):
                    if ready[i]: continue
                    ready[i] = windows[i].observe(frame_ids[i], position[i].cpu().numpy(),
                        quality_valid=bool(qvalid[i]), pose_valid=bool(pvalid[i]))
                    if ready[i]:
                        visual[i] = torch.as_tensor(windows[i].median, device=env.device)
                        ready_steps[i] = step
            env.episode_length_buf.zero_()
            ee, obj = base._poses_in_robot_frame(env)
            initial_z = env.scene['object'].data.root_pos_w[:, 2].clone()
            initial_error = torch.linalg.vector_norm(visual - obj[:, :3], dim=-1).cpu().tolist()
            camera_positions = env.scene['table_camera'].data.pos_w - env.scene.env_origins
            expected_camera = torch.tensor([d['camera']['position_m'] for d in domains], device=env.device)
            camera_error = torch.linalg.vector_norm(camera_positions-expected_camera, dim=-1)
            qvalid, rgb_std, depth_fraction = quality(snap)
            save(out / f'camera_wave{wave}.json', dict(
                ready=ready.tolist(), quality_valid=qvalid.tolist(), rgb_std=rgb_std,
                depth_valid_fraction=depth_fraction, frame_ids=env.scene['table_camera'].frame.cpu().tolist(),
                position_w=snap['pos_w'].cpu().tolist(), quat_w_ros=snap['quat_w_ros'].cpu().tolist(),
                camera_origin_error_m=camera_error.cpu().tolist(), warmup_steps=step,
                warmup_elapsed_s=time.perf_counter()-warmup_started))
            if not torch.all(camera_error < 1e-4):
                raise RuntimeError(f'Camera origin isolation check failed: {camera_error.cpu().tolist()}')
            from PIL import Image
            for i in range(n):
                Image.fromarray(snap['rgb'][i, ..., :3].cpu().numpy().astype(np.uint8)).save(out/f'wave{wave}_env{i}_initial.png')
            if probe_args.strict_benchmark and not ready.all():
                raise RuntimeError(f'Invalid camera initialization in wave {wave}: {ready.tolist()}')
            fsm = base.PickAndLiftStateMachine(env.step_dt, n, env.device)
            guards = [base.HybridActionGuard(maximum_position_step_m=.04, grasp_enable_distance_m=.075) for _ in range(n)]
            assert len(set(map(id, guards))) == n
            schedule = BatchChunkExecutor(n, args.execution_horizon, args.action_repeat)
            prediction_count = 0
            chunk_index = 0
            actions = torch.cat((ee, torch.ones((n, 1), device=env.device)), -1)
            done = ~ready.copy()
            success = np.zeros(n, dtype=bool)
            streak = np.zeros(n, dtype=int)
            end_steps = [0 if done[i] else None for i in range(n)]
            failure = ['invalid_visual_initialization' if done[i] else '' for i in range(n)]
            maximum_lift = np.zeros(n)
            invalid_camera = np.zeros(n, dtype=int)
            camera_checks = np.zeros(n, dtype=int)
            first_close = [None]*n
            early_close_requests = np.zeros(n, dtype=int)
            blocked = np.zeros(n, dtype=int)
            traces = [[] for _ in range(n)]
            criteria = base.benchmark_cfg.success_criteria
            control_started = time.perf_counter()
            for step in range(1, args.max_steps+1):
                _, _, terminated, truncated, _ = env.step(actions)
                if terminated.any() or truncated.any(): raise RuntimeError('Unexpected automatic reset')
                ee, obj = base._poses_in_robot_frame(env)
                robot = env.scene['robot']
                if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(ee).all():
                    raise RuntimeError('Nonfinite physical state')
                distance = torch.linalg.vector_norm(ee[:, :3]-obj[:, :3], dim=-1).cpu().numpy()
                z = env.scene['object'].data.root_pos_w[:, 2]
                lift = (z-initial_z).cpu().numpy()
                fingers = robot.data.joint_pos[:, -2:].mean(-1).cpu().numpy()
                hit = ((lift >= .10) & (z.cpu().numpy() >= base.TABLE_TOP_Z+float(criteria['minimum_height_above_table_m']))
                       & (distance < float(criteria['maximum_ee_object_distance_m']))
                       & (fingers < float(criteria['maximum_mean_finger_position'])))
                for i in range(n):
                    if done[i]: continue
                    maximum_lift[i] = max(maximum_lift[i], float(lift[i]))
                    streak[i] = streak[i]+1 if hit[i] else 0
                    if streak[i] >= int(criteria['sustained_steps']): success[i]=True; done[i]=True; end_steps[i]=step
                    elif float(z[i]) < base.TABLE_TOP_Z-.05: done[i]=True; failure[i]='object_drop'; end_steps[i]=step
                phase = fsm.state.clone().cpu().numpy()
                predicted = torch.zeros((n, 7), device=env.device)
                predicted[:, :3] = visual
                predicted[:, 3] = 1
                lift_pose = predicted.clone(); lift_pose[:, 2] += .18
                fallback = fsm.compute(ee, predicted, lift_pose)
                fresh = schedule.needs_prediction
                action_boundary = schedule.starts_action
                if fresh:
                    # Exactly one batched model forward, no per-environment inference loop.
                    base._refresh_camera(env)
                    snap = base._camera_snapshot(env)
                    qvalid, _, _ = quality(snap)
                    invalid_camera += ~qvalid
                    camera_checks += 1
                    if probe_args.strict_benchmark and not qvalid.all():
                        raise RuntimeError(f'Invalid RGB-D during wave {wave}, control step {step}')
                    state = base.pack_proprioceptive_state(robot.data.joint_pos.cpu().numpy(),
                        robot.data.joint_vel.cpu().numpy(), ee.cpu().numpy(), visual.cpu().numpy(),
                        ready.astype(np.float32), phase.astype(np.float32))
                    response = client.predict_batch(VLABatchRequest(snap['rgb'][..., :3].cpu().numpy(),
                        state, [args.task]*n, np.arange(n)))
                    latencies.append(response.inference_ms)
                    schedule.load(response.actions)
                    prediction_count += 1
                    peak_allocated=max(peak_allocated, torch.cuda.max_memory_allocated()/2**20)
                    peak_reserved=max(peak_reserved, torch.cuda.max_memory_reserved()/2**20)
                held = schedule.advance(ee.cpu().numpy())
                chunk_index = schedule.index-1
                for i in range(n):
                    if done[i]: continue
                    decision = guards[i].filter(held[i], fallback[i].cpu().numpy(),
                        current_eef_pose=ee[i].cpu().numpy(), visual_object_position=visual[i].cpu().numpy(),
                        perception_valid=bool(ready[i]))
                    if decision.source != 'vla':
                        done[i]=True; failure[i]=decision.reason; end_steps[i]=step
                        continue
                    actions[i] = torch.as_tensor(decision.action, device=env.device)
                    blocked[i] += decision.gripper_was_blocked
                    if action_boundary and schedule.raw[i, 7] < 0 and distance[i] > .025:
                        early_close_requests[i] += 1
                    if first_close[i] is None and decision.action[7] < 0:
                        first_close[i] = dict(step=step, true_distance_cm=float(distance[i]*100),
                            visual_distance_cm=float(torch.linalg.vector_norm(ee[i,:3]-visual[i]).item()*100))
                    if action_boundary:
                        traces[i].append(dict(step=step, phase=int(phase[i]), raw=schedule.raw[i].tolist(),
                            fresh_prediction=fresh, chunk_index=chunk_index,
                            filtered=decision.action.tolist(), eef=ee[i].cpu().tolist(), lift_cm=float(lift[i]*100)))
                if step % 100 == 0:
                    print('VLA_PARALLEL_PROGRESS='+json.dumps(dict(num_envs=n,wave=wave+1,step=step,
                        successes=int(success.sum()),camera_invalid=int(invalid_camera.sum()),
                        elapsed_s=time.perf_counter()-control_started)), flush=True)
                    save(out/'live_progress.json', dict(difficulty='medium', num_envs=n,
                        execution_horizon=args.execution_horizon, wave=wave+1, total_waves=probe_args.waves,
                        control_step=step, completed_episodes=len(all_rows),
                        completed_successes=sum(r['success'] for r in all_rows),
                        current_wave_successes=int(success.sum()), prediction_count=prediction_count))
            for i in range(n):
                row = dict(env_id=i,wave=wave,sample_index=domains[i]['sample_index'],
                    success=bool(success[i]),failure_reason='none' if success[i] else failure[i] or 'max_steps',
                    camera_ready=bool(ready[i]),ready_steps=ready_steps[i],visual_error_cm=100*initial_error[i],
                    camera_origin_error_m=float(camera_error[i]), maximum_lift_cm=100*maximum_lift[i],
                    first_close=first_close[i],gripper_blocked_steps=int(blocked[i]),
                    early_close_requests_over_2_5cm=int(early_close_requests[i]),
                    final_phase=int(fsm.state[i].item()), prediction_count=prediction_count,
                    invalid_camera_checks=int(invalid_camera[i]),camera_checks=int(camera_checks[i]),
                    end_step=args.max_steps if end_steps[i] is None else end_steps[i])
                all_rows.append(row)
                save(out/f'wave{wave}_env{i}_trace.json', dict(episode=row,trace=traces[i]))
            wave_reports.append(dict(wave=wave, control_wall_s=time.perf_counter()-control_started,
                total_wall_s=time.perf_counter()-t0,physical_control_steps=args.max_steps*n))
            save(out/'progress.json',dict(completed_waves=wave+1,episodes=all_rows))
            print('VLA_PARALLEL_WAVE_COMPLETE='+json.dumps(dict(wave=wave+1,
                completed_episodes=len(all_rows),successes=sum(r['success'] for r in all_rows))), flush=True)
        control_seconds=sum(w['control_wall_s'] for w in wave_reports)
        stable = all(r['camera_ready'] and r['invalid_camera_checks']==0 for r in all_rows)
        report=dict(status='PARALLEL_PROBE_COMPLETE',operationally_stable=stable,num_envs=n,
            waves=probe_args.waves,episodes=all_rows,successes=sum(r['success'] for r in all_rows),
            wave_reports=wave_reports,elapsed_s=time.perf_counter()-started,
            control_env_steps_per_s=args.max_steps*n*probe_args.waves/control_seconds,
            equivalent_600step_episodes_per_min=60*n*probe_args.waves/control_seconds,
            mean_batch_inference_ms=float(np.mean(latencies)),p95_batch_inference_ms=float(np.quantile(latencies,.95)),
            isaac_peak_allocated_mib=peak_allocated,isaac_peak_reserved_mib=peak_reserved,
            difficulty='medium',scene_seed=randomizer.seed,success_rate=sum(r['success'] for r in all_rows)/len(all_rows),
            camera=f'{probe_args.camera_mode} RGB-D 320x240',env_spacing_m=6.0,state_dim=30,
            execution_horizon=args.execution_horizon,action_repeat=5,
            scripted_fallback=False,truth_used_for_metrics_only=True,
            limitations=['Synchronous wave reset; finished slots hold target but remain in the inference batch.',
                'Capacity diagnostic, not success-rate equivalence with the legacy camera renderer.',
                'Only Medium tested; finite waves are not an indefinite stability guarantee.'])
        save(out/'parallel_report.json',report)
        print('VLA_PARALLEL_COMPLETE='+json.dumps({k:v for k,v in report.items() if k not in ('episodes','wave_reports')}),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        base.args_cli.output_dir.mkdir(parents=True, exist_ok=True)
        save(base.args_cli.output_dir/'probe_error.json',dict(error=f'{type(error).__name__}: {error}'))
        raise
    finally:
        base.simulation_app.close()
