# Phase 2：Pick-and-Lift Task

## 目标与结论

Phase 2 已完成并通过单环境 GPU 物理验收。Franka Panda 使用绝对位姿 Differential IK 与确定性的向量化状态机抓取桌面上的立方体，并将其提升到指定高度。当前阶段没有训练 RL，也没有启用 Domain Randomization。

## 实现结构

- `src/isaac_lab_data_engine/envs/pick_lift_env_cfg.py`：Manager-Based RL 环境配置、动作、奖励、终止条件、相机和固定目标。
- `src/isaac_lab_data_engine/envs/observations.py`：末端、物体及相对位姿观测。
- `src/isaac_lab_data_engine/controllers/scripted_pick_lift.py`：GPU Tensor 上运行的向量化状态机。
- `scripts/run_pick_lift.py`：运行、成功判定、结果汇总和 GUI 调试入口。

状态机顺序为：

```text
REST -> APPROACH_ABOVE_OBJECT -> APPROACH_OBJECT -> GRASP_OBJECT -> LIFT_OBJECT
```

动作张量 shape 为 `(num_envs, 8)`：

```text
[eef_x, eef_y, eef_z, quat_w, quat_x, quat_y, quat_z, gripper]
```

其中前 7 维是机器人根坐标系中的绝对末端目标位姿，最后 1 维是二值夹爪命令。

策略观测包括：

- 关节位置、关节速度；
- 物体位置与目标位置；
- 末端位姿 `(position + quaternion_wxyz)`；
- 物体位姿 `(position + quaternion_wxyz)`；
- 在末端坐标系表达的物体相对位置；
- 上一步动作。

## 成功判定

必须连续 10 个环境步同时满足：

- 物体高度不低于桌面上方 `0.10 m`；
- 两个手指关节的平均位置小于 `0.030 rad`；
- 末端与物体中心距离小于 `0.08 m`。

固定随机种子为 `42`。Phase 4 才会引入随机化。

## 实测结果

验收命令：

```powershell
python scripts\run_pick_lift.py --headless --num_envs 1 --max_steps 600 --save_camera
```

2026-09-15 在 RTX 4060 Laptop GPU 上的结果：

| 指标 | 实测值 |
|---|---:|
| 状态 | `PHASE2_SUCCESS` |
| 成功步 | 121 |
| 仿真时间 | 2.42 s |
| 初始物体高度 | 0.4800 m |
| 最大物体高度 | 0.6846 m |
| 实际提升量 | 0.2046 m |
| 最终末端—物体距离 | 0.0091 m |
| 动作 shape | `(1, 8)` |
| RGB 数值范围 | `[25, 255]` |
| 有效深度像素比例 | 0.5009 |

最终 RGB 验收图保存在 `results/phase2/pick_lift_final_rgb.png`。

## 可见窗口调试

实时观看动作，并在成功后保持最终抓取姿态：

```powershell
python scripts\run_pick_lift.py --num_envs 1 --max_steps 600 --realtime --keep_open
```

关闭 Isaac Sim 窗口即可退出。去掉 `--keep_open` 时，脚本在达到成功条件后自动关闭；去掉 `--realtime` 时，仿真会尽可能快地运行。
使用 `--realtime_speed 0.5` 可以按半速观察动作。

为了缩短日常调试启动时间，GUI 默认只启用 viewport，不创建 RGB-D/Replicator 渲染产品。需要在 GUI 中同时读取传感器时显式加入 `--enable_cameras`。轻量演示建议使用：

```powershell
python scripts\run_pick_lift.py --num_envs 1 --max_steps 600 --realtime --realtime_speed 0.4 --rendering_mode performance
```

## 已知的非阻塞警告

- 当前 NVIDIA 驱动低于 Isaac Sim 5.1 官方测试版本；现有场景可以运行，但复杂渲染或更大规模并行任务可能暴露兼容性问题。
- 320×240 低于 DLSS 推荐输入尺寸，因此运行时会看到分辨率提示；当前使用 DLAA，采集结果有效。
- RTX 4060 Laptop 的 8 GB 显存会限制后续带相机的并行环境数量，Phase 5 将实测可用规模。
