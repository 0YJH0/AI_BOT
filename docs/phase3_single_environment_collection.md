# Phase 3：单环境轨迹采集

## 结论

Phase 3 已完成。`collect_data.py` 可以连续运行单环境 Pick-and-Lift episode，将成功或失败轨迹写成版本化的 NPZ/JSON 数据集，并在每次写入后立即回读校验。

本阶段只采集低维状态，不保存图像。RGB、Depth、相机内外参将在 Phase 10 加入，避免将基础轨迹采集与 RTX 渲染耦合。

## 运行命令

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\collect_data.py --headless --episodes 1 --max_steps 600
```

指定输出目录：

```powershell
python scripts\collect_data.py --headless --episodes 10 --max_steps 600 --output_dir dataset\phase3_run
```

脚本固定使用一个环境；Phase 5 才扩展为 vectorized parallel rollout。

## 数据目录

```text
dataset/
├── dataset_metadata.json
├── episode_000001/
│   ├── trajectory.npz
│   └── metadata.json
└── episode_000002/
    ├── trajectory.npz
    └── metadata.json
```

`dataset_metadata.json` 保存格式版本、统一 tensor schema、坐标约定、episode 索引以及成功/失败计数。episode 编号自动递增，已有目录不会被覆盖。

## trajectory.npz schema

设 episode 长度为 `T`：

| 字段 | Shape | Dtype | 含义 |
|---|---:|---|---|
| `q` | `(T, 9)` | `float32` | 7 个手臂关节和 2 个手指关节位置 |
| `dq` | `(T, 9)` | `float32` | 关节速度 |
| `eef_pose` | `(T, 7)` | `float32` | 机器人根坐标系中的末端位姿 |
| `object_pose` | `(T, 7)` | `float32` | 机器人根坐标系中的物体位姿 |
| `action` | `(T, 8)` | `float32` | 绝对 IK 目标位姿和夹爪命令 |
| `reward` | `(T,)` | `float32` | Isaac Lab manager reward |
| `timestamp` | `(T,)` | `float64` | 仿真时间，单位秒 |
| `controller_state` | `(T,)` | `int64` | scripted controller 状态编号 |

位姿顺序统一为：

```text
x, y, z, qw, qx, qy, qz
```

每一行保存 `action[t]` 执行后的状态和奖励。时间戳严格递增，当前控制周期为 0.02 秒。

## metadata.json

每个 episode 单独记录：

- `episode_id`、格式版本和 UTC 创建时间；
- `success / failure_reason`；
- `steps / completion_time_s / total_reward`；
- 随机种子、任务名和控制器名；
- 初始、最大和最终物体高度；
- 成功判定阈值；
- 坐标系和四元数约定；
- 每个 tensor 的实际 shape 与 dtype；
- `camera_data=false`，明确说明当前数据不含图像。

## 自动校验

`validate_episode()` 会检查：

- NPZ 是否包含所有必需字段；
- 第一维是否与 metadata 中的步数一致；
- 固定尾部 shape 和 dtype；
- 所有数据是否为有限数；
- 时间戳是否严格递增。

无仿真 round-trip 单元测试：

```powershell
python -m pytest tests\test_episode_writer.py -q
```

实测结果为 `1 passed`。

## GPU 实测结果

测试日期：2026-09-15；设备：RTX 4060 Laptop GPU。

| Episode | 结果 | 步数 | 仿真时长 | 累计奖励 | 最大物体高度 | 校验 |
|---|---|---:|---:|---:|---:|---|
| `000001` | 成功 | 120 | 2.40 s | 6.1726 | 0.6845 m | `EPISODE_VALIDATION_OK` |
| `000002` | 失败（人为限制为 30 步） | 30 | 0.60 s | 0.0083 | 0.4800 m | `EPISODE_VALIDATION_OK` |

成功 episode 覆盖了状态机的全部五个状态；失败 episode 证明未完成轨迹仍能正常落盘、标注和进入 dataset manifest。
