# Phase 10：Camera RGB-D Data

## 结论

Phase 10 已完成。采集器现在可以在同一 episode 内同步保存机器人状态轨迹、RGB、米制 Depth、相机内参、world→camera 外参、相机/物体/末端世界位姿和物体在相机坐标系下的 6DoF Pose。Raw 数据可以继续导出为分片 HDF5，并由同一个 Dataset Reader 读取。

真实 RTX 相机采集、坐标变换校验、预览图、三维轨迹图和 HDF5 逐元素回读都已通过。

## 采集命令

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\collect_data.py `
  --headless `
  --enable_cameras `
  --num-envs 1 `
  --episodes 1 `
  --max_steps 300 `
  --camera_frame_interval 5 `
  --output_dir dataset\phase10_rgbd_validation `
  --randomization_config configs\randomization_medium.yaml
```

控制周期为 0.02 s，即低维轨迹为 50 Hz。`--camera_frame_interval 5` 每 5 个控制步保存一帧，对应 10 Hz。首帧和 episode 结束帧始终保存，因此 120 步成功轨迹得到 25 帧。

RGB-D 渲染开销显著高于纯物理仿真。当前验证使用 1 个环境保证显存和输出文件可控；后续批量采集可逐步测试 4、8、16 个相机环境，不应直接沿用 Phase 5 的 128–8192 纯状态并行规模。

## Raw episode 结构

```text
dataset/phase10_rgbd_validation/episode_000001/
├── trajectory.npz
├── camera.npz
├── metadata.json
├── domain_params.json
├── camera_rgb_preview.png
└── camera_depth_preview.png
```

`trajectory.npz` 保持原有 50 Hz schema。`camera.npz` 使用独立的帧维度 `F`：

| Tensor | Shape | Dtype | 定义 |
|---|---|---|---|
| `frame_step` | `[F]` | int64 | 对应 1-based 控制步 |
| `timestamp` | `[F]` | float64 | 仿真时间，秒 |
| `rgb` | `[F,H,W,3]` | uint8 | RGB 图像 |
| `depth` | `[F,H,W]` | float32 | image-plane depth，米 |
| `camera_intrinsics` | `[F,3,3]` | float32 | 针孔相机 K |
| `camera_pose_w` | `[F,7]` | float32 | ROS optical camera 在世界系中的 pose |
| `camera_extrinsics_w2c` | `[F,4,4]` | float32 | world→camera 齐次变换 |
| `object_pose_w` | `[F,7]` | float32 | 物体世界系 6DoF pose |
| `object_pose_camera` | `[F,7]` | float32 | 物体相机系 6DoF pose |
| `eef_pose_w` | `[F,7]` | float32 | 末端世界系 pose |

所有 pose 使用 `x,y,z,qw,qx,qy,qz`。`camera_pose_w` 使用 Isaac Lab 的 ROS optical camera 轴约定；Depth 的 `+inf` 表示射线没有命中，不能替换为 0。

## 坐标变换模块

`src/isaac_lab_data_engine/geometry/transforms.py` 统一实现：

- pose ↔ 4×4 transform；
- rigid transform inverse；
- world pose 到任意 reference frame 的 relative pose。

保存前根据 `camera_pose_w` 计算 `camera_extrinsics_w2c` 和 `object_pose_camera`。验证器会逐帧重新计算并比较矩阵，防止坐标系方向、四元数顺序或外参正反写错。

## 自动验证

`validate_episode()` 对相机数据额外检查：

- 必需 tensor、shape 和 dtype；
- `frame_step`、timestamp 严格递增且不超过轨迹长度；
- K、pose 和 extrinsics 全部有限；
- Depth 不含 NaN，有限值为正米制距离，且至少存在一个有效像素；
- world→camera 外参与 camera pose 一致；
- object-in-camera 与 camera/object 世界位姿一致。

## 实测结果

设备：NVIDIA GeForce RTX 4060 Laptop GPU；Medium domain randomization，320×240 RGB-D。

| 指标 | 结果 |
|---|---:|
| Episode | 1 |
| Success | 1 |
| State transitions | 120 |
| RGB-D frames | 25 |
| RGB shape | `[25,240,320,3]` |
| Depth shape | `[25,240,320]` |
| 有限 Depth 像素 | 49.71% |
| 有限 Depth 范围 | 1.923–4.977 m |
| RGB 范围 | 22–254 |
| Camera NPZ | 1,860,151 bytes |
| 纯 rollout 耗时 | 6.807 s |
| 实测 simulation FPS | 17.63 |
| 校验状态 | `EPISODE_VALIDATION_OK` |

首帧内参为：

```text
[[366.4996,   0.0000, 160.0000],
 [  0.0000, 366.4996, 120.0000],
 [  0.0000,   0.0000,   1.0000]]
```

## HDF5 导出

```powershell
python scripts\export_dataset.py `
  --input-dir dataset\phase10_rgbd_validation `
  --output-dir exports\pick_lift_rgbd_v1 `
  --episodes-per-shard 1
```

每个 HDF5 episode group 增加 `/camera/` 子组。Reader 示例：

```python
from isaac_lab_data_engine.data_collection.hdf5_dataset import HDF5EpisodeDataset

dataset = HDF5EpisodeDataset("exports/pick_lift_rgbd_v1")
sample = dataset[0]
rgb = sample["camera"]["rgb"]
depth = sample["camera"]["depth"]
object_pose_camera = sample["camera"]["object_pose_camera"]
```

真实导出包含 1 个 shard、120 条状态和 25 帧 RGB-D；源目录 1,921,182 bytes，HDF5 shard 1,681,934 bytes，逐 tensor 对比状态为 `DATASET_EXPORT_VALIDATION_OK`。

## 轨迹可视化

```powershell
python scripts\visualize_rgbd_episode.py
```

输出 `results/phase10/camera_object_eef_trajectory.png`，在世界坐标系中同时显示相机、物体和末端轨迹。RGB 与 Depth 预览位于对应 raw episode 目录。

## 自动测试

```powershell
python -m pytest tests\test_geometry.py tests\test_camera_episode.py `
  tests\test_rgbd_visualization.py tests\test_dataset_exporter.py -q
```

测试覆盖 SE(3) 组合/求逆、RGB-D raw round-trip、相机几何一致性、预览图、三维轨迹图和 camera-aware HDF5 round-trip。

## 当前边界

本阶段尚未保存 semantic/instance segmentation、法线或光流。当前数据已经包含 imitation learning、VLA 数据预处理和 6D pose perception 所需的核心 RGB + Depth + Object Pose + Robot State + Action，但不把 scripted controller 的 ground-truth state 输入误称为视觉策略。
