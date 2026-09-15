# Phase 4：Domain Randomization

## 结论

Phase 4 已完成。数据采集器现在可以在每个 episode 开始时，以可复现的随机数流改变物体、相机和光照参数；采样命令值与 PhysX/USD 实际回读值一同写入 episode，便于训练复现、数据审计和自动评测。

随机化是可选功能。不传 `--randomization_config` 时，`collect_data.py` 保持 Phase 3 行为。

## 运行命令

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\collect_data.py --headless --episodes 3 --max_steps 600 `
  --output_dir dataset\randomized --randomization_config configs\randomization.yaml
```

相机外参应用验证：

```powershell
python scripts\collect_data.py --headless --enable_cameras --episodes 1 --max_steps 600 `
  --output_dir dataset\randomized_camera --randomization_config configs\randomization.yaml
```

启用相机会初始化 RTX 渲染与 RGB-D 传感器，因此启动明显慢于纯物理采集。Phase 4 验证相机随机化，但不把图像写入 episode；图像和相机标定数据安排在 Phase 10。

## 随机化范围

范围定义在 `configs/randomization.yaml`，默认种子为 `20260916`。

| 类别 | 参数 | 默认范围 |
|---|---|---:|
| 物体 | x 位置 | 0.48–0.62 m |
| 物体 | y 位置 | -0.12–0.12 m |
| 物体 | yaw | -180–180° |
| 物体 | 质量 | 0.05–0.20 kg |
| 物体 | 静摩擦系数 | 0.45–0.95 |
| 物体 | 动摩擦系数 | 静摩擦的 0.70–0.95 倍 |
| 相机 | 位置扰动 | x/y ±0.08 m，z ±0.06 m |
| 相机 | 注视点扰动 | xyz 各 ±0.04 m |
| 光照 | DomeLight 强度 | 1800–3200 |
| 光照 | RGB 分量 | 各 0.75–1.00 |

物体 z 固定为 0.48 m，确保 reset 后落在桌面上。当前没有运行时随机缩放碰撞刚体：缩放会同时影响碰撞形状、惯量和抓取几何，适合在增加一致性重建与碰撞检查后单独实现。

## 应用流程

每个 episode 的顺序为：

1. 环境 reset，恢复默认机器人与场景状态；
2. `numpy.random.Generator` 按固定 seed 采样本 episode 参数；
3. 写入物体根位姿、速度、质量、按质量比例缩放的惯量和 PhysX 材质；
4. 若相机已启用，写入世界坐标系 eye/target 位姿；
5. 写入 USD DomeLight 强度和颜色；
6. `sim.forward()` 同步场景，再从 PhysX 与 USD 回读实际值；
7. 执行原有 Pick-and-Lift 控制器并采集轨迹。

同一配置和 seed 会产生相同的参数序列；连续 episode 使用不同的 `sample_index` 和样本。相机使用 USD local-to-world transform 作为权威回读，并以 `1e-5 m` 容差自动核对命令位置。

## Episode 数据

随机化 episode 比 Phase 3 多一个文件：

```text
episode_000001/
├── trajectory.npz
├── metadata.json
└── domain_params.json
```

`domain_params.json` 包含：

- `randomization_seed`、`sample_index` 和配置文件路径；
- `object / camera / lighting` 的采样命令值；
- `applied_values` 中的 PhysX/USD 回读值；
- 相机是否启用以及相机回读来源。

`metadata.json` 通过 `domain_randomization=true` 和 `domain_params_file` 关联 sidecar；数据集根目录的 manifest 同样建立索引。`validate_episode()` 会检查声明的 sidecar 存在、JSON 可解析且包含四个必需分区。

## 自动测试

```powershell
python -m pytest tests\test_episode_writer.py tests\test_domain_randomizer.py -q
```

测试覆盖随机序列复现、不同样本变化、数值范围、随机化 sidecar 回读与 manifest 索引。

## GPU 实测结果

测试日期：2026-09-16；设备：RTX 4060 Laptop GPU；Isaac Lab 2.3.2 / Isaac Sim 5.1.0。

连续三个随机 episode 均成功完成抓取：

| Episode | 步数 | 物体初始 x/y (m) | yaw | 质量 (kg) | 灯光强度 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| `000001` | 120 | 0.5526 / 0.0441 | -114.24° | 0.1385 | 3091.88 | 成功 |
| `000002` | 121 | 0.5819 / -0.0888 | 20.82° | 0.1968 | 2264.34 | 成功 |
| `000003` | 120 | 0.4917 / -0.1060 | -42.30° | 0.1297 | 2934.65 | 成功 |

独立相机测试中，采样位置 `[1.6004246, -2.0276048, 1.6102469] m`，USD 世界坐标回读为 `[1.60042465, -2.02760482, 1.61024690] m`，误差远小于自动校验阈值。
