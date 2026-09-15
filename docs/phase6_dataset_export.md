# Phase 6：Dataset Export

## 结论

Phase 6 已完成。系统保留 Phase 3–5 的 Raw episode 目录作为可审计源数据，并将其转换为带版本、索引、固定数据划分和完整性证明的 HDF5 shards。导出过程不依赖 Isaac Sim，可在普通 Python 进程中独立执行。

本阶段只处理低维机器人轨迹。RGB、Depth、相机内外参将在 Phase 10 扩展到同一导出协议。

## 导出命令

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
python scripts\export_dataset.py `
  --input-dir dataset\phase5_validation_128 `
  --output-dir exports\pick_lift_v1 `
  --format hdf5 `
  --episodes-per-shard 64 `
  --compression gzip `
  --compression_level 4 `
  --split_seed 42
```

导出默认执行源数据与 HDF5 的精确逐元素比较。只有明确传入 `--no_validate` 才会跳过，不建议在正式数据上使用。

## 输出结构

```text
exports/pick_lift_v1/
├── _SUCCESS
├── dataset_info.json
├── export_state.json
├── index.csv
├── splits.json
├── validation_report.json
└── shards/
    ├── shard_00000.h5
    └── shard_00001.h5
```

只有所有 shard、索引和校验都成功后才创建 `_SUCCESS`。Reader 会拒绝打开没有该标记的目录。

## HDF5 schema

每个 shard 的结构为：

```text
/episodes/episode_000001/
├── q                    float32 [T, 9]
├── dq                   float32 [T, 9]
├── eef_pose             float32 [T, 7]
├── object_pose          float32 [T, 7]
├── action               float32 [T, 8]
├── reward               float32 [T]
├── timestamp            float64 [T]
├── controller_state     int64   [T]
├── metadata_json
└── domain_params_json
```

数值 tensor 使用 chunked GZIP + shuffle；每个 episode 保留自己的长度和 JSON 元数据。shard 根属性记录格式版本、源数据目录、tensor schema、创建时间和 episode IDs。

## index.csv

每行对应一个 episode，包含：

- split、shard 路径和 HDF5 group；
- steps、success、failure reason、完成时间、累计奖励；
- parallel batch/env index；
- 物体位置、yaw、质量、静/动摩擦；
- 相机是否启用；
- Raw episode 相对路径与 trajectory SHA-256。

Phase 7 Benchmark 和 Phase 8 Failure Analysis 可以直接读取此索引，无需逐目录解析 JSON。

## 可复现数据划分

使用固定 seed 对 success/failure 分层后生成 Train/Validation/Test split，默认比例为 0.8/0.1/0.1。划分单位是完整 episode，同一 ID 不会跨 split，并自动检查无重叠和完整覆盖。

当前 128 条数据的结果：

| Split | Episodes |
|---|---:|
| Train | 102 |
| Validation | 13 |
| Test | 13 |

## 原子发布与断点恢复

导出首先写入输出目录旁的隐藏 staging 目录：

```text
.pick_lift_v1.staging/
```

每个 shard 先写 `.tmp`，关闭并 flush 后原子替换为正式文件；`export_state.json` 在每个 shard 完成后更新。全部验证成功后，整个 staging 目录才原子重命名为最终输出目录。

如果进程在中途退出：

```powershell
python scripts\export_dataset.py ... --resume
```

恢复前会核对源 manifest SHA-256 和所有导出参数，只跳过已经完成的 shard。如果最终目录已存在且包含 `_SUCCESS`，`--resume` 不重写数据，只重新执行完整性校验。默认拒绝覆盖已有输出。

## 完整性校验

`validation_report.json` 的检查包括：

- Raw episode 在导出前通过原始 schema 校验；
- index 中 episode ID 唯一；
- Train/Validation/Test 不重叠且覆盖完整；
- HDF5 tensor 的字段、长度、dtype、有限性和时间顺序正确；
- HDF5 tensor 与 Raw NPZ 逐元素完全相同；
- metadata/domain JSON 与源文件相同；
- 源 trajectory SHA-256 未发生变化；
- shard SHA-256 与 `dataset_info.json` 一致；
- shard 内不存在重复或未索引的 episode。

## Dataset Reader

```python
from isaac_lab_data_engine.data_collection.hdf5_dataset import HDF5EpisodeDataset

train = HDF5EpisodeDataset("exports/pick_lift_v1", split="train")
sample = train[0]

episode_id = sample["episode_id"]
q = sample["trajectory"]["q"]
action = sample["trajectory"]["action"]
domain_params = sample["domain_params"]
```

Reader 支持按索引或 episode ID 随机访问，并在每次访问时短暂打开对应 shard。这种实现可以安全复制到 PyTorch DataLoader worker。由于轨迹长度可变，组成 batch 时需要在训练代码中选择 padding、mask 或 sequence packing。

## 128-episode 实测结果

源数据：`dataset/phase5_validation_128`；输出：`exports/pick_lift_v1`。

| 指标 | 结果 |
|---|---:|
| Episodes | 128 |
| Shards | 2 × 64 episodes |
| Transitions | 15453 |
| 成功率 | 100% |
| Episode 长度 | 119–125，平均 120.73 |
| Raw episode 文件大小 | 2416423 bytes |
| HDF5 shard 大小 | 2526651 bytes |
| HDF5 / Raw 大小比 | 1.0456 |
| 完整导出与校验耗时 | 1.237 s |
| 校验状态 | `DATASET_EXPORT_VALIDATION_OK` |

HDF5 略大约 4.6%，因为源 NPZ 已经压缩，而 HDF5 还保存 group 结构和内嵌 JSON。这里采用 HDF5 的主要收益是减少零散文件、提供统一索引和训练读取接口，而不是进一步压缩容量。

记录的软件版本：Isaac Lab 2.3.2、Isaac Sim 5.1.0.0、Torch 2.7.0+cu128、NumPy 1.26.0、h5py 3.14.0。

## 自动测试

```powershell
python -m pytest tests\test_dataset_exporter.py -q
```

测试覆盖真实 HDF5 round-trip、分层 split、Reader、已有输出保护，以及在第二个 shard 人为中断后从 staging 继续完成导出。
