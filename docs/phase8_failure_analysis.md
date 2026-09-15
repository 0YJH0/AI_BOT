# Phase 8：Failure Analysis

## 结论

Phase 8 已完成。分析器读取 Phase 7 的逐 episode 报告，在不启动 Isaac Sim 的情况下自动验证数据、筛选失败样本、按 domain parameter 分箱、计算失败率与 Wilson 95% 区间，并输出 CSV、JSON 和 PNG。

正式分析默认只使用 Hard 的 256 条 episode。这样可以避免把 Easy / Medium / Hard 本身的难度差异误判为某一个参数的影响。Hard 中共有 16 次失败，整体失败率 6.25%。

## 运行命令

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
python scripts\analyze_failure.py
```

显式指定输入与输出：

```powershell
python scripts\analyze_failure.py `
  --input results\benchmark\benchmark_episodes.csv `
  --output-dir results\failure_analysis `
  --difficulty hard
```

传入 `--difficulty all` 可以分析所有难度，但跨难度结果存在明显混杂，只适合做全局数据质量检查，不应用于判断单个物理参数的影响。

## 数据校验

输入加载阶段会检查：

- 必需列是否存在；
- `(difficulty, episode_index)` 是否唯一；
- 布尔值和数值字段是否合法且有限；
- `success` 与 `failure_type` 是否一致；
- 筛选后的样本是否为空。

报告保存输入 `benchmark_episodes.csv` 的 SHA-256。本次输入哈希为：

```text
22599fc7705b222add958a2421c005fe5c95e472427a79eb10a96d88526aeffd7
```

## 统计方法

分析四个参数：

- `static_friction`；
- `object_mass_kg`；
- `object_yaw_deg`；
- `object_planar_distance_m = sqrt(object_x_m² + object_y_m²)`，坐标位于机器人基座 XY 平面。

摩擦、距离和质量各使用 5 个等宽分箱，朝向使用 6 个等宽分箱。边界由所选数据的实际最小值和最大值确定，最后一个分箱包含右端点。每个分箱输出样本数、成功数、失败数、失败率、Wilson 95% 区间、参数均值和各失败类型数量。

报告也给出参数与失败标志的 Pearson 相关系数。这个数只描述线性关联，无法识别 U 形关系，也不代表因果。

## 实测结果

### 失败类型

| 类型 | Episodes | 占全部 Hard episode | 占失败样本 |
|---|---:|---:|---:|
| Lift failure | 15 | 5.86% | 93.75% |
| Approach failure | 1 | 0.39% | 6.25% |

没有出现 grasp failure、object drop 或 timeout。当前 scripted controller 的主要瓶颈是已经形成近距离抓持后没有达到指定抬升高度。

### 高失败率区间

| 参数 | 最高风险区间 | 失败数 / 样本数 | 失败率 | Pearson r |
|---|---|---:|---:|---:|
| Object planar distance | `[0.7333, 0.8248]` m | 6 / 29 | 20.69% | +0.190 |
| Object mass | `[0.02136, 0.1067)` kg | 9 / 57 | 15.79% | -0.193 |
| Object yaw | `[117, 176]` deg | 7 / 47 | 14.89% | +0.087 |
| Static friction | `[0.08151, 0.2534)` | 5 / 44 | 11.36% | -0.087 |

最明显的描述性信号是远距离物体：最远分箱失败率 20.69%，而 `[0.4590, 0.5505)` m 分箱为 1.67%。低质量、正向大 yaw 和低摩擦区间也出现更高失败率。

这些区间存在参数共同变化。例如某条轨迹可以同时属于远距离、低质量和低摩擦分箱。Hard 只有 16 个失败，Wilson 区间仍较宽，因此不能从当前数据直接断言单一因果。Phase 9 会把这些结果作为增加困难样本覆盖率的采样信号，而不是当作控制策略性能已经改善的证据。

## 输出结构

```text
results/failure_analysis/
├── failure_analysis_report.json
├── failure_cases.csv
├── failure_distribution.csv
├── failure_distribution.png
├── failure_by_friction.csv
├── failure_by_friction.png
├── failure_by_distance.csv
├── failure_by_distance.png
├── failure_by_mass.csv
├── failure_by_mass.png
├── failure_by_orientation.csv
└── failure_by_orientation.png
```

所有 CSV 中的 rate 均保存为 0–1 的数值，图中转换为百分比。`failure_cases.csv` 保留 Phase 9 所需的失败类型、物体位置、平面距离、yaw、质量、摩擦和轨迹状态指标。

## 自动测试

```powershell
python -m pytest tests\test_failure_analysis.py -q
```

测试覆盖 Wilson 区间、分箱计数对账、输入一致性拒绝，以及从 CSV 到统计表、JSON 和 PNG 的端到端生成。
