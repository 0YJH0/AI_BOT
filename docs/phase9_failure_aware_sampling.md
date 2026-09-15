# Phase 9：Failure-aware Sampling

## 结论

Phase 9 已完成。项目已经形成可运行的数据闭环：

```text
Uniform Hard Benchmark
→ Failure Analysis
→ Failure-aware Distribution
→ GPU Round 2
→ Coverage Comparison
```

在相同的 256 episode 预算下，Uniform Hard 生成 16 条失败样本，Failure-aware Round 2 生成 46 条，困难样本产出提高到 2.875 倍。四个 Phase 8 高风险区间的覆盖率都得到提升。

这里的目标是增加长尾困难数据，而不是证明控制器性能提高。Round 2 改变了测试分布，因此它的成功率不能与 Uniform Hard 成功率直接解释为策略退化。

## 一键运行

```powershell
conda activate isaaclab23
cd C:\Users\yjh\Desktop\work_project\ai_BOT
$env:OMNI_KIT_ACCEPT_EULA = "YES"
python scripts\run_data_loop.py
```

该命令顺序执行：

1. 对 `results/benchmark/benchmark_episodes.csv` 重新运行 Hard failure analysis；
2. 从失败明细生成 `configs/randomization_failure_aware.yaml`；
3. 启动 Isaac Lab，运行 256 个 Round 2 episode；
4. 比较 Uniform 与 Failure-aware 的失败样本产出和高风险区间覆盖。

已有完整输出时，可只重建对比报告：

```powershell
python scripts\run_data_loop.py --reuse-existing
```

## 采样策略

Failure-aware 分布由两部分组成：

- 70%：从 Phase 8 的 16 个真实失败 episode 中等概率选择一个联合参数原型，在物体 X/Y、yaw、质量、静摩擦和动摩擦比附近加入截断高斯抖动；
- 30%：继续使用原始 Uniform Hard 分布，保留探索能力并降低对少量失败样本的过拟合。

抖动标准差由原始 Hard 参数范围的固定比例定义：位置和质量/静摩擦为 6%，yaw 为 5%，动摩擦比为 4%。所有结果被裁剪到 Hard 配置的合法边界。

使用失败样本的联合参数组合比独立提高多个高风险分箱的概率更稳妥。它能够保留“远距离 + 低质量 + 特定摩擦/yaw”在真实失败轨迹中的共现关系。

生成的 YAML 保存以下审计信息：

- base Hard config；
- failure cases CSV 与 SHA-256；
- Phase 8 report 与 SHA-256；
- sampling seed 9101；
- hard-case 概率和所有 jitter 比例。

加载时会重新计算两个来源文件的 SHA-256，不匹配则拒绝运行。每个 episode 另外记录 `hard_case_selected`、来源失败 episode 和来源失败类型。

## 离线分布检查

正式 GPU rollout 前使用固定 seed 各抽样 10,000 次：

| 区域 | Uniform Hard | Failure-aware |
|---|---:|---:|
| Hard-case 分支 | 0.00% | 70.28% |
| Far distance | 15.81% | 32.67% |
| Low mass | 20.09% | 45.56% |
| Low friction | 19.24% | 27.31% |
| High yaw | 17.05% | 27.69% |

这一步只验证采样器分布，不启动 Isaac Sim。

## GPU Round 2 实测

设备与 Phase 7 相同：NVIDIA GeForce RTX 4060 Laptop GPU，128 个并行环境，共 256 个 episode。

| 指标 | Round 1 Uniform Hard | Round 2 Failure-aware |
|---|---:|---:|
| Episodes | 256 | 256 |
| Success | 240 | 210 |
| Failure | 16 | 46 |
| Failure rate | 6.25% | 17.97% |
| Failed episodes / 100 | 6.25 | 17.97 |
| Average episode reward | 5.4654 | 4.7673 |
| Lift failure | 15 | 39 |
| Approach failure | 1 | 4 |
| Object drop | 0 | 3 |

Round 2 实际选择 hard-case 分支 176 次，占 68.75%；其中 43 次失败，条件失败率 24.43%。保留的 80 次 uniform exploration 中有 3 次失败，条件失败率 3.75%。

新增 object drop 表明重采样覆盖了原均匀 256 条样本中没有出现的边缘接触状态。

## 高风险区间覆盖

| Phase 8 高风险区间 | Uniform | Failure-aware | 覆盖倍数 |
|---|---:|---:|---:|
| Low friction | 17.19% | 26.95% | 1.57× |
| Far distance | 11.33% | 32.81% | 2.90× |
| Low mass | 22.27% | 37.11% | 1.67× |
| High yaw | 18.36% | 23.83% | 1.30× |

提升最大的是远距离区间：29 条增加到 84 条，覆盖率提高 21.48 个百分点。

## 输出结构

```text
configs/
├── benchmark_failure_aware.yaml
└── randomization_failure_aware.yaml

results/benchmark_failure_aware/
├── benchmark_results.csv
├── benchmark_episodes.csv
└── benchmark_report.json

results/data_loop/
├── round_comparison.csv
├── high_risk_coverage.csv
├── high_risk_coverage.png
├── difficult_episode_yield.png
├── data_loop_report.json
└── full_loop_run.log
```

## 可复现性

完整一键闭环再次运行后，Round 2 episode CSV 的哈希保持一致：

```text
first  SHA-256: CF746910543DB8FCC4971A513B4041E9E16F68EF4F8CBD4B1234D9995E0B016E
second SHA-256: CF746910543DB8FCC4971A513B4041E9E16F68EF4F8CBD4B1234D9995E0B016E
```

采样分支、来源失败 episode、domain parameters 和物理结果均逐字节一致。wall time、吞吐量、日志和报告时间不属于复现判据。

## 自动测试

```powershell
python -m pytest tests\test_failure_aware_sampling.py tests\test_round_comparison.py -q
```

测试覆盖固定种子复现、范围裁剪、uniform exploration、来源哈希拒绝、Round 1/2 对账、覆盖率提升和图表生成。
