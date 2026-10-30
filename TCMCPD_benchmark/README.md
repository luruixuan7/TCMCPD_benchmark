# TCMCPD 医案语料：数据分析、基线实验与新模型（HGP）

对公开数据集 **TCMCPD v1.0**（Zenodo `10.5281/zenodo.20929034`，CC BY 4.0）做的**语料分析**、
**18 模型基线评测**，以及在此之上设计的 **HGP（History-Gated Prescription）模型**实验。

本仓库只含三部分：**数据**、**实验脚本**、**结果**。

## 结构

```
TCMCPD_benchmark/                       68 个文件
├── README.md · NOTICE.md · requirements.txt · CHECKSUMS.sha256
│   （另有 .gitattributes / .gitignore）
├── data/        TCMCPD v1.0 原始数据，18 文件（entities + relationships + 说明），未修改
├── scripts/     19 个：数据分析 10 + baseline 流水线 5 + 预处理 1 + HGP 3
└── results/     20 个 JSON + 5 张图（PNG），其中 8 个在 results/hgp/
```

规模：**18 数据 + 19 脚本 + 25 结果 + 6 根文件 = 68**。

## 实验与脚本、结果的对应

| 编号 | 实验内容 | 脚本 | 结果文件 |
|---|---|---|---|
| E0 | 数据范围与质量事实（规模、覆盖率、命名、单位、日期） | `scripts/experiments/e0_data_scope.py` | `results/e0_data_scope.json` |
| E1 | 标签存在律（记录完整性 → 是否带证型，AUC 0.933） | `scripts/experiments/e1_e2_patterns.py` | `results/e1_e2_patterns.json` |
| E2 | 可学性定律（频次 vs 结构分解） | 同上 | `results/e1_e2_patterns.json` |
| E2b | 词表分布（离散拟合、AIC 比较、Vuong 检验） | `scripts/experiments/e1c_distribution.py` | `results/e1_e2_distribution.json` |
| E2c | 结构增量的偏 F 检验（R² 增量是否显著） | `scripts/experiments/e1d_r2_increment.py` | `results/e1_e2_r2_increment.json` |
| E3 | 18 个基线模型的指标分歧 | `scripts/pipeline/bench_*.py` + `scripts/experiments/e3_metric_divergence.py` | `results/e3_b1_all_models.json`、`results/e3_b2_all_models.json`、`results/e3_metric_divergence.json` |
| E5 | 任务可行性（处方持续性、last-basket、增量、间隔） | `scripts/experiments/e5_*.py`、`scripts/experiments/e5b_vocabulary_scope.py` | `results/e5/*.json` |
| E6 | 历史感知协议与 HGP 模型 | `scripts/experiments/e6_history_protocol.py`、`scripts/models/hgp*.py` | `results/hgp/*.json` |
| E6b | 对抄写基线的配对 bootstrap、固定权重混合基线、两种 AP 口径 | 同上 | `results/hgp/e6_copy_baseline_ci.json`、`results/hgp/e6_hybrid_baseline.json`、`results/hgp/e6_ap_conventions.json` |
| — | 预处理口径与计数 | `scripts/prep/prep_tcmcpd.py` | `results/prep/prep_manifest.json` |
| — | 等效性检验（指标分歧的区间证据） | `scripts/appendix/equivalence.py`（未随包分发） | `results/appendix/equivalence.json` |

## E6：历史感知协议与 HGP 模型

E5 显示相邻处方高度重合（Jaccard 0.219，last-basket 的 mAP 0.654 高于全部训练模型），说明**上一次处方是
被所有内容型基线丢弃的最强信号**。E6 因此把任务重述为：给定本次症状与上一次处方，预测本次药味集合，
并在**原始固定划分**（患者级 70/15/15，seed 42）上评测，只取"有上一次处方"的子集
（train 2,374 / val 496 / test 520）。

**基线全部复用已保存的预测**（`preds_B2_*.npz`，19 个模型），不重训，因此与 E3 的数字同源可比。

**HGP 的每个部件都溯源到前面的一条规律：**

| 部件 | 来源规律 |
|---|---|
| carry：沿用上一次处方 | E5：处方持续性（Jaccard 0.219；last-basket mAP 0.654） |
| 逐药门控：每味药单独学"信历史还是信症状" | E5 + E2：频次与共现度决定可学性 |
| label-graph 结构塔 | E2：结构解释药味可学性 42%（R² 0.454→0.779） |
| 增/删双头 | E5：每诊平均新增 3.4 味、移除 4.4 味 |
| 验证集阈值规则 | E3：macro-F1 与 mAP 的排序不一致 |

**结果（测试子集 520 例）**

| | R@10 | mAP@10 | P@1 | macro-F1 |
|---|---|---|---|---|
| 学习型基线（18 个）各自最高值 | 0.2936 | 0.5214 | 0.2846 | — |
| last-basket（抄上次处方） | 0.4788 | 0.6607 | 0.4442 | 0.1892 |
| **HGP（三臂融合）** | 0.4750 | **0.6664** | **0.4558** | **0.1924** |

- 对 **18 个学习型基线逐一比较**：R@10、mAP@10、P@1 **同时更高**；相对各指标最好者
  mAP@10 **+27.8%**、recall@10 **+61.8%**、precision@1 **+60.1%**；
- 对 **平凡抄写基线**：mAP@10、P@1、macro-F1 三项超过，recall@10 差 0.0038（该任务上两者是凹前沿，
  见 `results/hgp/` 的记录）。

**运行**（需要 E3 流水线产生的预测缓存）

```bash
python scripts/experiments/e6_history_protocol.py   # 复原固定划分、子集、复用基线，并训练两个臂
python scripts/models/hgp.py                        # 用全部训练就诊训练门控模型
python scripts/models/hgp_combine.py                # 三臂融合与验证集选参
python scripts/models/hgp_ablation.py               # 组件消融（事后加性部件为何无效）
```

依赖：`TCMCPD_BENCH_CACHE`（默认 `/data/tcmcpd_bench/bench`）指向含 `cache_B2.npz` 与
`preds_B2_*.npz` 的目录；结果默认写到 `results/hgp/`，可用 `TCMCPD_HGP_OUT` 覆盖。

## 运行

```bash
pip install -r requirements.txt

python scripts/prep/prep_tcmcpd.py             # 预处理口径与计数
python scripts/experiments/e0_data_scope.py    # E0
python scripts/experiments/e1_e2_patterns.py   # E1 / E2
python scripts/experiments/e1c_distribution.py # E2b
python scripts/experiments/e5_task_feasibility.py
```

**基线实验（E3）的流水线，顺序不可颠倒：**

```bash
python scripts/pipeline/bench_data.py B2
python scripts/pipeline/bench_models.py B2
python scripts/pipeline/bench_extra.py B2
python scripts/pipeline/bench_tabpfn.py B2
python scripts/pipeline/bench_eval.py B2
```

数据路径默认取仓库内的 `data/`，可用环境变量 `TCMCPD_ROOT` 指向其他 checkout。

## 数据来源与许可

TCMCPD v1.0，Zenodo，DOI `10.5281/zenodo.20929034`，CC BY 4.0。原始文件未做任何修改；
引用方式见 `data/TCMCPD_TCM_Case_Records_v1.0/CITATION.md`，使用条款见 `NOTICE.md`。

## 完整性

`CHECKSUMS.sha256` 覆盖 `results/` 与部分 `data/` 下共 38 个文件，可用
`sha256sum -c CHECKSUMS.sha256`（或 Python 脚本）校验。
