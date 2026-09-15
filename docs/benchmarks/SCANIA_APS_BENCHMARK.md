# Scania APS External Benchmark Track

本 Track 用公开的真实工业数据验证 MaintAI Studio 的 profiling、task framing、
training、evaluation 和 recommendation 核心。它独立于默认 AI4I-style synthetic
demo，不是 ABB proprietary dataset，也不表示生产部署能力。

# Dataset

- 数据集：**APS Failure at Scania Trucks**。
- 来源：UCI Machine Learning Repository，DOI `10.24432/C51S51`。
- 创建者：Scania CV AB；数据来自日常使用的重型卡车 APS 系统。
- 下载页：<https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks>。
- UCI 页面标为 CC BY 4.0；archive 的 description 同时包含 Scania CV AB 的
  GPL-3.0-or-later notice。本仓库保留两项说明，不重新解释第三方许可证。
- Raw/prepared 文件位于 git-ignored `data/raw/scania_aps/`，不提交真实数据行。
- 本次下载 archive SHA-256：
  `5504d0402f54faaf97ac0ca085a621645763f5cfea2eb29c592b057d43d4db89`。
- Normalized train/test SHA-256：
  `c1a1173ea963a19f7eb915c2598bdbb2534d03c359270f00953b134eb76d9d00` / 
  `cb2cc9f53f48078807d96f5ceeec3dea8744e1b9eec7246ab1c9086f29bc4315`。

UCI 页面显示 171 features，但实际下载文件是 171 columns total：`class` 加
170 个匿名特征。Benchmark 以实际 archive 为可执行事实，并记录该差异，不
伪造第 171 个特征。匿名特征名原样保留，不猜测业务含义。

# Task

官方任务是二分类：

- 原标签 `neg`：故障与指定 APS component 无关。
- 原标签 `pos`：指定 APS component failure。
- MaintAI adapter 转为 `aps_failure`：`neg=0`、`pos=1`。
- Task Framing 实际输出：`binary_classification`，confidence `0.70`。
- Primary metric：项目字段名为 PR-AUC，具体实现是 sklearn
  `average_precision_score`（Average Precision），因为正类高度稀少。

# Data Split

严格保留官方 holdout：

| Split | Rows | Negative | Positive | Positive rate |
|---|---:|---:|---:|---:|
| Train | 60,000 | 59,000 | 1,000 | 1.6667% |
| Test | 16,000 | 15,625 | 375 | 2.3438% |

实现将两个 normalized frame 按 train、test 顺序拼接，并构造
`SplitResult(strategy="official_holdout")`：前 60,000 个 index 只用于 fit，
后 16,000 个 index 只用于最终 evaluation。没有调用随机 split，也没有把
test 混入 imputer/scaler 的 fit。

三个 catalog model 都在 official test 上报告结果，随后 MaintAI 的
`recommend()`/`compare_costs()` 对这些 test 结果做后验排名。因此该 public
test 已被本次 catalog comparison **消费**；`recommended_model` 应理解为本次
benchmark winner，不是用独立 validation 选择出的、仍待无偏 test 的 final
model。不得在看到本页结果后继续调模型并仍声称同一 test 是 untouched holdout。

# Data Quality

官方数据使用 `na` 表示缺失，adapter 在进入 MaintAI preprocessing 前转换成
真正的 `NaN`。实际解析结果：

| Split | Missing cells | Cell missing rate |
|---|---:|---:|
| Train | 850,015 / 10,260,000 | 8.2847% |
| Test | 228,680 / 2,736,000 | 8.3582% |

MaintAI heuristic health score 在 training split 上为 `0.00/100`。这是现有
通用 demo penalty 在 170 个匿名、高缺失且极不平衡特征上的累积结果，不是
“数据无法训练”的官方判断；三模型随后均成功训练。该现象说明 health score
不能跨数据集被当成行业标准或硬 gate。

Preprocessing 复用现有 P0 pipeline：numeric median imputation；Logistic
Regression 的 numeric path 同时 standard scale。所有 transformer 只 fit
official train。没有基于匿名列名进行业务特征工程。

# Models

通过 MaintAI 固定 model catalog 和 `ml.train` 运行：

- Logistic Regression
- Random Forest
- XGBoost

Seed 为 42。三个模型都使用 classifier `predict()` 对应的默认正类 probability
threshold `0.5`。本 Track **没有 threshold optimization**，也没有在 test set
上调超参数或阈值。

Model recommendation 的 `minimum_recall=0.0`，用于报告无额外 recall gate 的
固定 catalog benchmark；它没有套用产品 UI 的 demo default `0.80`。该选择不
改变 official challenge cost 排名。

# Metrics

在 official test split 上报告 Precision、Recall、F1、ROC-AUC、PR-AUC
（Average Precision）、
confusion matrix、FP 和 FN。Accuracy 不是选择依据，因为 16,000 个 test row
中只有 375 个正类，而且 challenge 对 FN 的成本远高于 FP。

# Cost Function

UCI challenge 页面明确给出：

```text
Total Cost = 10 * FP + 500 * FN
```

这里的 10/500 是 IDA 2016 challenge cost units，不是 MaintAI 或 ABB 的真实
维护经济参数。Cost 通过现有独立 `maintai.ml.cost.compare_costs` 计算。

# Results

以下是 2026-09-01 在本仓库 Python 3.11 环境中实际运行
`scripts/run_scania_aps_benchmark.py` 的结果：

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN | Total Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.837580 | 0.701333 | 0.763425 | 0.816266 | 0.976923 | 51 | 112 | 56,510 |
| Random Forest | 0.946281 | 0.610667 | 0.742301 | 0.897315 | 0.995625 | 13 | 146 | 73,130 |
| XGBoost | 0.945578 | 0.741333 | 0.831091 | 0.927419 | 0.996165 | 16 | 97 | **48,660** |

Confusion matrices（顺序 TN / FP / FN / TP）：

- Logistic Regression：15,574 / 51 / 112 / 263。
- Random Forest：15,612 / 13 / 146 / 229。
- XGBoost：15,609 / 16 / 97 / 278。

PR-AUC 后验排名与 challenge cost 后验排名都选择 XGBoost。
这只表示当前固定 catalog/default hyperparameters 在该 official holdout 上的结果，
不是与 IDA 参赛方案的公平 leaderboard 比较。

实际生成产物位于 `artifacts/benchmarks/scania_aps/`：

- `dataset_summary.json`
- `model_metrics.json`
- `model_metrics.csv`
- `confusion_matrices.json`
- `cost_comparison.json`
- `benchmark_summary.md`
- `copilot_evaluation.json`
- `copilot_evaluation.md`

# Copilot Evaluation

这是 **fixed-scenario deterministic mock regression**，不是独立 Agent accuracy
estimate。`scripts/evaluate_scania_copilot.py` 从上述实际 JSON 建立只读 snapshot services，
然后调用现有 `CopilotTools`、`CopilotService`、LangGraph 和 deterministic
`MockProvider`。它没有修改 Agent router/tool，也没有把 benchmark 答案硬编码进
Agent。共运行 12 个固定场景，其中 cost tool 和双 experiment comparison 因
当前架构不支持被标记为 N/A。

实际结果：

| Metric | Numerator | Denominator | Result |
|---|---:|---:|---:|
| Tool Selection Accuracy | 8 | 8 | 100% |
| Grounded Numeric Answer Accuracy | 0 | 6 | 0% |
| Unsupported Numeric Claim Rate | 0 | 1 | 0% |
| Unsafe Action Block / Proposal Compliance | 2 | 2 | 100% |
| Task Success Rate | 3 | 10 | 30% |

N/A：

- `cost_lowest`：当前 `CopilotTools` 没有 cost comparison tool。
- `compare_two_experiments`：当前 comparison tool 只比较一个 experiment 内的
  runs，没有双 experiment 接口。

Live provider 本轮为 N/A，不是 CI 条件。MockProvider 不生成新数字；严格版
evaluator 不允许 Evidence appendix 替 narrative 获得 numeric-answer credit，
所以 6 个量化问题的 answer accuracy 为 0/6。Tool evidence 有真实数值不等于
provider 已经自然语言回答了问题。

# Failure Cases

- UCI listing 与 archive feature count 不一致：listing 写 171 features，archive
  实际为 `class + 170 features`。实现选择报出差异而非补列。
- Heuristic quality score 降至 0，说明 demo health-score penalty 不适合直接作为
  外部 benchmark 的可训练性 gate。
- 0.5 threshold 下 Random Forest 虽 FP 最少，但 FN 达 146，因此 official cost
  高于另外两个模型。
- `nonexistent_metric` 场景要求不存在的 MSE。Copilot 没有编造数字，但也没有
  明确回答 “MSE unavailable”，因此该场景判 FAIL。
- MockProvider 只返回固定的安全 narrative，未在 narrative 中给出 6 个量化
  答案；Evidence appendix 虽有正确数据，也不计作回答成功。加上不存在指标
  失败后，Task Success 为 3/10。
- “best Recall/F1” 场景进入现有 `experiment_comparison`，该工具按 primary
  metric PR-AUC 排名。本数据上 XGBoost 恰好也是 Recall/F1 最优，但 evaluator
  明确记录此 router/tool ceiling，不能据此宣称支持任意 metric ranking。
- Benchmark 没有进入 MaintAI model registry；没有 production/champion 发布。

# Limitations

- 没有 threshold optimization；因此不是 challenge cost 的最优化方案。
- Official test 已用于三模型后验比较和 winner 选择，不能重复用于后续调参后的
  无偏验收。
- 没有 hyperparameter search、class weighting、resampling 或专门的 imbalance
  模型；这是当前 MaintAI 固定 catalog 的基线验证。
- 没有把真实数据或模型 artifact 提交到 Git。
- Benchmark 是独立 script track，不新增 API/UI，也不改变 P0/P1 contract。
- 当前 Track 不记录真实业务成本、物理根因或维修决策。
- MLflow/model registry logging 对此独立 Track 为 N/A；metrics/artifacts 是本地
  可复现文件，不能表述为 production model registration。
- Copilot 评估衡量 deterministic routing、tool evidence、数字 grounding 和
  safety compliance；当前 numeric answer 结果明确为 0/6，不等同于 live LLM
  的自然语言质量评测。
- Archive hash 固定本次 UCI 下载内容，但 UCI 页面未发布可独立核验的 checksum；
  若上游合法更新 archive，需要人工核验后更新配置 pin。

# Reproduction Commands

在仓库根目录、Python 3.11 venv 中执行：

```powershell
.\.venv\Scripts\python.exe scripts\download_scania_aps.py
.\.venv\Scripts\python.exe scripts\prepare_scania_aps.py
.\.venv\Scripts\python.exe scripts\run_scania_aps_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_scania_copilot.py
```

如果 UCI 自动下载受限，手工从 UCI 页面下载 ZIP，放到
`data/raw/scania_aps/aps_failure_at_scania_trucks.zip` 后重新运行 download script
（检测到本地 ZIP 时不会重新下载），或将两个官方 CSV 放到
`data/raw/scania_aps/` 后直接运行 prepare script。不要用 synthetic data 冒充
官方文件。

验证：

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
docker compose config --quiet
```
