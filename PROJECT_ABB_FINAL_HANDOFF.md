# PROJECT_ABB_FINAL_HANDOFF.md — MaintAI Studio 最终交接文档

> **用途。** 这是 Project_ABB（MaintAI Studio）在 Portfolio v1.0 冻结后的**唯一交接上下文**。
> 把它整体喂给未来的 ChatGPT，即可获得关于该项目的完整、诚实、可审计、可用于学习/简历/面试的事实基础。
> 本文不修改任何代码、产物、测试、配置或 README；只汇总并固化以下权威来源的事实：
> `PROJECT_ABB_REVIEW_MERGED.md`、`docs/CURRENT_STATUS.md`、`docs/BENCHMARKS.md`、
> `docs/AGENT_EVALUATION.md`、`docs/INTERVIEW_GUIDE.md`、`README.md`。
> **冲突时以 `docs/CURRENT_STATUS.md`（snapshot 2026-09-01）为当前权威。**
> 所有数字均为 **FACT FREEZE（事实冻结基线）**，不得为美化而重算、重训、调参或修改 Agent。

---

# 1. 一页速览

**项目定位。** MaintAI Studio 是 ABB Accelerator 2026 Theme 1（AI-powered AutoML + MLOps Copilot for Industrial Equipment）的**竞赛 + 作品集原型**，Python 3.11 **模块化单体（modular monolith）**，不是生产软件。

**解决的问题。** 预测性维护往往在建模之前就失败：工业传感器数据脏、异质；非 ML 工程师不知道这是分类还是回归任务；评估指标与实际维护成本脱节；notebook 模型几乎不会变成受监控的工作流。绝大多数 "AutoML demo" 退化成 `上传 CSV → 训 XGBoost → 秀 accuracy`，这不是维护工程师的真实工作方式。

**用户与系统。** 维护工程师通过 Streamlit 控制室上传数据，经唯一入口 FastAPI HTTP API 驱动整个闭环：`ingest → profile → task → train → explain → register → predict → UI → audit`。P1 增加三条受控路径：`monitoring → retraining recommendation`（不自动部署）、`approval → champion/mock CMMS`，以及 prediction 上的 append-only feedback。Postgres 存业务元数据 + 审计，MLflow 独立做 tracking/registry，SQLite 仅作测试替身。

**Agent 本质。** 一个 LangGraph 编排的 `route → tool → synthesize` 状态图 Copilot：确定性关键词意图路由 → 固定七个只读工具的 allowlist → 基于工具证据渲染 + 量化 grounding 门禁。Agent 只读、只解释、只提案，绝不写文件/SQL/registry/DB，不执行任何 mutation。

**ML / MLOps。** 确定性 Python 服务负责 profiling/quality/leakage/task framing/split/train/evaluate/SHAP/recommend/MLflow 注册/推理；P1 加 monitoring(drift/anomaly)、approval、champion 生命周期、feedback、mock CMMS。Agent 不做任何数值计算。

**Top 3 结果。**
1. Scania APS 外部 benchmark（真实公开卡车数据）：XGBoost 以 PR-AUC/AP `0.927`、F1 `0.831`、官方成本 `48,660` 胜出（阈值 0.5、无调参）。
2. 确定性软件契约 + 回归安全：`684/684` 测试通过（2026-09-01），`ruff`/`pip check`/`docker compose config` 全过。
3. Agent 路由与安全被验证：tool selection `8/8`、unsafe-action 拦截 `2/2`。

**Top 3 局限。**
1. Agent grounded numeric synthesis 失败：`0/6`，task success 仅 `3/10`（MockProvider 固定叙述不含数字）。
2. 无生产部署/无 live-LLM 质量评估/无企业认证/RBAC/无真实 CMMS。
3. 官方 Scania test 已被三模型后验排名消费，不再是 untouched holdout；Docker runtime 当前重跑未验证（历史 P0 通过）。

---

# 2. 项目最终定位

## 它是什么

- 一个**竞赛 + 作品集原型（competition + portfolio prototype）**：用于证明一套完整的工业预测性维护 + Agentic Copilot + MLOps 闭环可以**端到端确定性运行**，并把工程决策与验证边界讲清楚。
- 一个 **Python 3.11 模块化单体**：FastAPI 唯一业务入口，Streamlit 只走 HTTP，Postgres（业务元数据 + 审计）与 MLflow（tracking/registry）分离，SQLite 仅测试。
- 一个 **确定性 ML 管线 + 受约束的 LangGraph Copilot + HITL 人机回路** 的演示系统。

## 它不是什么

- **不是 Production AI Platform**：`demo_deployed` 与 approval-gated champion 生命周期都是**演示能力**，不是生产 serving；没有 Production stage、没有生产连接器。
- **不是 Autonomous Agent**：Agent 只读、只提案，不执行；`agent proposes, deterministic services execute, human approves high-risk actions`。
- **不是 Fully LLM-driven planner / tool selector**：意图路由是**确定性关键词规则**（中英文、固定优先级），不是 LLM 自主规划与自由选工具。
- **不是企业级 RBAC / 认证**：审批门禁是 localhost demo gate（bearer token + 自报 `X-Human-Actor-ID`），无 SSO/OIDC/目录/角色/限流。
- **不是真实 CMMS**：CMMS 是 mock，`mock: true` + 免责声明，绝不联系真实维护系统。
- **不是生产部署 / 线上学习**：无重训练自动部署、无在线学习、无真实工业连接器（OPC UA/Kafka/edge）。

**一句话中文定义：** MaintAI Studio 是一个面向工业设备的预测性维护 + MLOps Copilot 竞赛原型，用确定性 Python 服务完成数据到模型注册的闭环，用一个受约束的 LangGraph Copilot 只读解释并提案，所有高风险动作由人工审批。

**One-sentence English definition:** MaintAI Studio is a competition/portfolio prototype of an agentic predictive-maintenance & MLOps copilot that closes the ingest→train→explain→register→predict→audit loop with deterministic Python services, while a constrained LangGraph copilot only reads, explains, and proposes — and every high-risk action requires human approval.

---

# 3. 为什么做这个项目

工业预测性维护的真实难点不在模型，而在**建模之前**与**建模之后**。本项目逐条把这些工程现实做成机制，而不是营销口号：

1. **工业数据脏、异质（dirty / heterogeneous）。** 传感器存在缺失、flatline（恒值）、离群点、重复、时间戳乱序、多资产混合。系统在建模前先做 schema inference、missingness、cardinality、数值统计、类别频率，以及 ≥6 项质量检查（constant/flatline/outlier/timestamp/asset imbalance/target imbalance/sample count）→ health score。演示数据故意注入 flatline、torque 离群、缺失、重复行来压测质量路径。

2. **任务不明确（task framing）。** 非 ML 工程师不知道数据对应分类、多分类还是回归。系统用确定性规则引擎推断 `binary_classification` / `multiclass` / `regression`（Scania 被推断为 `binary_classification`，置信度 0.70）。

3. **极端不平衡（imbalance）。** 故障是稀有事件：Scania test 正类仅 `375/16,000`（约 2.3%），train 正类 `1,000/60,000`（约 1.7%）。Accuracy 会被占 97%+ 的负类主导——「永远预测负类」也能拿 ~97.7% accuracy 却漏掉全部故障，因此主指标用 PR-AUC（Average Precision），不用 accuracy。

4. **非对称成本（asymmetric cost）。** 漏报（FN，真实故障被判健康）远比误报（FP，健康车送检）昂贵。官方 IDA 2016 challenge 成本 `Total Cost = 10*FP + 500*FN`，FN 是 FP 的 **50 倍**。**注意：10/500 是官方 challenge 成本单位，不是 ABB/本项目的真实维护经济参数。**

5. **生命周期（lifecycle）。** 模型不是训完就结束：要 tracking（每次 run 记 params/metrics/artifacts/tags）、registry（`candidate` alias）、demo deploy（只服务 demo）、推理（每次预测存 content-hash 的 immutable event，不存原始输入）、以及 P1 的 champion/challenger 晋升（审批门控）。

6. **漂移（drift）。** 上线后数据会变：P1 实现数值 PSI + KS（统计量 + p-value）+ mean/std shift、类别 PSI、severity 阈值（标注"非通用行业标准"）、无监督 anomaly（Z-score/MAD + IsolationForest）。漂移只触发**重训练建议**，绝不自动部署。

7. **人的决策（human decisions）。** 模型只做决策支持，不替代维护/安全/工程判断。SHAP 输出明确标注为"基于模型的解释，非已验证的物理根因"；模型晋升、mock CMMS 工单等高风险动作必须人工审批；所有 mutation 都写审计事件。UI 上有必需的免责声明。

---

# 4. 系统架构

```text
                        ┌─────────────────────────────────────────────┐
                        │            工程师 (Engineer / Human)          │
                        └──────────────────────┬──────────────────────┘
                                               │ HTTP
                        ┌──────────────────────▼──────────────────────┐
                        │   Streamlit UI (HTTP only)  — 只调 API      │
                        └──────────────────────┬──────────────────────┘
                                               │ HTTP (MAINTAI_API_URL)
                        ┌──────────────────────▼──────────────────────┐
                        │      FastAPI  (唯一业务入口)                  │
                        │   request_id / audit / human_gate / routers │
                        └───────┬───────────────┬──────────────┬──────┘
                                │               │              │
              ┌─────────────────▼───┐   ┌───────▼────────┐  ┌──▼─────────────┐
              │  PostgreSQL         │   │  MLflow        │  │  LangGraph      │
              │  (业务元数据 + 审计) │   │ (tracking +    │  │  Copilot        │
              │  SQLite=测试替身     │   │  registry)     │  │ (只读工具)      │
              └─────────────────────┘   └────────────────┘  └─────────────────┘

P0 管线: ingest → profile → task(leakage) → train → evaluate → recommend
        → register(candidate) → demo-deploy → predict
P1 分支: monitoring → retraining recommendation（不自动部署）
         approval → champion lifecycle / mock CMMS
         prediction → append-only feedback（不触发在线学习）
```

五层职责（每层：输入 / 组件 / 输出 / 为什么这样设计）：

| 层 | 输入 | 组件 | 输出 | 为什么 |
|---|---|---|---|---|
| **UI / API** | 工程师的浏览器操作 | Streamlit UI + FastAPI 路由 | HTTP 响应、UI 渲染 | Streamlit 只走 HTTP、从不 import ML 内部或 DB，守住"FastAPI 唯一业务入口"边界，便于测试与替换前端 |
| **Data / ML** | 数据集（CSV/Parquet） | `data/`（profile/quality/leakage/split）+ `tasks/`（infer）+ `ml/`（catalog/train/evaluate/recommend/cost/explain/confidence） | profile 报告、质量分、任务推断、模型、指标、SHAP、推荐 | 确定性 Python 做全部数值计算，Agent 不碰计算；泄漏感知 split 与 fit-on-train-only 保证评估无偏 |
| **MLOps** | 训练 run、推荐结果、推理请求 | `mlops/`（tracker/registry）+ `application/`（models/predictions/lifecycle） | MLflow run/版本、`candidate`/`demo_deployed` alias、预测事件 | tracking/registry 与业务 DB 分离；demo deploy 只是 demo 服务，语义上绝不等于生产 |
| **Agent / Copilot** | 自然语言问题 | `agent/`（graph/state/tools/provider/service） | 基于工具证据的解释、动作提案 | 确定性路由 + 固定只读工具 allowlist + grounding 门禁，把 LLM 限制在"解释确定性输出、提案"的窄边界内 |
| **Human / Safety** | 审批决策、审计 | `approvals/`、`feedback/`、`cmms/`、`audit/`、`human_gate` | 审批状态、工单草稿、审计事件 | 高风险动作永远停在"提案→人批准→执行"之间，绝不自动越过人 |

---

# 5. End-to-End Workflow

完整流程（P0 全闭环 + P1 决策回路），**如实描述、不含假能力**：

1. **Upload（上传）** — 扩展名 allowlist（`.csv`/`.parquet`）、200 MB 大小限制、文件名 sanitize、路径穿越拒绝、SHA-256 去重、受控存储路径。
2. **Profile（画像）** — schema 推断、缺失率、cardinality、数值统计、类别频率。
3. **Quality（质量）** — ≥6 项检查（constant/flatline/outlier/timestamp/asset imbalance/target imbalance/sample count）→ health score。
4. **Leakage（泄漏）** — 目标泄漏检测 → `safe`/`warning`/`block`；被 block 的特征默认排除（demo 中 `serial_no` 被 block）。
5. **Task framing（任务推断）** — 确定性规则引擎推断 `binary_classification`/`multiclass`/`regression`。
6. **Train（训练）** — 单个 in-process worker（FastAPI BackgroundTasks + 进程级协调锁，最多同时一个实验）；分类 LR/RF/XGBoost，回归 Ridge/RF/XGBoost；所有 run 记入 MLflow。
7. **Evaluate（评估）** — Precision/Recall/F1/ROC-AUC/PR-AUC(AP)/confusion matrix。
8. **Recommend（推荐）** — 确定性 best-model 算法（非 LLM），可选 cost-aware（`ml.cost.compare_costs`）。
9. **Register（注册）** — MLflow registry gateway，`candidate` alias。
10. **Demo deploy（演示部署）** — `demo_deployed` alias 只翻一个版本；**仅 demo 服务，不是生产，绝不设 champion/Production**。
11. **Predict（预测）** — 单条 + 批量推理，只针对 `demo_deployed` 模型；每条存为 immutable `PredictionEvent`（仅 content hash，不存原始输入）；带 per-record 置信度 + 本地 SHAP 解释。
12. **Monitor（监控）** — anomaly（Z-score/MAD + IsolationForest）、drift（PSI/KS/mean/std）。
13. **Approval（审批）** — `pending → approve/reject/modify` 状态机，version-based CAS；审批**只决定、不执行**。
14. **Champion / Feedback / mock CMMS** — champion 晋升与 mock CMMS 受审批门控：前者移动 `champion` alias、归档旧版本并写幂等收据，后者从已审批 `cmms_work_order` 起草明确标记为 mock 的工单。Feedback 是 prediction 上独立的 append-only 记录，不需要上述 action approval，也不触发在线学习。

**无假能力清单：** 无重训练自动部署（仅建议）、无 P1 one-command E2E、无 Alembic 迁移（仅 `create_all`）、无真实 CMMS、无企业认证、无生产部署。

---

# 6. Agent 部分必须真正讲懂

**为什么用 LangGraph？** 因为这里要的是**可控的图结构**，而不是让 LLM 自由发挥。LangGraph 把 "规划、选工具、解释" 做成一条确定性的 `route → tool → synthesize` 状态图（`CopilotState`），并在图上强制安全不变量：**agent 只读、只提案**。

**三步图：**
- **route** — 确定性意图路由器（中英文关键词、固定优先级），把请求映射到**唯一**一个读意图或动作意图。
- **tool** — 把意图分派到**唯一**一个 allowlist 只读工具，参数固定。
- **synthesize** — 从工具 evidence 构建最终答案；动作意图返回 `proposed_action`（`status: not_executed` + `approval_required`/`manual_steps`），**绝不执行**。

**CopilotState。** 贯穿图的状态对象，携带 user request、路由后的 intent、工具 evidence、最终 narrative 与 proposed_action，保证每步输入输出可追踪、可测试。

**确定性关键词意图路由器。** 不是 LLM 选工具：路由是纯规则（中文 + 英文关键词、固定优先级）。这带来可测试、可回归的路由行为（tool selection `8/8`），也意味着它**不是**"LLM 自主 planner/tool selector"。

**七个 allowlist 只读工具（`src/maintai/agent/tools.py`），固定参数、path-free：**
`dataset_profile`、`dataset_quality`、`dataset_task`、`experiment_results`、`experiment_comparison`、`model_deployment_status`、`prediction_explain`。`_public_dataset` 会剥离 `file_path`/`file_hash`/`original_filename`/`size_bytes` 等内部字段。

**工具边界。** 没有 `**kwargs`、没有动态 import、不暴露 repository/session/SQL；动作意图（`action_train`/`action_register`/`action_deploy`/`action_production_promotion`/`action_work_order`）**永不执行**。Agent 从不读文件、从不写 SQL、从不直接改 artifact/registry/DB。

**Provider 抽象（`src/maintai/agent/provider.py`）。**
- `MockProvider` — 离线、确定性、无密钥的**默认** provider；其 narrative 是一句固定话术，**不含任何数字、不含任何模型名**，所有量化数字只来自工具 evidence。
- `OpenAICompatibleProvider` — 任意 OpenAI-compatible `/chat/completions` 端点（`httpx`）；只在配置了且存在 key 时才启用，否则回退 mock；key 绝不 log/raise/`repr`。**此 provider 未被评估（not evaluated）。**

**量化 grounding 门禁。** 最终 narrative 里出现的每个数字都必须是工具 evidence 数字的子集，否则被拒绝。这阻止了幻觉数字，但也暴露了薄弱点（见 §10）。

**动作提案 + 人工审批。** 高风险动作只产出 `proposed_action`，由 P1 审批状态机决定；Agent 侧只提案，不执行。

**必须讲清楚、不能混淆的表述。** 本项目 Agent **不能**被称为 "LLM autonomous planner / tool selector"。**推荐短语（原样）：** `structured LangGraph Copilot with deterministic routing and constrained tools`。

**Agent 不是自主的。** 无 P1 完整 action/cost/dual-experiment 覆盖（`cost_lowest`、`compare_two_experiments` 工具不存在，benchmark 中为 N/A）。已知**拒绝话术陈旧**（`graph.py` 部分文本仍说某些 P1 promotion/CMMS 路由不可用，而路由实际已存在；Agent 仍不能执行它们，文本在 fact freeze 期间未修）。

---

# 7. ML / MLOps 部分必须真正讲懂

**确定性 Python 做全部数值计算，Agent 不碰任何 compute。** 逐模块：

- **profiling** — schema 推断、缺失率、cardinality、数值统计、类别频率（Scania train 缺失率实测 8.2847%）。
- **quality** — ≥6 项工业检查 → health score；**注意**：health score 不是跨数据集可训练性门禁（Scania train 得 0/100，三个模型仍都训练成功）。
- **leakage** — 目标泄漏检测 → `safe`/`warning`/`block`，block 特征默认排除。
- **task inference** — 规则引擎 `binary_classification`/`multiclass`/`regression`。
- **split** — 确定性 split + 泄漏感知 train/test；外部 benchmark 保留官方 holdout，全部 transformer 只 fit train。
- **models** — 分类：Logistic Regression / Random Forest / XGBoost；回归：Ridge / RF / XGBoost。
- **SHAP** — tree/linear SHAP，permutation-importance 回退；global + local（标注"非已验证物理根因"）。
- **confidence** — `predict_proba`（+ 可选 calibration）；回归用经验残差区间。
- **recommendation** — 确定性 best-model 算法（非 LLM）；按主指标 → 复杂度 → 延迟排序，可选 cost-aware。
- **MLflow tracking** — 每个 run 记 params/metrics/artifacts/tags。
- **registry** — MLflow registry gateway，`candidate` alias。
- **candidate** — 推荐 run 注册为 `candidate`。
- **demo_deployed** — demo 服务别名，只翻一个版本，**不是生产**。
- **monitoring/drift** — anomaly（Z-score/MAD + IsolationForest）、drift（PSI/KS/mean/std/categorical PSI）。
- **champion/challenger** — approval-gated 晋升（demo 能力）：`request_promotion`（challenger + approval 提案）、`execute_promotion`（移动 `champion` alias、归档旧版本、幂等收据、审计）。

**边界不变量：** `agent proposes, deterministic services execute, human approves high-risk actions.` 写操作只经 FastAPI/UI 且写审计；promotion 与 mock CMMS executor 必须经人工审批。

---

# 8. Scania APS Benchmark

**数据集（frozen facts）。**
- 名称：APS Failure at Scania Trucks（重型卡车空气压力系统故障）。
- 来源：UCI Machine Learning Repository，**DOI `10.24432/C51S51`**；页面 https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks 。
- **真实公开卡车数据，不是 ABB 数据**，也不属于合成 demo。创建方 Scania CV AB。
- **特征数差异（frozen）**：UCI 页面标 171 特征，实际 archive 为 **171 列 = `class` + 170 匿名特征**。以 archive 为可执行事实，记录差异，**不伪造第 171 列**，不猜测匿名列业务含义。
- 标签编码：`neg → 0`，`pos → 1`（目标 `aps_failure`）。

**官方 split（严格保留，无随机 split）。**

| Split | Rows | Negative | Positive | Missing-cell rate |
|---:|---:|---:|---:|---:|
| Train | 60,000 | 59,000 | 1,000 | 8.2847% |
| Test | 16,000 | 15,625 | 375 | 8.3582% |

**Preprocessing。** 复用 P0 管线：数值中位数 imputation；Logistic Regression 数值路径额外 standard-scaled。**全部 transformer 只 fit official train**；`na` token 先转 `NaN`；不做猜测性特征工程。

**协议。** 阈值 `0.5`（三模型一致）；**无 threshold optimization、无超参/阈值调优**；PR-AUC 字段即 `sklearn.average_precision_score`（Average Precision）；test 被三模型后验排名消费（不是 untouched holdout）。

**结果（frozen，2026-09-01，official test，threshold 0.5，无调参）。**

| Model | Precision | Recall | F1 | PR-AUC (AP) | ROC-AUC | FP | FN | Total Cost |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.837580 | 0.701333 | 0.763425 | 0.816266 | 0.976923 | 51 | 112 | 56,510 |
| Random Forest | 0.946281 | 0.610667 | 0.742301 | 0.897315 | 0.995625 | 13 | 146 | 73,130 |
| XGBoost | 0.945578 | 0.741333 | 0.831091 | 0.927419 | 0.996165 | 16 | 97 | **48,660** |

Confusion matrices (TN/FP/FN/TP)：LR `15,574/51/112/263`；RF `15,612/13/146/229`；XGB `15,609/16/97/278`。

**成本函数：** `Total Cost = 10 * FP + 500 * FN`（10/500 是 IDA 2016 challenge 官方成本单位，非 ABB/本项目维护经济参数）。

**关键解读。**
- **XGBoost** 在 PR-AUC、F1、总成本三项同时胜出（48,660），不是牺牲指标换成本。
- **Random Forest** precision 最高（0.946281）、FP 最少（13），但 recall 最低（0.610667）、FN 最多（146），成本 73,130（最高）——证明高精度/少误报的"保守"模型可能因漏掉大量真实故障而最贵。
- **Accuracy 不适合**：正类仅 375/16,000，永远预测负类也能 ~97.7% accuracy。
- **FN 是 FP 的 50 倍**来自官方 challenge，不是 ABB 经济参数；FN 主导成本（RF 146×500=73,000）。
- 这是**固定 catalog + 默认超参 + 阈值 0.5 的基线**，不是与 IDA 2016 参赛方案的公平 leaderboard 比较。

---

# 9. Evaluation Framework

三类评估，各自度量不同东西，**不得混写**：

1. **ML Quality（模型质量）** — 度量指标：Precision/Recall/F1/ROC-AUC/PR-AUC(AP)/confusion matrix/FP/FN/Total Cost。回答"固定 catalog 在官方 holdout 上表现如何"。主指标 PR-AUC（Average Precision），选择依据含成本函数。
2. **Agent Quality（Agent 质量）** — 度量指标：Tool Selection Accuracy、Grounded Numeric Answer Accuracy、Unsupported Numeric Claim Rate、Unsafe Action Block/Proposal Compliance、Task Success Rate。回答"路由、安全、grounded synthesis 各自如何"。当前是 `MockProvider` 下的**确定性 mock 回归**，不是 live LLM 准确率。
3. **Software / System Quality（软件/系统质量）** — 度量指标：`pytest -q` 通过数、`ruff check .`、`pip check`、`docker compose config --quiet`、历史 Docker runtime + restart persistence。回答"确定性软件契约是否正确、是否回归安全"。

**684 tests 证明什么 / 不证明什么（精确）**：`684/684` 证明**确定性契约 + 回归安全**（profiling/quality/leakage/task framing/split/train/eval/SHAP/registry/inference/monitoring/drift/approval/feedback/mock CMMS 按契约运行；加入 benchmark track 与 P1 切片后无回归）。它**不证明** Agent 准确率、生产能力、泛化能力、live LLM 质量、企业认证、真实 CMMS。一句话：**684 验证确定性契约与回归，不验证生产能力或泛化。**

---

# 10. 当前 Agent Baseline 的真实问题

Frozen MockProvider 基线（12 个固定探针 = 10 计分 + 2 N/A，`cost_lowest`/`compare_two_experiments` 因工具不存在而 N/A）：

| Metric | Numerator | Denominator | Value |
|---|--:|--:|--:|
| Tool Selection Accuracy | 8 | 8 | 100% |
| Grounded Numeric Answer Accuracy | 0 | 6 | 0% |
| Unsupported Numeric Claim Rate | 0 | 1 | 0% |
| Unsafe Action Block / Proposal Compliance | 2 | 2 | 100% |
| Task Success Rate | 3 | 10 | 30% |

**逐项解读。**
- **路由正常**：8/8 读探针选对 intent/tool，allowlist 行为符合设计。
- **安全正常**：2/2 生产晋升探针被拦截，返回 `not_executed` + `approval_required`，无任何 tool/write 调用。
- **证据里有正确数字**：tool evidence 忠实渲染了 benchmark artifact 里的真实数值（如推荐模型 primary metric `0.9274190305308689`）。
- **固定 Mock 叙述不含数字/模型名**：`MockProvider.invoke` 忽略输入，返回一句固定 safety narrative（"Here is the grounded answer, derived solely from the deterministic tool evidence below…"），不 restate 任何数字，所以 `numeric_ok` 对 6 个数值探针全 false → **0/6**。这是确定性 mock 的性质，不是 benchmark 数据缺陷。
- **最终合成/任务完成失败**：grounded numeric 0/6 + `nonexistent_metric`（没编造数字——安全正确，但也没明确说"MSE 不可用"，仍判 FAIL）→ task success 仅 **3/10**。

**诚实的定位：这不是简历亮点。** 路由与安全值得讲（8/8、2/2），但 0/6 与 3/10 是"改进前基线"，被冻结为未来改进的可测参照。被问到时坦率说明：确定性层（路由/取数/反幻觉门禁）坚实，grounded numeric synthesis 是当前最弱环节，且是架构边界问题（tool evidence 有数据 vs 最终叙述真正回答量化问题之间的断层），不是路由或安全的问题。

---

# 11. 关键 Architecture Decisions

| Decision | Reason | Trade-off |
|---|---|---|
| **Streamlit 只走 HTTP、FastAPI 唯一业务入口** | 守住"所有读写经 API"，前端可替换、可测试，避免 UI 直连 ML/DB | 多一跳 HTTP；UI 无法直接用内部对象 |
| **Postgres（业务+审计）与 MLflow（tracking/registry）分离** | 业务元数据与实验追踪属于不同生命周期；registry 语义独立 | 两个存储要分别保证一致；测试用 SQLite 替身 |
| **Agent 永不直接读写 Python/SQL** | 杜绝任意代码执行/agent 生成 SQL；参数化 SQL 全覆盖 | Agent 只能经 allowlist 工具取数，能力受限于七工具 |
| **只读工具 + 动作提案（而非执行）** | 高风险动作必须人批；Agent 无写路径 | 无法自动闭环，重训练部署未实现 |
| **`demo_deployed` ≠ 生产** | 明确演示服务边界，避免误称生产部署 | 不覆盖真实工厂 serving/champion/Production |
| **审批 ≠ RBAC** | 用 demo gate（bearer + actor 头）演示 HITL 契约，而非冒充企业认证 | 无 SSO/OIDC/目录/角色/限流，仅 localhost 可信环境 |
| **mock CMMS ≠ 真实 CMMS** | 演示工单闭环与幂等收据，不接触外部系统 | 无 Fiix/SSP 连接器/token |
| **无 Kafka/Celery/Redis** | P0 训练用单 in-process worker + 进程级锁，控制复杂度 | 无并发/分布式训练扩展 |
| **无 Multi-Agent** | 单 Copilot + 确定性服务足够覆盖 P0/P1 切片，避免过度设计 | 无多角色协作/分工 |
| **模块化单体（modular monolith）** | 逻辑分层清晰、可测试，部署单一 | 非微服务，横向扩展受限 |
| **`create_all` 引导 schema（无 Alembic）** | 原型阶段简化；Alembic 显式 deferred | 无法迁移既有数据库 |

---

# 12. 最终 Scope Freeze

## 已完成且保留

- P0 全闭环：ingest → profile → task(leakage) → train → evaluate → recommend → register(candidate) → demo-deploy → predict → UI → audit（P0 Freeze 2026-08-30 通过；历史 live HTTP E2E + restart persistence 通过）。
- P1 决策回路：monitoring(anomaly/drift/replay)、cost comparison、retraining recommendation、approvals、champion/challenger 生命周期、feedback、mock CMMS、consolidated P1 UI（demo 能力）。
- Scania APS 外部 benchmark（XGBoost PR-AUC 0.927 / F1 0.831 / cost 48,660）。
- Baseline Copilot 评估（8/8、0/6、0/1、2/2、3/10）。
- 确定性测试：684/684（2026-09-01）+ ruff/pip check/docker compose config 全过。
- LangGraph Copilot（route→tool→synthesize、七只读工具、grounding 门禁、动作提案）、provider 抽象（MockProvider 默认 + OpenAI-compatible env provider）。

## 已知缺陷但不再修

- **Grounded numeric synthesis 失败（0/6）**：固定 Mock 叙述不含数字，不 restate 证据中的正确数值。
- **无 live LLM 评估**：MockProvider 为默认，live provider N/A。
- **确定性路由器局限**：关键词规则，非语义理解/非 LLM 自由选工具。
- **缺 P1 cost/dual-experiment 工具**：`cost_lowest`、`compare_two_experiments` 为 N/A。
- **Quality score 可移植性差**：Scania train health 0/100 仍能训练——health score 不是跨数据集可训练性门禁。
- **官方 test 已消费**：被三模型后验排名消费，不能当 untouched holdout。
- **无企业认证/RBAC**：审批是 localhost demo gate。
- **无生产部署**：demo_deployed/champion 生命周期仅演示。
- **无真实 CMMS**：mock only。
- **Docker runtime 当前重跑未验证**（daemon 不可用）；历史 P0 runtime + restart 通过。
- **Retraining deployment executor 缺失**（仅推荐，不自动部署）。
- **P1 one-command E2E 缺失**（P0 有 one-command demo，P1 无等价脚本）。
- **Alembic 迁移缺失**（仅 `create_all`）。
- **Agent 拒绝话术陈旧**（graph.py 部分文本说 P1 promotion/CMMS 路由不可用，路由已存在）。

## Future Work / 面试中可以说未来扩展

（**只属于未来，不是当前能力**；被问"接下来做什么"时据此回答，但必须标明"未实现、待验证"。）

- Grounded numeric synthesis：让 provider 稳定输出受 evidence 约束且命中具体数字的答案。
- Live LLM evaluation：把 N/A 换成真实 provider 评估（新增节，不覆盖 mock 基线）。
- Unsupported-metric handling：对不存在指标明确回答"不可用"。
- 补 cost-comparison 与 dual-experiment 工具（消除两个 N/A）。
- Retraining deployment executor、P1 one-command E2E、Alembic 迁移。
- P2 方向（仅方向，未开始）：C-MAPSS RUL、深度时序模型、Kafka streaming、edge ONNX、OPC UA live connector、真实 CMMS connector 抽象、RAG over maintenance manuals、多模态诊断、digital twin、cloud/K8s、因果诊断。

---

# 13. 简历 Claim 红线

## 可以写

- "实现了一个 **structured LangGraph copilot**（`route → tool → synthesize` 状态图 + 固定七只读工具 allowlist + 量化 grounding 门禁）。"
- "高危动作走 **human-in-the-loop**：Agent 只产出 `proposed_action`，由人工审批门禁决定，绝不自动执行。"
- "在公开外部数据集（UCI Scania APS，DOI `10.24432/C51S51`）上跑通 profiling→task→train→evaluate→recommend，XGBoost 取得 F1 `0.831`、PR-AUC/AP `0.927`、官方成本 `48,660`。"
- "确定性 Python 服务完成数据质量/泄漏/任务推断/训练/SHAP/recommend/MLflow 注册/推理，`684/684` 测试通过。"

## 建议谨慎写

- "Agent 路由与安全经固定场景回归验证（tool selection 8/8、unsafe-action 2/2）"——可写，但要同时说明**不**代表端到端任务成功率（3/10）或 live LLM 质量。
- "cost-aware 模型选择"——准确说法是"复用官方 challenge 成本做固定 catalog 后验排名"，不是"成本优化求解器"。
- "监控/漂移/审批/反馈/mock CMMS 闭环"——都是 **demo 能力**，写时需加"demo/prototype"限定。

## 绝对不能写

- ❌ "LLM 自主规划并执行 ML/数据操作"（实际是确定性路由 + 只读工具，Agent 不执行任何 mutation）。
- ❌ "已生产部署 / Production serving"（实际只有 `demo_deployed`，明确非生产）。
- ❌ "企业级认证 / RBAC / SSO"（实际是 localhost demo gate）。
- ❌ "对接真实 CMMS / 真实维护系统"（实际是 mock CMMS）。
- ❌ "经过 live LLM 验证 / 真实 LLM 回答质量达标"（实际 live provider N/A）。
- ❌ "业界 SOTA / 预测性维护最优"（固定 catalog + 默认超参 + 阈值 0.5，非公平 leaderboard 比较）。
- ❌ "官方 test 仍是 untouched holdout"（test 已被后验排名消费）。
- ❌ 把 0/6 或 3/10 写进简历 bullet（这属于诚实访谈范畴，不是卖点）。

---

# 14. 中文简历最终推荐版本

### MaintAI Studio — Agentic Predictive Maintenance & MLOps Copilot

**时间 / 项目属性：** 2026 ｜ ABB Accelerator 2026 Theme 1 ｜ 竞赛 + 作品集原型 ｜ Python 3.11 模块化单体（FastAPI / Postgres / MLflow / Streamlit / Docker Compose / LangGraph）

- 面向工业预测性维护，构建**从数据到模型注册再到推理的完整 MLOps 闭环**（ingest → profile → task → train → explain → register → predict → audit），以 FastAPI 为唯一业务入口，Streamlit 仅走 HTTP，Postgres 存业务元数据 + 审计、MLflow 独立做 tracking/registry，确定性 Python 服务完成质量检查、泄漏检测、任务推断、训练、SHAP 解释与推荐。
- 实现一个 **structured LangGraph Copilot**：确定性意图路由 + 固定七只读工具 allowlist + 量化 grounding 门禁，Agent 只读、只解释、只提案；模型晋升、mock CMMS 等高风险动作走**人工审批（human-in-the-loop）**，并落地 monitoring/drift、champion/challenger 生命周期与 feedback 的演示闭环。
- 在公开真实工业数据集 **UCI Scania APS**（60K 训练 / 16K 测试，官方 holdout，DOI `10.24432/C51S51`）上验证核心管线：以 `Total Cost = 10·FP + 500·FN` 的官方成本 + PR-AUC 双维度做 cost-aware 模型选择，**XGBoost 取得 F1 0.831、PR-AUC/AP 0.927、总成本 48,660**，并说明在不平衡 + 非对称成本下 accuracy/precision 会误导模型选型。
- 工程化保障：`684/684` 测试通过（`pytest`），`ruff`/`pip check`/`docker compose config` 全过，历史 P0 全链路 HTTP E2E 与 API/MLflow 重启持久化验证通过。

---

# 15. 30 秒 / 90 秒 / 3 分钟项目介绍

## 30 秒

我做的是 ABB Accelerator 2026 Theme 1 的一个预测性维护 + MLOps Copilot 原型，叫 MaintAI Studio。它把工业设备的脏数据从 profiling、质量检查、泄漏检测、任务推断，一路走到训练、SHAP 解释、MLflow 注册和推理，形成完整闭环；外层套一个 LangGraph Copilot，确定性路由到七个只读工具，只解释、只提案，高风险动作要人工审批。在公开的 Scania APS 卡车数据集上，XGBoost 拿到 F1 0.831、PR-AUC 0.927、官方成本 48,660。它不是生产系统，是演示验证能力与边界的原型。

## 90 秒

这是一个竞赛 + 作品集原型，目标是解决预测性维护"在建模前就失败"的问题：数据脏、任务不明确、评估和成本脱节、模型上线后没人管。技术上分三块。第一块是确定性数据/ML 管线：profile、≥6 项质量检查、目标泄漏检测、规则引擎推断分类/回归、三种模型训练、SHAP 解释、确定性推荐。第二块是 MLOps：每个 run 记 MLflow，注册 `candidate`，`demo-deploy`（仅 demo 服务，不是生产），推理存 content-hash 事件，P1 加监控、漂移、审批、champion 生命周期、feedback、mock CMMS。第三块是 Copilot：LangGraph 的 route→tool→synthesize 图，确定性关键词路由到七个只读工具，量化 grounding 门禁阻止幻觉数字，动作只提案不执行。外部验证用 UCI Scania APS（60K/16K 官方 holdout），成本函数 10·FP+500·FN，XGBoost 胜出：F1 0.831、PR-AUC 0.927、成本 48,660。诚实地说，684 测试验证的是确定性契约与回归；Agent 路由和安全是好的（8/8、2/2），但 grounded 数值合成是 0/6，这是冻结的"改进前基线"。

## 3 分钟

**背景与问题。** ABB Theme 1 要一个 AutoML + MLOps Copilot。我的观察是：预测性维护通常不是模型不行，而是建模之前的数据、任务、成本问题没人解决，建模之后又没人监控。所以我把重点放在"完整闭环 + 人机回路 + 可审计"，而不是又一个"上传 CSV 训 XGBoost 秀 accuracy"的 demo。

**架构。** Python 3.11 模块化单体。FastAPI 唯一入口，Streamlit 只走 HTTP，Postgres 存业务元数据 + 审计，MLflow 独立做 tracking/registry，SQLite 只做测试。数据管线确定性：profile → quality（≥6 检查）→ leakage（block 目标泄漏特征）→ 规则引擎任务推断 → 时间/分组感知 split → LR/RF/XGBoost 训练 → 评估 → SHAP（标注"非物理根因"）→ 确定性推荐。MLOps：tracking → `candidate` 注册 → `demo_deployed`（明确不是生产）→ 预测（存 hash，不存原始输入）→ P1 监控/drift → 审批 → champion 生命周期 → feedback → mock CMMS。

**Agent。** LangGraph 编排 route→tool→synthesize。路由是确定性关键词规则（中英文），不是 LLM 自由选工具；分派到七个固定只读工具；合成层用量化 grounding 门禁，叙述里任何数字都必须是证据子集；动作意图只返回 `proposed_action`，绝不执行。Provider 抽象默认 MockProvider（固定叙述、无数字无模型名），OpenAI-compatible 只是 env 配置、未评估。

**外部验证。** UCI Scania APS，真实卡车数据，官方 holdout 60K/16K 严格保留，transformer 只 fit train，阈值 0.5 无调参。成本 `10·FP + 500·FN`（官方 IDA 2016 单位，FN 是 FP 的 50 倍）。结果：XGBoost PR-AUC 0.927、F1 0.831、成本 48,660 三线胜出；Random Forest precision 最高但 FN 最多、成本最差，正好说明不对称成本下不能只看 precision/accuracy。

**诚实边界。** 684 测试证明确定性契约与回归安全；Agent 路由 8/8、安全 2/2，但 grounded numeric 0/6、task success 3/10，这是冻结的改进前基线。没有生产部署、没有 live LLM 评估、没有企业认证、没有真实 CMMS。官方 test 已被后验排名消费，不能再当 untouched holdout。这些边界我在文档里都写死了。

---

# 16. 面试必会 20 问

1. **为什么选 Scania APS？**
   - **30 秒回答：** 它是 UCI 上真实的重型卡车运营数据（DOI 10.24432/C51S51），独立于合成 demo，避免"自己出题自己验证"；官方给了 `10·FP+500·FN` 成本公式，能验证 cost-aware 选型；正类约 2%，正好压测 PR-AUC 与代价驱动逻辑。
   - **深挖方向：** 固定 catalog + 默认超参 + 阈值 0.5 只是基线，非 leaderboard 公平比较；不是 ABB 数据。

2. **为什么不用 Accuracy 当主指标？**
   - **30 秒回答：** 正类仅 375/16,000，永远预测负类也有 ~97.7% accuracy，却漏掉全部故障。主指标用 PR-AUC（实现为 average_precision_score），成本函数让 FN 主导。
   - **深挖方向：** precision/recall/F1/ROC/PR-AUC 各自含义；不平衡下的指标选择。

3. **为什么 FN 成本是 FP 的 50 倍？**
   - **30 秒回答：** 官方公式 `10·FP + 500·FN`，500/10=50，是 IDA 2016 challenge 明示的，不是我拍的。业务直觉：漏报真实故障（非计划停机/更贵维修）比误报（健康车多检查一次）贵得多。
   - **深挖方向：** 这是 challenge 单位，不是 ABB/本项目的真实维护经济参数。

4. **为什么 XGBoost cost 最低？**
   - **30 秒回答：** 成本由 FN 主导，XGBoost FN 最少（97）且 FP 只有 16：`16×10 + 97×500 = 48,660`；同时它的 F1 和 PR-AUC 也最高，主指标与成本双胜。
   - **深挖方向：** 该结论只在阈值 0.5、固定 catalog/default hyperparameters 下成立；匿名特征不支持进一步猜测物理原因。

5. **为什么 Random Forest precision 高但 cost 差？**
   - **30 秒回答：** RF precision 0.946 最高、FP 13 最少，但 recall 0.611 最低、FN 146 最多；146 个漏报 × 500 = 73,000，远超省下的 FP，成本反而最高。
   - **深挖方向：** 高精度保守模型在不平衡+不对称成本下会最贵——选型必须看成本函数。

6. **preprocessing 如何避免 leakage？**
   - **30 秒回答：** 官方 split 严格保留（前 60K 只 fit、后 16K 只评估），全部 transformer（median imputation、LR 的 standard scaling）只 fit train，`na` 先转 NaN，不做猜测性特征工程。
   - **深挖方向：** imputer/scaler 为何不能 fit 全量数据；目标泄漏检测（block 特征）。

7. **为什么不在 test 上调 threshold？**
   - **30 秒回答：** 调阈值等于拿 holdout 做模型选择，既消费 test 又失去无偏泛化估计；本 benchmark 目的是测固定 catalog 开箱即用表现，所以固定 0.5 并明确记录"无优化"。
   - **深挖方向：** 事后调参会引入 cherry-pick，破坏诚实基线。

8. **为什么 official test 已 consumed？**
   - **30 秒回答：** 三模型都在 test 上报告结果，随后 recommend/compare_costs 基于 test 结果做后验排名选出 XGBoost，所以 test 已是"本次 winner"而非"待验收的 untouched holdout"。
   - **深挖方向：** 后续任何调优需全新独立验证集；不能继续对这个 test 调参还声称无偏。

9. **684 tests 证明什么？**
   - **30 秒回答：** 确定性契约正确 + 回归安全：profiling/quality/leakage/task/split/train/eval/SHAP/registry/inference/monitoring/drift/approval/feedback/mock CMMS 按契约跑；加入 benchmark + P1 后无回归；Agent 确定性部分（8/8 路由、2/2 安全）经回归。
   - **深挖方向：** 385→630→684 是特定工作（Scania + Copilot eval）导致，不是静默重跑。

10. **684 tests 不证明什么？**
    - **30 秒回答：** 不证明生产部署、泛化、live LLM 质量、真实 Agent 任务成功率（3/10）、通用预测性维护优势、企业认证、真实 CMMS。
    - **深挖方向：** 机制存在 ≠ 质量已验证；测试是契约验证不是能力验证。

11. **为什么 Agent Task Success 只有 3/10？**
    - **30 秒回答：** 12 固定场景里 2 个 N/A（cost tool、双 experiment comparison 工具不存在），10 个计分。主因是 grounded numeric 0/6：MockProvider 固定叙述不含数字，不 restate 证据里的正确数值；加上 nonexistent_metric 没编造数字但也没明确说"不可用"。
    - **深挖方向：** 这是确定性 mock 的性质，不是 live LLM 准确率。

12. **0/6 grounded numeric 暴露什么架构问题？**
    - **30 秒回答：** 路由和取数是对的（8/8，证据里有正确数字），但合成层（provider narrative）不能稳定回答具体数字——暴露"tool evidence 有数据"与"最终答案真正回答量化问题"之间的断层，是架构边界问题，不是路由/安全问题。
    - **深挖方向：** 反幻觉门禁阻止编造但 mock 满足不了量化问题，严格评分计"未作答"。

13. **LangGraph 解决什么问题？**
    - **30 秒回答：** 把"规划、选工具、解释"做成可控图结构（route→tool→synthesize over CopilotState），强制安全不变量：只读、只提案，不写文件/SQL/registry/DB。
    - **深挖方向：** 为什么用确定性路由而非让 LLM 自由 planner；图结构与 agent-service 边界。

14. **Agent 与 deterministic ML services 的边界？**
    - **30 秒回答：** 确定性 Python 负责所有计算与写操作（profile/train/eval/SHAP/registry/inference/monitor/approval/feedback/mock CMMS）；Agent 只规划、路由、选只读工具、解释、提案。Agent 永远不能读文件/写 SQL/改 registry/执行 mutation。
    - **深挖方向：** 一句话不变量 `agent proposes, deterministic services execute, human approves`。

15. **Human Approval 是什么 / 不是什么？**
    - **30 秒回答：** 是 `pending→approved/rejected/modified` 状态机 + 版本 CAS + 审计，token 常量时间比较；审批只决定不执行，执行是独立 token 门控端点。不是企业认证/RBAC——无 SSO/OIDC/目录/角色/限流，是 localhost demo gate。
    - **深挖方向：** 审批与执行分离；把 demo token 说成 enterprise auth 是错误表述。

16. **demo_deployed 与 production 的区别？**
    - **30 秒回答：** demo_deployed 只给 demo 服务，绝不设 MLflow champion 或 Production stage；production 意味着真实工厂集成、champion alias、promoted stage、带认证推理——本项目未声称。
    - **深挖方向：** benchmark 模型没进 registry，无 production/champion 发布。

17. **MLOps 闭环各环节？**
    - **30 秒回答：** 每次 run 记 MLflow（params/metrics/artifacts/tags）→ 确定性推荐 best run → 注册 `candidate` → demo-deploy 只翻一个版本 → 推理只针对 demo_deployed、预测存 content-hash event → P1 监控/drift 触发重训练建议（不自动部署）→ 审批门控 champion 晋升。
    - **深挖方向：** candidate vs demo_deployed vs champion 三个别名语义差异。

18. **为什么用 LLM/tool calling 却又用确定性路由？**
    - **30 秒回答：** 可测试、可回归、可审计。确定性关键词路由保证 tool selection 8/8 可复现，grounding 门禁保证叙述里的数字可追溯到证据；LLM 只负责叙述，不负责选工具或执行。
    - **深挖方向：** 这决定它不能被称为 autonomous planner/tool selector。

19. **下一步最值得改什么 + pre/post 对比？**
    - **30 秒回答：** grounded numeric synthesis、live LLM eval、unsupported-metric 明确回答"不可用"、补 cost/dual-experiment 工具。基线冻结永不覆盖，改进新增独立 Post-Improvement Evaluation（append 不 overwrite），同一套固定场景保证可比。
    - **深挖方向：** 每个 metric 记定义/numerator/denominator/provider，数字回溯到 artifact。

20. **生产化还缺什么（production gap）？**
    - **30 秒回答：** 企业认证/RBAC、真实 CMMS、重训练自动部署 executor、P1 one-command E2E、Alembic 迁移、live LLM 质量验证、真实工业连接器、以及一个未被消费的最终 holdout。
    - **深挖方向：** 哪些是原型边界（OUT OF SCOPE），哪些是当前缺口（NOT IMPLEMENTED）。

---

# 17. 代码学习地图

| 优先级 | 路径 | 责任 | 面试深度 |
|---|---|---|---|
| P0 必学 | `src/maintai/agent/graph.py` + `state.py` | LangGraph `route→tool→synthesize` 状态图、确定性意图路由、grounding 门禁、CopilotState | 深（能画出图、说出路由规则与安全不变量） |
| P0 必学 | `src/maintai/agent/tools.py` | 七个只读工具 allowlist、固定参数、`_public_dataset` 剥离内部字段 | 深（背出七工具名与边界） |
| P0 必学 | `src/maintai/agent/provider.py` | MockProvider（固定叙述无数字）+ OpenAICompatibleProvider（env 配置、未评估） | 中（讲清 provider 抽象与密钥安全） |
| P0 必学 | `src/maintai/agent/service.py` | CopilotService 入口，编排 graph + provider | 中 |
| P0 必学 | `src/maintai/api/main.py`（含 routers） | FastAPI 唯一业务入口、request_id、human_gate、审批/生命周期/反馈/CMMS 路由 | 中（讲清"唯一入口"边界） |
| P0 必学 | `src/maintai/application/`（models/predictions/lifecycle） | 应用服务层：demo_deployed、推理、champion 晋升 | 中 |
| P0 必学 | `src/maintai/data/{profile,quality,leakage,split}.py` | 数据画像、≥6 质量检查、泄漏检测、泄漏感知 split | 深（fit-on-train-only 是核心） |
| P0 必学 | `src/maintai/ml/{catalog,train,evaluate,recommend,cost,explain,confidence}.py` | 模型目录、训练、评估、确定性推荐、成本比较、SHAP、置信度 | 深（背出三模型 + PR-AUC/成本逻辑） |
| P0 必学 | `src/maintai/tasks/infer.py` | 规则引擎任务推断（binary/multiclass/regression） | 中 |
| P0 必学 | `src/maintai/mlops/{tracker,registry}.py` | MLflow tracking + registry gateway、candidate alias | 中 |
| P1 建议 | `src/maintai/approvals/service.py` + `api/approvals.py` | 审批状态机（pending→approve/reject/modify）、版本 CAS | 中（讲清"只决定不执行"） |
| P1 建议 | `src/maintai/application/lifecycle.py` | champion/challenger 晋升 executor（审批门控） | 中 |
| P1 建议 | `src/maintai/monitoring/{anomaly,drift,recommend}.py` | anomaly（Z-score/MAD+IsolationForest）、drift（PSI/KS）、重训练建议 | 中 |
| P1 建议 | `src/maintai/benchmarks/copilot_eval.py` + `scripts/{run_scania_aps_benchmark,evaluate_scania_copilot}.py` | 外部 benchmark + Copilot 评估（12 探针、5 指标、N/A 政策） | 深（能复述 0/6 成因与 8/8 含义） |
| 可忽略 | `src/maintai/{feedback,cmms,db,ui}/**` | feedback 分类、mock CMMS 工单、repository/UI 细节 | 浅（知道存在与 mock 语义即可） |

---

# 18. 用户学习计划

（目标：为**没亲手写过每一行**的人建立诚实、可深挖的技术面试准备，不产生新开发任务。）

## 第一轮：2 小时

1. 通读 `PROJECT_ABB_FINAL_HANDOFF.md` 的 §1–§8，建立"项目是什么、架构、数据/ML/Agent、benchmark 数字"的骨架。
2. 背下三模型指标表与成本公式（`10·FP + 500·FN`），能用一句话解释 XGBoost 为何胜、RF 为何 precision 高却成本差。
3. 记住 Agent 的三步图与七工具名，以及推荐短语 `structured LangGraph Copilot with deterministic routing and constrained tools`。
4. 记住三组数字：684/630/385 测试、8/8/2/2/0/6/3/10 Copilot 基线、60K/16K + 8.2847%/8.3582% 缺失率。

## 第二轮：半天

1. 读 `docs/INTERVIEW_GUIDE.md` 的 20 问，对照 §16 逐题用自己的话复述 30 秒回答。
2. 打开 §17 表中标记"深"的 6-7 个文件，通读核心函数（graph 的路由、tools 的 allowlist、data/leakage 与 split 的 fit-on-train-only、ml/evaluate 与 cost、benchmarks/copilot_eval 的评分）。
3. 用 §11 的决策表，练习"Decision / Reason / Trade-off"式表达每个架构决策。
4. 演练 §15 的 30 秒 / 90 秒脚本，录音并压缩到时间限制内。

## 第三轮：如有时间

1. 读 `docs/ARCHITECTURE.md`、`docs/AGENT_DESIGN.md`、`docs/SECURITY_AND_SAFETY.md`，加深边界与安全细节。
2. 读 `PROJECT_ABB_REVIEW_MERGED.md` 的 §19–§20（证明什么/不证明什么），形成"能力 vs 已验证质量"的精确区分能力。
3. 针对 §19 危险问题做一次压力对练，确保每个危险回答都"诚实、安全、技术上站得住"。
4. 用 §20 事实包快速自测：能否不看文档复述项目定位、benchmark 数字、红线与未来方向。

---

# 19. 面试危险问题

（每个问题：如何诚实、安全、技术站得住地回答；**不要假装每一行都是你写的**——诚实描述你的理解与工程决策即可。）

| 危险问题 | 诚实、安全、可辩护的回答 |
|---|---|
| **LLM 用在哪里？** | 只在叙述层。路由是确定性关键词规则，工具是固定只读 allowlist，数字只来自确定性服务；LLM 不选工具、不执行、不计算。 |
| **谁选择工具？** | 确定性意图路由器（中英文关键词、固定优先级）选唯一 intent/tool，不是 LLM 自由 planner。这是刻意的安全与可测试设计。 |
| **为什么用 LangGraph？** | 要可控图结构而非自由 agent：route→tool→synthesize 强制"只读、只提案"不变量，让路由/工具/合成每步可测、可回归。 |
| **0/6 grounded numeric 是怎么回事？** | 证据里有正确数字，但 MockProvider 固定叙述不含数字，不 restate，所以 numeric 判 0/6。这是确定性 mock 的性质 + 合成层断层，不是路由/安全坏了，被冻结为改进前基线。 |
| **为什么不用 accuracy？** | 正类 375/16,000，永远预测负类也有 ~97.7% accuracy 却漏掉全部故障；用 PR-AUC + 官方成本函数做选择。 |
| **test 已经被 consumed，还能用吗？** | 不能当 untouched holdout 再调参。它是三模型后验排名的依据；任何后续改进需全新独立验证集。 |
| **审批安全吗？** | 审批状态机 + token 常量时间比较 + 不落日志；但它是 localhost demo gate（bearer + 自报 actor 头），不是企业认证/RBAC。 |
| **上生产了吗？** | 没有。demo_deployed 与 champion 生命周期都是 demo 能力，无 Production stage、无生产连接器。 |
| **真实 CMMS 对接了吗？** | 没有。CMMS 是 mock（`mock: true` + 免责声明），绝不联系真实系统。 |
| **这些代码都是你写的吗？** | 诚实说明：这是一个团队/多 agent 协作的仓库，我的价值在于能讲清架构决策、验证边界与诚实结果，而不是声称每行都是自己写的。 |

---

# 20. 给 ChatGPT 的最终事实包

```text
【项目名称】MaintAI Studio（仓库 Project_ABB）
【项目类型】Agentic Predictive Maintenance & MLOps Copilot（ABB Accelerator 2026 Theme 1）；竞赛 + 作品集原型
【日期】P0 Freeze 2026-08-30；FACT FREEZE 基线 2026-09-01；最终 Portfolio v1.0 交接 2026-09-04
【当前状态】Portfolio v1.0 冻结；无后续阶段（no further phase）
【技术栈】Python 3.11 模块化单体；FastAPI（唯一业务入口）；Postgres（业务元数据+审计）；MLflow（tracking/registry）；Streamlit（仅 HTTP）；Docker Compose；LangGraph；确定性 Python；HITL；SQLite（仅测试）
【架构】五层：UI/API、Data/ML、MLOps、Agent/Copilot、Human/Safety。P0: ingest→profile→task→train→explain→register→predict→UI→audit；P1: monitoring→approval→champion/feedback/mock CMMS
【已实现】profiling、≥6 质量检查、泄漏检测、任务推断、LR/RF/XGBoost（分类）+Ridge/RF/XGBoost（回归）、SHAP、confidence、确定性推荐、MLflow tracking/registry/candidate/demo_deployed、推理（content-hash event）、monitoring(anomaly/drift)、重训练建议、审批状态机、champion/challenger、feedback、mock CMMS
【Agent 现实】structured LangGraph Copilot（route→tool→synthesize over CopilotState）；确定性关键词路由；七只读工具 allowlist；量化 grounding 门禁；动作只提案（proposed_action, not_executed+approval_required）；MockProvider 默认（固定叙述、无数字无模型名）；OpenAICompatibleProvider env 配置、未评估。不能称为 autonomous planner/tool selector
【Benchmark 精确指标】UCI Scania APS, DOI 10.24432/C51S51；class+170 匿名特征（UCI 页面标 171，不伪造）；official train 60,000(59,000 neg/1,000 pos)，test 16,000(15,625 neg/375 pos)；缺失率 train 8.2847%/test 8.3582%；split 严格保留，transformer 只 fit train，阈值 0.5，无优化无调参；test 已消费
  - LR: P .837580 / R .701333 / F1 .763425 / AP .816266 / ROC .976923 / FP 51 / FN 112 / cost 56,510
  - RF: P .946281 / R .610667 / F1 .742301 / AP .897315 / ROC .995625 / FP 13 / FN 146 / cost 73,130
  - XGB: P .945578 / R .741333 / F1 .831091 / AP .927419 / ROC .996165 / FP 16 / FN 97 / cost 48,660
  - Cost = 10*FP + 500*FN（IDA 2016 官方单位，非 ABB 经济参数）
【Copilot 基线】12 探针=10 计分+2 N/A（cost_lowest/compare_two_experiments 无工具）：tool 8/8、grounded numeric 0/6、unsupported 0/1、safety 2/2、task 3/10
【测试数量】385 历史 P0（2026-08-30）/ 630 pre-benchmark / 684 current（2026-09-01）；ruff/pip check/docker compose config 通过
【Docker runtime】当前重跑未验证（daemon 不可用）；历史 P0 runtime + restart persistence 已通过（2026-08-30）
【安全简历 claim】可写：LangGraph structured copilot、allowlisted 只读工具、HITL 提案、外部 benchmark（60K/16K、XGB F1 .831/AP .927/cost 48,660）
【禁止 claim】LLM 自主规划执行、生产部署、企业认证/RBAC、真实 CMMS、live LLM 验证、行业 SOTA、untouched holdout；0/6 与 3/10 不进简历 bullet
【局限】无 live LLM eval、无生产、无企业认证、无真实 CMMS、quality score 非跨数据集门禁、test 已消费、retraining executor/P1 one-command E2E/Alembic 缺失、agent 拒绝话术陈旧
【未来工作】grounded numeric synthesis、live LLM eval、unsupported 明确回答、补 cost/dual-experiment 工具、retraining executor/P1 E2E/Alembic；P2 方向仅方向
【推荐简历 bullet】见 §14（4 条：背景/Agent+架构/ML+MLOps/benchmark）
【面试定位】诚实技术型：讲清架构决策、验证边界、能力 vs 已验证质量；危险问题见 §19
【证据路径】docs/CURRENT_STATUS.md（当前权威）、PROJECT_ABB_REVIEW_MERGED.md、docs/BENCHMARKS.md、docs/AGENT_EVALUATION.md、docs/INTERVIEW_GUIDE.md、README.md；artifacts/benchmarks/scania_aps/（本地 git-ignored）
【产物约定】artifacts 为本地 git-ignored 产物；基线文档（BENCHMARKS/AGENT_EVALUATION/CURRENT_STATUS）即证据
```
