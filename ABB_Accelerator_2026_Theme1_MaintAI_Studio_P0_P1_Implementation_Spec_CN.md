# ABB Accelerator 2026 — Theme 1
# MaintAI Studio: Agentic Predictive Maintenance & MLOps Copilot
## P0 + P1 完整实现规范 / Kilo Code Agents 执行文档

**版本**：v1.0  
**日期**：2026-08-21  
**目标比赛**：ABB Accelerator 2026  
**选定主题**：Theme 1 — Agentic Predictive Maintenance Studio  
**项目代号**：`MaintAI Studio`  
**优先级**：P0 必须完整可演示；P1 在 P0 稳定基础上逐项增加。  
**本地可复用项目**：`Project_01_Industrial_Maintenance_Root_Cause_Agent_CN_EN`

---

## 0. 文档用途

本文件不是概念性 brainstorming，而是给 Kilo Code Agents 直接执行的工程实施规范。

所有 agents 的总原则：

1. **先审计、后复用、再新写。** 优先检查 `Project_01_Industrial_Maintenance_Root_Cause_Agent_CN_EN` 中已有模块；能复用的不要重写。
2. **不得修改或破坏原项目。** 复用时复制到新项目，保留来源说明。
3. **P0 先闭环。** 不允许因为追求复杂模型、漂亮 UI、额外 RAG、Kafka、Kubernetes 等导致核心链路无法演示。
4. **Agent 负责规划、调用和解释；确定性工具负责数据处理与模型执行。** 禁止让 LLM 随意生成 Python 后 `eval/exec`。
5. **工业可信性优先于模型炫技。** 数据质量、泄漏防护、类别不平衡、成本、可解释性、版本追踪、人工审批必须比“用了最先进模型”更重要。
6. **只使用公开数据或合成数据。** 不允许复制 SSP/Bosch/其他公司的真实数据、API token、内部文档、资产名称、报价、工单或任何 proprietary information。
7. **所有主要操作必须可复现、可审计。**
8. **README 中必须给出一条命令启动 Demo。**
9. **任何 feature 在合并前必须满足对应 Acceptance Criteria。**
10. **P0 未通过全部 smoke tests 前，不开发 P2。**

---

# 1. 官方题目约束与本项目解释

## 1.1 官方题目明确要求/建议的内容

根据 ABB Accelerator 2026 Theme 1 页面，挑战目标是：

> Build an AI-powered AutoML + MLOps Copilot for Industrial Equipment.

官方问题背景强调：
- 工业设施产生大量传感器数据；
- 从数据到 predictive maintenance insight 通常需要大量 ML expertise；
- 需要自动化 end-to-end ML workflow；
- 帮助工程师分析设备数据、提前识别潜在故障、推荐模型、解释预测，并形成 production-ready pipeline；
- 需要 intuitive、AI-powered experience。

官方给出的可包含功能：
- Automated dataset profiling and quality assessment
- Intelligent task and model selection
- Data preprocessing and feature engineering
- Model training and evaluation
- Explainable AI using feature importance and confidence scores
- Experiment tracking and model comparison
- One-click deployment of trained models
- Interactive prediction and inference dashboard

官方建议技术：
- AI / ML / AutoML / MLOps
- Python
- FastAPI
- MLflow
- LangGraph
- Docker
- SHAP
- LightGBM / XGBoost
- PostgreSQL

## 1.2 本项目的工程化扩展

以下内容不是网页逐字要求，而是为了增强工业可信性、比赛区分度和求职复用价值而增加：

- 工业数据质量规则；
- task auto-detection；
- time/group-aware split；
- target leakage 检测；
- maintenance cost-aware model selection；
- drift monitoring；
- human-in-the-loop approval；
- model registry / champion-challenger；
- retraining recommendation；
- mock CMMS work-order draft；
- audit log；
- agent tool safety；
- reproducible Docker stack；
- public/synthetic industrial demo dataset。

---

# 2. Product Vision

## 2.1 一句话定义

**MaintAI Studio 是一个面向工业工程师的 Agentic Predictive Maintenance + MLOps Copilot：从未知设备数据开始，自动完成数据健康检查、任务识别、建模计划、模型训练比较、可解释分析、模型注册与部署，并在 P1 中持续监控 drift、计算维护成本、建议 retraining，并经人工审批生成 mock CMMS maintenance action。**

## 2.2 核心价值

不是：

> Upload CSV → Train XGBoost → Show Accuracy.

而是：

> **Understand Data → Identify Maintenance Task → Build & Compare Models → Explain Risk → Deploy → Monitor → Human Review → Maintenance Action**

## 2.3 目标用户

Primary:
- Maintenance Engineer
- Reliability Engineer
- Manufacturing / Process Engineer
- Industrial Data Analyst

Secondary:
- Data Scientist
- OT / Digital Manufacturing Engineer
- Plant IT / Industrial Systems Engineer

## 2.4 产品设计原则

1. **Engineer-first, not Data-Scientist-first**
2. **Evidence before recommendation**
3. **Human approval before operational action**
4. **Deterministic tools underneath agentic orchestration**
5. **Model lifecycle, not one-time notebook**
6. **Industrial context, not generic AutoML**
7. **Graceful degradation**：没有 LLM key 时核心 ML workflow 仍可运行。
8. **Reproducibility**：同 dataset/config/seed 应得到可重复结果。
9. **Traceability**：每次 model run、decision、approval 可追踪。
10. **Demo reliability** 优先于 feature count。

---

# 3. Scope

## 3.1 P0 — 必须完整实现

P0 是比赛 Prototype 的最小完整闭环，必须全部工作：

1. CSV / Parquet dataset ingestion
2. Dataset schema inference
3. Automated data profiling
4. Industrial data quality assessment
5. Target / task selection
6. Intelligent task recommendation
7. Data preprocessing pipeline
8. Basic feature engineering
9. Baseline + tree-based model training
10. Model evaluation and comparison
11. Industrial-aware metric selection
12. SHAP / fallback explainability
13. Confidence score / prediction probability where applicable
14. Best-model recommendation with explanation
15. MLflow experiment tracking
16. Model artifact packaging
17. Model registration
18. FastAPI inference service
19. Batch/single prediction
20. Interactive Streamlit dashboard
21. LangGraph maintenance ML copilot
22. Docker Compose deployment
23. Audit log
24. Unit + integration + smoke tests
25. Public demo dataset
26. README and demo script

## 3.2 P1 — 在 P0 稳定后实现

1. Unsupervised anomaly detection
2. Drift detection
3. Production-vs-training monitoring
4. Maintenance cost-aware model selection
5. User-configurable FN/FP cost
6. Human approval workflow
7. Champion/challenger model lifecycle
8. Retraining recommendation
9. Retraining workflow trigger
10. Mock CMMS work-order draft
11. Technician feedback capture
12. Monitoring dashboard
13. Model promotion/rejection audit trail
14. Synthetic production stream / batch replay
15. P1 end-to-end demo scenario

## 3.3 明确不属于 P0/P1 的内容

除非 P0/P1 全部完成且有余力，否则不做：

- Kafka
- Kubernetes
- cloud deployment
- PLC real integration
- real Fiix connector
- real ABB systems connector
- deep time-series Transformer
- LSTM/TCN/Transformer model zoo
- vector database
- RAG over manuals
- OCR
- knowledge graph
- edge hardware deployment
- multi-agent swarm
- full React frontend
- mobile app
- fine-tuned LLM
- full AutoGluon/H2O integration
- real-time millisecond streaming claims

这些可列为 Future Work / P2。

---

# 4. 复用现有项目的强制流程

目标项目：

`Project_01_Industrial_Maintenance_Root_Cause_Agent_CN_EN`

## 4.1 第一个 Agent 必须先做 Repository Audit

新项目开始前，生成：

`docs/REUSE_MANIFEST.md`

至少包含：

| Existing Source | Capability | Reuse? | Destination | Required Modification | Risk |
|---|---|---|---|---|---|

必须搜索旧项目中是否已经存在：

- FastAPI app factory
- Pydantic schemas
- PostgreSQL connection
- SQLAlchemy models
- Docker / docker-compose
- LangGraph graph
- LLM provider abstraction
- tool calling
- audit logging
- Streamlit UI
- maintenance asset schemas
- sensor simulator
- CMMS mock
- work-order object
- approval state
- RAG components
- prompt templates
- test fixtures
- configuration loader
- logging
- CI scripts
- README setup
- `.env.example`

## 4.2 复用规则

### 可以直接复制
如果模块：
- 是通用基础设施；
- 没有 proprietary data；
- API contract 与本项目兼容；
- 复制比重写风险低。

### 可以复制后重构
如果：
- 逻辑有用但耦合 root-cause agent；
- schema 与新项目不同；
- prompt 需要改为 AutoML/MLOps；
- CMMS 逻辑可作为 P1 mock output。

### 不复用
如果：
- 包含公司真实数据；
- 包含 token/credential；
- 依赖旧项目强耦合；
- 测试缺失且重构成本高；
- 与 Theme 1 无关。

## 4.3 禁止行为

- 不得让 agent 为“保持整洁”把旧项目全部重写。
- 不得复制 `.env`、secret、API key。
- 不得复制 SSP 实际资产/工单/报价/BOM 数据。
- 不得从旧项目导入真实 Fiix base URL、tenant ID 或 credential。
- 新项目必须能独立运行。

---

# 5. 推荐技术栈

## 5.1 Core

- Python 3.11
- FastAPI
- Pydantic v2
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM（若安装稳定；否则 XGBoost 为主）
- SHAP
- MLflow
- PostgreSQL
- SQLAlchemy
- Alembic
- LangGraph
- Streamlit
- Plotly
- joblib
- Docker / Docker Compose
- pytest

## 5.2 Optional

- scipy
- evidently（仅当轻量整合，不得替代核心自有 drift interface）
- pydantic-settings
- httpx
- tenacity
- structlog
- uvicorn
- ruff
- mypy

## 5.3 LLM Provider

实现 provider abstraction：

```python
class LLMProvider(Protocol):
    def invoke(self, messages: list[Message], **kwargs) -> LLMResponse:
        ...
```

至少支持：

- `mock`：测试、离线演示
- `openai-compatible`：通过环境变量配置

不得把项目硬绑定某一家商业 API。

`.env.example`：

```env
LLM_PROVIDER=mock
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
DATABASE_URL=postgresql+psycopg://maintai:maintai@postgres:5432/maintai
MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

# 6. 推荐 Repository Structure

```text
maintai-studio/
├─ README.md
├─ LICENSE
├─ .env.example
├─ .gitignore
├─ pyproject.toml
├─ docker-compose.yml
├─ Makefile
├─ configs/
│  ├─ default.yaml
│  ├─ model_catalog.yaml
│  ├─ quality_rules.yaml
│  ├─ cost_defaults.yaml
│  └─ demo_ai4i.yaml
├─ data/
│  ├─ README.md
│  ├─ sample/
│  └─ synthetic/
├─ docs/
│  ├─ ARCHITECTURE.md
│  ├─ REUSE_MANIFEST.md
│  ├─ DATA_CONTRACT.md
│  ├─ MODELING.md
│  ├─ AGENT_DESIGN.md
│  ├─ DEMO_SCRIPT.md
│  ├─ TEST_PLAN.md
│  ├─ SECURITY_AND_SAFETY.md
│  └─ ABB_SUBMISSION_NOTES.md
├─ src/
│  └─ maintai/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ logging.py
│     ├─ db/
│     │  ├─ session.py
│     │  ├─ models.py
│     │  └─ repositories.py
│     ├─ data/
│     │  ├─ ingest.py
│     │  ├─ schema.py
│     │  ├─ profile.py
│     │  ├─ quality.py
│     │  ├─ leakage.py
│     │  ├─ split.py
│     │  └─ features.py
│     ├─ tasks/
│     │  ├─ infer.py
│     │  └─ schemas.py
│     ├─ ml/
│     │  ├─ catalog.py
│     │  ├─ preprocess.py
│     │  ├─ train.py
│     │  ├─ evaluate.py
│     │  ├─ recommend.py
│     │  ├─ explain.py
│     │  ├─ confidence.py
│     │  ├─ package.py
│     │  └─ registry.py
│     ├─ monitoring/
│     │  ├─ anomaly.py
│     │  ├─ drift.py
│     │  ├─ cost.py
│     │  └─ retraining.py
│     ├─ agent/
│     │  ├─ state.py
│     │  ├─ graph.py
│     │  ├─ tools.py
│     │  ├─ prompts.py
│     │  └─ provider.py
│     ├─ cmms/
│     │  ├─ schemas.py
│     │  ├─ mock_client.py
│     │  └─ recommendations.py
│     ├─ approvals/
│     │  ├─ service.py
│     │  └─ schemas.py
│     ├─ api/
│     │  ├─ main.py
│     │  ├─ deps.py
│     │  └─ routes/
│     │     ├─ datasets.py
│     │     ├─ experiments.py
│     │     ├─ models.py
│     │     ├─ predict.py
│     │     ├─ monitoring.py
│     │     ├─ approvals.py
│     │     ├─ cmms.py
│     │     └─ copilot.py
│     └─ ui/
│        └─ app.py
├─ scripts/
│  ├─ download_demo_data.py
│  ├─ generate_synthetic_data.py
│  ├─ seed_demo.py
│  ├─ run_demo_pipeline.py
│  └─ smoke_test.py
└─ tests/
   ├─ unit/
   ├─ integration/
   ├─ fixtures/
   └─ e2e/
```

---

# 7. 系统架构

```mermaid
flowchart LR
    U[Engineer / User] --> UI[Streamlit UI]
    UI --> API[FastAPI Backend]
    UI --> COP[LangGraph Copilot]

    COP --> T1[Data Profiling Tool]
    COP --> T2[Task Recommendation Tool]
    COP --> T3[Training Tool]
    COP --> T4[Evaluation Tool]
    COP --> T5[Explainability Tool]
    COP --> T6[Deployment Tool]
    COP --> T7[P1 Monitoring Tool]
    COP --> T8[P1 CMMS Draft Tool]

    API --> DATA[Data Pipeline]
    API --> ML[ML Pipeline]
    ML --> MLF[MLflow]
    API --> DB[(PostgreSQL)]
    API --> REG[Model Registry]
    REG --> INF[Inference Service]

    DATA --> DB
    ML --> DB
    COP --> DB

    P1[Production Replay / Monitoring] --> API
    API --> APPROVAL[Human Approval]
    APPROVAL --> CMMS[Mock CMMS]
```

---

# 8. 数据模型

## 8.1 Dataset

```text
dataset_id
name
source_type
file_path
file_hash
row_count
column_count
uploaded_at
schema_json
profile_json
quality_score
target_column
asset_id_column
timestamp_column
task_type
```

## 8.2 Experiment

```text
experiment_id
dataset_id
mlflow_run_id
task_type
model_name
config_hash
git_commit
status
primary_metric
primary_metric_value
created_at
```

## 8.3 RegisteredModel

```text
registered_model_id
experiment_id
name
version
artifact_uri
alias
approval_status
deployed
created_at
```

## 8.4 PredictionEvent

```text
prediction_id
model_version
asset_id
event_time
prediction
probability
input_hash
created_at
```

## 8.5 Approval

```text
approval_id
action_type
object_id
proposed_action_json
status: pending|approved|rejected|modified
reviewer
review_comment
created_at
resolved_at
```

## 8.6 AuditEvent

```text
event_id
actor_type: user|agent|system
actor_id
action
entity_type
entity_id
payload_json
timestamp
```

## 8.7 P1 MockCMMSWorkOrder

```text
work_order_id
asset_id
priority
recommended_action
reason
risk_score
evidence_json
source_model_version
approval_id
status: draft|approved
created_at
```

---

# 9. P0 功能规范

## 9.1 数据导入

支持：
- CSV
- Parquet

限制：
- 默认最大 200 MB，可配置
- UTF-8 优先
- 明确错误提示
- 自动计算 SHA-256 hash
- 不覆盖同名 dataset
- 对重复 hash 提示已存在

API：

```text
POST /api/v1/datasets/upload
GET  /api/v1/datasets
GET  /api/v1/datasets/{dataset_id}
```

Acceptance Criteria:
- AI4I CSV 可成功导入。
- 非法文件返回 4xx 和可理解错误。
- Dataset metadata 入库。
- hash 可复现。
- 测试覆盖 CSV 与 Parquet。

---

## 9.2 Schema Inference

自动推断：

- numeric
- categorical
- boolean
- datetime
- identifier-like
- possible asset ID
- possible timestamp
- possible target

输出：

```json
{
  "columns": [
    {
      "name": "Air temperature [K]",
      "dtype": "float",
      "semantic_type": "sensor_numeric",
      "missing_rate": 0.0
    }
  ],
  "candidate_targets": [],
  "candidate_asset_ids": [],
  "candidate_timestamps": []
}
```

不得仅依赖 LLM。Schema inference 必须是 deterministic Python。

---

## 9.3 Industrial Data Profiling

必须计算：

### 基础统计
- shape
- type distribution
- missingness
- duplicate rows
- unique count
- cardinality
- numeric min/max/mean/std/median/IQR
- categorical frequencies

### 工业数据特有检查
- constant sensor
- near-constant sensor
- flatline segment（存在 timestamp 时）
- spikes / robust outlier rate
- impossible range（如果 YAML 定义）
- sampling irregularity（存在 timestamp 时）
- timestamp gaps
- duplicate timestamps
- asset-level record imbalance
- target imbalance
- trainable sample count

输出 `Data Health Score`，但评分必须可解释。

建议：

```text
100
- missing penalty
- duplicate penalty
- flatline penalty
- severe outlier penalty
- imbalance warning penalty
- timestamp integrity penalty
```

不要伪装成行业标准；UI 标记：

> “MaintAI heuristic data health score”

---

## 9.4 Target Leakage Detection

必须至少检查：

- target exact duplicate
- feature highly identical to target
- post-event columns by suspicious names: failure/fault/repair/downtime_after/maintenance_result
- IDs with near-one-to-one mapping
- timestamp columns after known event time（若有）
- obvious label-derived columns

输出：
- `safe`
- `warning`
- `block`

任何 `block` feature 默认从训练中剔除，用户可手动 override，但必须记录 audit。

---

## 9.5 Task Recommendation

系统不要求 maintenance engineer 自己先理解 ML task。

P0 支持：

### Classification
适用：
- binary/multiclass failure label
- discrete target

### Regression
适用：
- continuous RUL / health score / time-to-failure target

P1 增加：

### Unsupervised Anomaly Detection
适用：
- 无 target
- sensor/condition data

Task recommendation 输出：

```json
{
  "recommended_task": "binary_classification",
  "confidence": 0.92,
  "evidence": [
    "target 'Machine failure' has two discrete classes",
    "positive class rate is 3.4%"
  ],
  "alternatives": []
}
```

LLM 只负责把 deterministic evidence 翻译成人类语言。Task 类型由规则引擎给出。

---

## 9.6 Data Split Strategy

这是工业可信性的关键。

优先级：

1. 如果有 asset ID + timestamp：time-aware/group-aware split
2. 有 asset ID，无 timestamp：GroupShuffleSplit / GroupKFold
3. 有 timestamp，无 asset：chronological split
4. 都没有：classification 用 stratified；regression 用 fixed-seed random split

严禁：
- 同一 asset 的未来记录泄漏到训练过去记录；
- scaler 在 full dataset fit；
- feature engineering 先于 split 造成 leakage。

所有 transformer 必须在 sklearn Pipeline 中 fit on train only。

---

## 9.7 Preprocessing

Numeric:
- median imputation
- optional scaling

Categorical:
- most-frequent imputation
- OneHotEncoder(handle_unknown="ignore")

Boolean:
- deterministic mapping

Identifier:
- 默认排除高 cardinality pure IDs
- asset ID 用于 split/context，不作为普通 feature，除非显式允许

Datetime:
- 不直接转整数作为默认特征
- 可生成 hour/day/elapsed 等受控 feature

所有 preprocessing 必须保存为 pipeline artifact，与模型一起部署。

---

## 9.8 Feature Engineering

P0 保持克制。

支持：
- ratio/interaction（仅 config 定义）
- rolling mean/std（有 timestamp + group 时）
- lag 1/3/5（有 timestamp + group 时）
- delta / first difference
- elapsed time
- simple domain-independent statistical features

禁止自动生成上千个特征。
所有 feature names 可追踪。

---

## 9.9 Model Catalog

### Classification P0
1. LogisticRegression
2. RandomForestClassifier
3. XGBoostClassifier
4. LightGBMClassifier（如果环境稳定）

### Regression P0
1. Ridge
2. RandomForestRegressor
3. XGBoostRegressor
4. LightGBMRegressor（可选）

规则：
- 小规模 demo 不做昂贵 hyperparameter sweep。
- 使用固定、可解释搜索空间。
- 默认 RandomizedSearchCV 或 small grid。
- runtime 上限可配置。
- 记录 seed。

---

## 9.10 Metrics

### Classification
必须显示：
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

默认 primary metric：
- positive rate < 10%：PR-AUC 优先
- 否则：F1 或 ROC-AUC

UI 必须解释为什么。

### Regression
必须显示：
- MAE
- RMSE
- R²

默认 primary：MAE 或 RMSE，可配置。

---

## 9.11 Confidence

Classification:
- `predict_proba`
- 可选 CalibratedClassifierCV
- calibration curve（有足够样本时）

Regression:
- P0 不伪造“概率”
- 可使用 CV residual quantile 形成 empirical interval
- 明确标为 empirical prediction interval，而不是 statistical guarantee

---

## 9.12 Model Recommendation

推荐引擎必须综合：
- primary metric
- recall constraint
- inference latency
- model complexity
- calibration
- optional cost（P1）

示例：

```json
{
  "recommended_model": "xgboost_v3",
  "reason": {
    "pr_auc": 0.81,
    "recall": 0.88,
    "f1": 0.72,
    "latency_ms": 4.8,
    "notes": [
      "highest PR-AUC",
      "meets minimum failure recall of 0.85"
    ]
  }
}
```

不得让 LLM 凭主观选模型。LLM 只解释推荐算法生成的结果。

---

## 9.13 Explainability

首选：
- SHAP TreeExplainer for tree models
- LinearExplainer for linear models

Fallback：
- permutation importance

必须支持：

### Global
- top feature importance
- summary plot

### Local
对一条预测：
- predicted risk
- top positive drivers
- top negative drivers

Copilot 可生成语言解释，但必须显示：

> “Model-based explanation, not a verified physical root cause.”

禁止把相关性解释成物理因果。

---

## 9.14 MLflow

每个 run 必须记录：

Parameters:
- dataset_id
- dataset_hash
- task_type
- target
- split_strategy
- feature config
- model name
- hyperparameters
- seed
- git commit
- config hash

Metrics:
- 全部 task metrics
- training time
- inference latency

Artifacts:
- model pipeline
- feature list
- confusion matrix / residual plot
- SHAP summary
- schema
- quality report
- evaluation JSON

Tags:
- `project=maintai-studio`
- `phase=P0|P1`
- `dataset_name`
- `model_family`

---

## 9.15 Model Registry

P0：
- 将推荐模型注册到 MLflow Registry
- 生成 project-level model record

至少 aliases：
- `candidate`
- `champion`

P0 可手动 promote。P1 增加审批与 challenger。

---

## 9.16 FastAPI Inference

Endpoints：

```text
GET  /health
GET  /api/v1/models
GET  /api/v1/models/{model_id}
POST /api/v1/predict
POST /api/v1/predict/batch
```

`POST /predict` 示例：

```json
{
  "model_id": "failure_model",
  "records": [
    {
      "air_temperature": 300.1,
      "process_temperature": 310.2,
      "rotational_speed": 1450,
      "torque": 42.5,
      "tool_wear": 120
    }
  ]
}
```

Response：

```json
{
  "model_version": "3",
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.82,
      "top_features": [
        {"feature": "torque", "impact": 0.21}
      ]
    }
  ]
}
```

必须：
- schema validation
- missing required field errors
- model version returned
- request audit
- predictable latency

---

## 9.17 Streamlit UI

页面：

### Page 1 — Home / Project
- What MaintAI does
- current dataset
- current model
- pipeline progress
- demo quick-start

### Page 2 — Dataset & Health
- upload
- schema
- quality score
- missingness
- outliers
- class imbalance
- leakage warnings
- sensor health warnings

### Page 3 — Task & Plan
- recommended task
- target
- split strategy
- selected features
- selected model catalog
- editable constraints

### Page 4 — Experiments
- training trigger
- model comparison table
- metrics
- runtime
- recommendation

### Page 5 — Explainability
- SHAP global
- single prediction explanation
- confidence / probability

### Page 6 — Registry & Deploy
- registered versions
- promote candidate/champion
- deployment status
- endpoint example
- health

### Page 7 — Predict
- manual single record
- CSV batch prediction
- results download

### Page 8 — Copilot
支持：
- “What problems do you see in this dataset?”
- “Why did you recommend classification?”
- “Which model should I deploy and why?”
- “Explain prediction #123.”
- “What should I fix before retraining?”

P1 增加：Monitoring / Approvals / CMMS Draft。

---

# 10. Agentic Copilot 设计

## 10.1 核心原则

LLM 不直接：
- 读任意本地文件；
- 写 arbitrary SQL；
- 执行 Python；
- 修改 model artifact；
- promote model；
- 创建 CMMS action。

LLM 只能调用受控 tools。

## 10.2 LangGraph State

```python
class CopilotState(TypedDict):
    conversation_id: str
    user_request: str
    dataset_id: str | None
    experiment_id: str | None
    model_id: str | None
    intent: str | None
    tool_results: list[dict]
    evidence: list[dict]
    proposed_action: dict | None
    final_answer: str | None
```

## 10.3 P0 Tools

- `get_dataset_profile(dataset_id)`
- `get_quality_report(dataset_id)`
- `get_task_recommendation(dataset_id)`
- `get_training_plan(dataset_id)`
- `start_training(dataset_id, config)`
- `get_experiment_results(experiment_id)`
- `get_model_comparison(dataset_id)`
- `get_model_explanation(model_id, record_id)`
- `register_model(experiment_id)`
- `get_deployment_status(model_id)`

## 10.4 P1 Tools

- `get_drift_report(model_id, batch_id)`
- `calculate_maintenance_cost(model_id, config)`
- `create_retraining_recommendation(model_id)`
- `submit_model_promotion_for_approval(model_id)`
- `draft_cmms_work_order(prediction_id)`
- `get_approval_status(approval_id)`

## 10.5 Graph

```mermaid
flowchart TD
    A[User Request] --> B[Intent Router]
    B --> C[Read-Only Analysis Tools]
    C --> D{Action Required?}
    D -- No --> E[Evidence Synthesis]
    D -- Yes --> F[Propose Action]
    F --> G{Requires Approval?}
    G -- Yes --> H[Create Pending Approval]
    G -- No --> I[Execute Allowed Tool]
    H --> E
    I --> E
    E --> J[Final Answer with Evidence]
```

## 10.6 Prompt Rules

System prompt 必须包含：
- You are an industrial predictive-maintenance ML copilot.
- Do not invent dataset statistics.
- Every quantitative claim must come from a tool result.
- Do not claim correlation proves root cause.
- Do not deploy/promote/create maintenance action without required approval.
- Prefer deterministic tool outputs over model intuition.
- Clearly distinguish prediction, explanation, and verified maintenance diagnosis.
- If evidence is insufficient, say so.
- Never expose secrets.

---

# 11. P1 功能规范

## 11.1 Anomaly Detection

至少实现：
1. Robust Z-score / MAD baseline
2. IsolationForest

输入：numeric condition/sensor features

输出：
- anomaly score
- anomaly flag
- top deviating features（heuristic）

如果存在 label，可以额外计算 precision/recall/PR-AUC。
没有 label 时禁止声称真实 detection accuracy。

---

## 11.2 Drift Detection

至少实现：

Numeric:
- PSI
- KS statistic + p-value
- mean/std shift

Categorical:
- PSI / distribution distance

Concept:
- training baseline vs production batch

Severity：

```text
LOW
MEDIUM
HIGH
```

阈值必须在 config 中，UI 写明：

> “Configurable monitoring thresholds; not universal industry standards.”

Drift report：

```json
{
  "batch_id": "prod_2026_09_20_001",
  "overall_status": "HIGH",
  "features": [
    {
      "feature": "torque",
      "psi": 0.31,
      "ks": 0.28,
      "severity": "HIGH"
    }
  ],
  "recommendation": "review/retraining recommended"
}
```

---

## 11.3 Synthetic Production Replay

`scripts/generate_synthetic_data.py`

生成：
- baseline batch
- mild drift batch
- severe drift batch
- optional increased failure-risk batch

要求可固定 seed。

UI 提供：
- Replay Normal
- Replay Mild Drift
- Replay Severe Drift

---

## 11.4 Cost-Aware Model Selection

用户可配置：

```yaml
false_negative_cost: 10000
false_positive_cost: 500
true_positive_maintenance_cost: 1200
```

最小实现：

```text
Expected Error Cost =
FN × false_negative_cost
+
FP × false_positive_cost
```

展示：
- metric-best model
- cost-best model
- difference

Copilot 示例：

> Model A has the highest F1, but Model B is recommended under the current maintenance-cost assumptions because it misses fewer failures.

必须允许用户修改成本并实时重算。

不要声称默认 dollar value 是 ABB/任何真实公司的成本。默认值明确写 `demo assumptions`。

---

## 11.5 Human-in-the-Loop Approval

以下操作必须 P1 审批：
- promote challenger → champion
- trigger retraining deployment
- generate final CMMS work-order draft status=approved

Flow：

```text
Agent proposes
→ Pending Approval
→ Engineer sees evidence
→ Approve / Reject / Modify
→ Action execution
→ Audit event
```

任何 agent 不允许绕过。

---

## 11.6 Champion / Challenger

状态：
- candidate
- challenger
- champion
- archived

规则：
- 新训练模型 → candidate
- 经评估达到门槛 → challenger
- 人工批准 → champion
- 原 champion → archived 或 previous_champion

比较：
- primary metric
- recall
- cost
- drift robustness
- latency

---

## 11.7 Retraining Recommendation

触发条件可包括：
- HIGH drift
- performance drop（有标签回流）
- anomaly rate jump
- scheduled interval
- manual request

Recommendation 不等于自动训练部署。

输出：

```json
{
  "should_retrain": true,
  "severity": "HIGH",
  "reasons": [
    "PSI > configured threshold on 3 critical features",
    "failure recall dropped from 0.88 to 0.74"
  ],
  "proposed_data_window": "...",
  "requires_approval": true
}
```

---

## 11.8 Technician Feedback

对预测/maintenance recommendation：

```text
Confirmed issue
False alarm
Different issue
No action needed
Comment
```

保存到 DB。

Future use：
- evaluation
- threshold tuning
- retraining label source

P1 不要求实现在线学习。

---

## 11.9 Mock CMMS Integration

必须是 Mock。

Endpoint：

```text
POST /api/v1/cmms/work-orders/draft
GET  /api/v1/cmms/work-orders
```

Schema：

```json
{
  "asset_id": "MOTOR_017",
  "priority": "HIGH",
  "recommended_action": "Inspect drive-end bearing",
  "reason": "Elevated model-based failure risk",
  "risk_score": 0.82,
  "evidence": [
    "vibration_rms",
    "bearing_temperature"
  ],
  "source_model_version": "3",
  "status": "draft"
}
```

UI 显示：

> “Mock CMMS integration for demonstration; no live plant system is connected.”

严禁真实 Fiix token、真实 SSP endpoints、真实 assets/work orders、真实生产数据。

---

# 12. Demo Dataset

## 12.1 P0 Primary Demo

优先：**UCI AI4I 2020 Predictive Maintenance Dataset**

理由：
- 工业设备语义明确；
- failure classification；
- 数据规模适中；
- 能快速确保 P0 end-to-end 稳定。

脚本不得默认把外部 URL 写死到 runtime。
提供：
- download script
- README source instruction
- small committed sample（若 license 允许）
- synthetic fallback

## 12.2 P1 Demo

基于 P0 dataset 创建 synthetic production batches：
- normal
- covariate drift
- sensor flatline
- increased anomaly
- shifted failure risk

## 12.3 Optional Secondary

NASA C-MAPSS 可作为：
- RUL regression
- degradation/time-series
- Future Work / extended demo

不允许因为 C-MAPSS 导致 P0 延期。

---

# 13. Demo Story

最终 Demo 必须讲一个连贯故事，而不是逐页面展示功能。

## Scenario

一家工厂希望预测 machine failure，但 maintenance team 没有 ML specialist。

1. **Upload**：Engineer 上传设备数据。
2. **Data Health**：MaintAI 发现 class imbalance / sensor issue / missing-flatline-outlier warning。
3. **Task**：自动识别为 failure-risk classification。
4. **Training**：比较 LR/RF/XGBoost。
5. **Recommendation**：不是看 accuracy，而是 PR-AUC / recall / maintenance cost。
6. **Explanation**：对高风险 asset 显示 risk、top drivers、confidence、非 root-cause disclaimer。
7. **Deploy**：注册 model，暴露 FastAPI endpoint。
8. **Monitor**：production batch 出现 drift。
9. **Retraining**：Copilot 建议 retraining，但要求 human approval。
10. **Maintenance Action**：高风险预测经 engineer approval，生成 mock CMMS draft work order。

Final message：

> MaintAI does not replace the maintenance engineer; it compresses the ML/MLOps workflow and keeps engineering decisions evidence-based and human-controlled.

---

# 14. Acceptance Criteria — P0

## Installation
- [ ] `docker compose up --build` 一条命令启动。
- [ ] UI、API、Postgres、MLflow 全部 healthy。
- [ ] `.env.example` 足够启动 mock-LLM mode。
- [ ] 无 hardcoded secrets。

## Data
- [ ] AI4I / sample dataset 可导入。
- [ ] Profiling report 自动生成。
- [ ] 至少 6 类 data-quality check 工作。
- [ ] leakage warning 有测试。
- [ ] task recommendation 可解释。

## ML
- [ ] classification path end-to-end。
- [ ] regression code path 有单元测试。
- [ ] 至少 3 个 classification models。
- [ ] metrics correct。
- [ ] imbalance 场景不以 accuracy 作为唯一指标。
- [ ] SHAP 或 fallback explanation 可生成。
- [ ] best-model recommendation deterministic。

## MLOps
- [ ] 每次训练进入 MLflow。
- [ ] parameters/metrics/artifacts 齐全。
- [ ] model 可注册。
- [ ] model artifact 包含 preprocessing pipeline。
- [ ] FastAPI 可加载注册模型。
- [ ] batch prediction 可工作。

## Agent
- [ ] LangGraph 能回答至少 5 个 canonical questions。
- [ ] quantitative answer 必须来自 tools。
- [ ] mock provider 下测试不需要网络。
- [ ] agent 不执行 arbitrary code。
- [ ] agent 不能直接修改 DB/model registry。

## UI
- [ ] Dataset Health
- [ ] Task Plan
- [ ] Experiments
- [ ] Explainability
- [ ] Registry/Deploy
- [ ] Predict
- [ ] Copilot

## QA
- [ ] `pytest` 通过。
- [ ] smoke test 通过。
- [ ] demo script 可重复。
- [ ] fixed random seed。
- [ ] README 可让陌生开发者在 15 分钟内启动。

---

# 15. Acceptance Criteria — P1

- [ ] IsolationForest anomaly detection。
- [ ] robust baseline anomaly detector。
- [ ] production replay。
- [ ] PSI/KS drift report。
- [ ] normal vs drift batch UI 清晰。
- [ ] cost-aware model comparison。
- [ ] FN/FP cost 可编辑。
- [ ] approval workflow。
- [ ] model promotion 需要 approval。
- [ ] champion/challenger 可追踪。
- [ ] retraining recommendation。
- [ ] retraining 不自动部署。
- [ ] technician feedback 保存。
- [ ] mock CMMS draft。
- [ ] CMMS action 需要 approval。
- [ ] 全部操作写 audit log。
- [ ] P1 e2e demo 可一键 seed/replay。

---

# 16. API Contract

建议 prefix：`/api/v1`

## Dataset

```text
POST /datasets/upload
GET  /datasets
GET  /datasets/{id}
POST /datasets/{id}/profile
GET  /datasets/{id}/quality
POST /datasets/{id}/task-recommendation
```

## Experiments

```text
POST /experiments
GET  /experiments
GET  /experiments/{id}
GET  /experiments/{id}/comparison
```

## Models

```text
GET  /models
GET  /models/{id}
POST /models/{id}/register
POST /models/{id}/promotion-request
```

## Prediction

```text
POST /predict
POST /predict/batch
```

## Explain

```text
POST /explain/local
GET  /explain/global/{model_id}
```

## Copilot

```text
POST /copilot/chat
```

## P1 Monitoring

```text
POST /monitoring/batches
GET  /monitoring/drift/{batch_id}
GET  /monitoring/anomalies/{batch_id}
POST /monitoring/retraining-recommendation
```

## Approval

```text
GET  /approvals
POST /approvals/{id}/approve
POST /approvals/{id}/reject
POST /approvals/{id}/modify
```

## CMMS

```text
POST /cmms/work-orders/draft
GET  /cmms/work-orders
```

---

# 17. Configuration

`configs/default.yaml` 至少：

```yaml
project:
  name: MaintAI Studio
  random_seed: 42

data:
  max_upload_mb: 200
  missing_warning_rate: 0.05
  missing_severe_rate: 0.20
  near_constant_unique_ratio: 0.001

split:
  test_size: 0.20
  validation_size: 0.20

classification:
  positive_class_threshold: 0.5
  minimum_recall: 0.80
  imbalance_pr_auc_threshold: 0.10

training:
  max_models: 4
  max_search_trials: 20

monitoring:
  psi_medium: 0.10
  psi_high: 0.25
  ks_alpha: 0.05

cost:
  false_negative_cost: 10000
  false_positive_cost: 500

approval:
  require_model_promotion: true
  require_cmms_action: true
```

文档必须声明这些是 demo defaults，不是 ABB 或任何工业企业标准。

---

# 18. Tests

## 18.1 Unit

必须覆盖：
- schema inference
- data profile
- missingness
- flatline
- outlier
- leakage
- task inference
- split logic
- preprocessing
- metrics
- model recommendation
- SHAP fallback
- cost calculation
- PSI
- approval state machine
- CMMS schema
- agent tool permissions

## 18.2 Integration

- upload → profile
- profile → task
- train → MLflow
- MLflow → registry
- registry → inference
- copilot → tool → grounded response
- P1 batch → drift → retraining recommendation
- approval → model promotion
- approval → CMMS draft

## 18.3 E2E

`scripts/run_demo_pipeline.py`

必须从干净数据库开始：
1. seed demo
2. profile
3. train
4. compare
5. register
6. deploy
7. predict
8. replay drift
9. approval
10. mock CMMS

返回 exit code 0。

---

# 19. Security / Safety / Industrial Reliability

至少实现：
- upload extension allowlist
- file size limit
- filename sanitization
- no arbitrary path access
- no `eval`
- no `exec`
- no arbitrary shell tool
- DB parameterized queries
- secrets only env vars
- model artifact trusted path only
- agent tool allowlist
- action approvals
- audit log
- user-facing disclaimers

必须显示：

> This prototype provides model-based decision support. It does not replace qualified maintenance, safety, or engineering judgment.

如果做 maintenance recommendation：

> Recommendation requires technician review before action.

---

# 20. Agent 并行开发分工

为了避免 Kilo agents 互相踩文件，按 ownership 分工。

## Agent A — Repo Auditor / Architect

Ownership:
- `docs/REUSE_MANIFEST.md`
- `docs/ARCHITECTURE.md`
- root configs
- repo skeleton

任务：
1. 审计旧项目。
2. 列出可复用组件。
3. 创建新 repo skeleton。
4. 配 Docker Compose。
5. 定 API/domain contracts。
6. 不实现复杂业务逻辑。

完成后先 commit。

## Agent B — Data Intelligence

Ownership:
- `src/maintai/data/**`
- `src/maintai/tasks/**`
- related tests

任务：ingestion / schema / profile / quality / leakage / split / features / task inference。

## Agent C — ML / Explainability

Ownership:
- `src/maintai/ml/**`
- ML tests

任务：model catalog / preprocessing / training / evaluation / recommendation / SHAP / confidence / packaging。

## Agent D — MLOps / Backend

Ownership:
- `src/maintai/db/**`
- `src/maintai/api/**`
- MLflow integration
- Alembic

任务：DB / REST API / experiment tracking / registry / inference / audit。

## Agent E — Agentic Copilot

Ownership:
- `src/maintai/agent/**`
- agent tests

优先检查旧 Root Cause Agent 的 LangGraph、provider abstraction、tools、prompt、evidence、approval logic。

## Agent F — UI

Ownership:
- `src/maintai/ui/**`

任务：Streamlit pages / API client / plots / data health / experiments / explainability / deploy / copilot / P1 monitoring/approval。

原则：UI 不直接调用 ML internals；UI 通过 API/service。

## Agent G — P1 Monitoring / Operations

Ownership:
- `src/maintai/monitoring/**`
- `src/maintai/approvals/**`
- `src/maintai/cmms/**`

任务：anomaly / drift / cost / retraining / approval / CMMS mock / synthetic replay。

只在 P0 integration test 通过后开始。

## Agent H — QA / Demo / Submission

Ownership:
- `tests/e2e/**`
- `scripts/**`
- `docs/DEMO_SCRIPT.md`
- `docs/TEST_PLAN.md`
- `README.md`

任务：reproduce setup / smoke tests / demo seed / demo script / screenshots checklist / architecture / failure-mode testing / clean README。

---

# 21. Git / Integration Rules

建议：

```text
main
├─ feat/repo-foundation
├─ feat/data-intelligence
├─ feat/ml-pipeline
├─ feat/backend-mlflow
├─ feat/copilot
├─ feat/ui
└─ feat/p1-monitoring
```

每个 PR/merge 前：

```bash
ruff check .
pytest -q
python scripts/smoke_test.py
```

如使用 typing：

```bash
mypy src/maintai
```

Commit 示例：

```text
feat(data): add industrial data quality profiler
feat(ml): add cost-aware model recommendation
feat(agent): add grounded training-plan tools
fix(api): validate model input schema
test(e2e): add full P0 demo pipeline
```

---

# 22. 实施顺序

## Phase A — Foundation
目标：repo 可启动；Postgres + API + MLflow + UI health；reuse manifest 完成。

**Stop condition**：`docker compose up` 成功。

## Phase B — Data Intelligence
目标：upload → profile → quality → task。

**Stop condition**：AI4I data health page 可完整展示。

## Phase C — ML Pipeline
目标：train → compare → explain。

**Stop condition**：至少 3 models + MLflow + SHAP。

## Phase D — Deploy
目标：register → FastAPI predict → UI inference。

**Stop condition**：P0 deterministic workflow 完整。

## Phase E — Copilot
目标：LangGraph 调用已有 tools；grounded answer。

**Stop condition**：canonical demo questions 全通过。

## Phase F — P0 Freeze
必须：tests / demo / docs / UI polish / no new features。

只有 Freeze 完成后进入 P1。

## Phase G — P1 Monitoring
- anomaly
- drift
- replay

## Phase H — P1 Decision Loop
- cost
- approval
- champion/challenger
- retraining
- CMMS mock

## Phase I — Submission Polish
- 3–5 min demo
- screenshots
- architecture
- value proposition
- README
- future roadmap

---

# 23. Canonical Copilot Questions

P0 测试必须包括：

1. “What data-quality problems do you see?”
2. “What maintenance ML task should I use and why?”
3. “Which model performed best?”
4. “Why are you recommending this model instead of the one with the highest accuracy?”
5. “Explain why this machine received a high failure-risk prediction.”
6. “What should I fix before deploying this model?”
7. “Is the model currently deployed?”

P1：

8. “Has the production data drifted?”
9. “Should we retrain?”
10. “Which model is cheaper under my failure-cost assumptions?”
11. “Promote the challenger.” → 必须创建 approval，不直接 promote。
12. “Create a maintenance work order for this high-risk machine.” → 必须创建 draft/proposal + approval。

---

# 24. Demo Failure Cases

系统必须提前测试：

- CSV 无 target
- target 全部一个 class
- 90% missing column
- pure ID column
- duplicated target feature
- timestamp unordered
- tiny dataset
- unknown categories at inference
- missing prediction field
- corrupted model artifact
- MLflow unavailable
- LLM unavailable
- Postgres unavailable
- drift batch with extra column
- model promotion without approval
- CMMS action without approval

目标：graceful error / no crash loop / clear message。

---

# 25. README 必须包含

1. Project overview
2. ABB Theme mapping
3. Architecture
4. Feature matrix P0/P1
5. Quick start
6. Demo data
7. Demo walkthrough
8. API examples
9. Agent safety model
10. MLOps lifecycle
11. Screenshots
12. Testing
13. Known limitations
14. Future work
15. Data/privacy statement

Quick Start 理想：

```bash
git clone ...
cd maintai-studio
cp .env.example .env
docker compose up --build
```

然后：

```text
UI: http://localhost:8501
API: http://localhost:8000/docs
MLflow: http://localhost:5000
```

---

# 26. 比赛 Differentiators

最终 pitch 不要只说“Agentic + AutoML”。

1. **Industrial Data Health Before ML** — 先解决传感器/数据质量，再建模。
2. **Task Intelligence** — 工程师不需要先懂 classification/regression/anomaly。
3. **Cost-Aware, Not Accuracy-Only** — 失败漏报与误报成本不同。
4. **Explainable + Confidence-Aware** — 结果附证据，不是 black-box score。
5. **Agentic Orchestration with Deterministic Tools** — LLM 不执行不受控代码。
6. **Closed-Loop MLOps** — deploy 之后仍 monitor、detect drift、recommend retraining。
7. **Human-in-the-Loop** — 模型升级和 maintenance action 保留工程师控制。
8. **Maintenance Workflow Bridge** — 最终可形成 mock CMMS action，而不是停留在 dashboard。

---

# 27. Pitch 核心信息

## Problem

Predictive maintenance often fails before the model stage: industrial data is messy, task framing is unclear, model evaluation is disconnected from maintenance cost, and successful notebook models rarely become monitored production workflows.

## Solution

MaintAI Studio converts an equipment dataset into an auditable predictive-maintenance workflow:

> profile → frame → train → explain → deploy → monitor → review → act

## Why Agentic

Agentic AI 负责：理解用户意图、编排 tools、解释 evidence、建议下一步。

但 deterministic services 执行 ML，human 审批高风险 action。

## Why It Matters

The system lowers the ML expertise barrier without removing engineering control.

---

# 28. Future Work / P2

只在文档中列，不影响 P0/P1：

- C-MAPSS RUL
- deep time-series models
- streaming via Kafka
- edge ONNX deployment
- OPC UA live connector
- real CMMS connector abstraction
- maintenance manuals/RAG
- multi-modal diagnostics
- digital twin context
- federated/edge learning
- multi-site model governance
- cloud/Kubernetes
- automated threshold tuning
- causal diagnosis

---

# 29. Definition of Done

项目不以“代码写完”作为 Done。

Done =

1. 全新环境可启动；
2. sample dataset 可完整跑通；
3. UI 可现场 demo；
4. MLflow 记录完整；
5. model 可部署和预测；
6. Copilot 的数字结论有 tool evidence；
7. drift/cost/approval/CMMS P1 闭环可演示；
8. 无 proprietary data；
9. tests 通过；
10. README 可复现；
11. Demo 不需要临时手工修代码；
12. 所有高风险 action 经过 human approval；
13. 已知 limitations 清楚写出。

---

# 30. Kilo Code Agents 第一轮统一 Prompt

## Mission

Build `MaintAI Studio`, an ABB Accelerator 2026 Theme 1 prototype: an agentic predictive-maintenance AutoML + MLOps copilot for industrial equipment.

You must implement P0 first and freeze it before P1. The system must be reproducible, auditable, industrially credible, and demo-safe.

## Existing Code Reuse

Before writing new infrastructure, inspect the local project:

`Project_01_Industrial_Maintenance_Root_Cause_Agent_CN_EN`

Create `docs/REUSE_MANIFEST.md`.

Reuse or adapt useful generic components such as:
- FastAPI foundation
- LangGraph orchestration
- provider abstraction
- PostgreSQL models
- Docker setup
- Streamlit UI
- audit logging
- approval workflow
- mock CMMS schemas
- synthetic sensor data
- test fixtures

Do not modify the source project.

Do not copy any proprietary data, credentials, real CMMS endpoints, company documents, real work orders, parts/BOM data, quotes, or internal identifiers.

## Architecture Rule

LLM/agent:
- plans
- selects tools
- explains tool outputs
- proposes actions

Deterministic Python services:
- profile data
- detect quality issues
- infer task
- split/preprocess
- train models
- evaluate
- calculate SHAP
- register/deploy models
- detect drift
- calculate costs

Human approval:
- model promotion
- retraining deployment
- CMMS maintenance action

Never use `eval`, `exec`, arbitrary shell execution, or arbitrary agent-generated SQL.

## P0 Required

Implement:
- CSV/Parquet upload
- profiling
- industrial data-quality checks
- leakage detection
- task recommendation
- classification/regression preprocessing
- LR/RF/XGBoost model comparison
- metrics
- SHAP
- model recommendation
- MLflow
- registry
- FastAPI inference
- Streamlit
- LangGraph Copilot
- Docker Compose
- audit log
- tests
- demo data
- demo script

P0 is not complete until the full workflow works from a clean environment.

## P1 Required

After P0 freeze:
- anomaly detection
- PSI/KS drift
- production replay
- cost-aware model comparison
- human approval
- champion/challenger
- retraining recommendation
- technician feedback
- mock CMMS work-order draft
- audit trail

## Final Quality Bar

A judge must be able to understand within minutes:
1. what problem the system solves;
2. why it is more than generic AutoML;
3. why the agent is trustworthy;
4. how the model is selected;
5. how predictions are explained;
6. how deployment is tracked;
7. how drift is detected;
8. why human approval remains in the loop.

Prioritize a stable, coherent, defensible prototype over feature count.
