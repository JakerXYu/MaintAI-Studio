# MaintAI Studio 本地使用手册

本手册用于本地 Docker Compose 演示环境，覆盖启动、模拟数据、P0 数据与
模型流程、P1 监控与人工决策流程，以及结果正确性检查。

> MaintAI Studio 是演示和决策支持系统，不替代维护工程师。健康分、漂移
> 阈值、异常分数和成本参数均为可解释的演示假设，不是通用行业标准。

## 1. 启动与访问

先启动 Docker Desktop，然后在 PowerShell 或 CMD 中执行：

```powershell
cd C:\Users\Xiangyu\Documents\Playground\AgenticAI\Project_ABB
docker compose up --build
```

首次构建可能需要几分钟。看到 `postgres`、`mlflow`、`api`、`ui` 健康后，
打开：

| 服务 | 地址 | 用途 |
|---|---|---|
| Streamlit | http://127.0.0.1:8501 | 主要操作界面 |
| FastAPI | http://127.0.0.1:8000/docs | 查看和试调用 API |
| MLflow | http://127.0.0.1:5000 | 查看训练 run、指标和模型版本 |
| Ready | http://127.0.0.1:8000/health/ready | 检查数据库和 MLflow |

正确的 Ready 响应为：

```json
{"status":"ready","database":"ok","mlflow":"ok"}
```

停止前台服务按 `Ctrl+C`，然后执行：

```powershell
docker compose down
```

不要加 `-v`，否则会删除 Postgres、MLflow 和上传文件的持久化卷。

## 2. 模拟数据在哪里

默认演示数据：

```text
data/synthetic/maintai_ai4i_style_demo.csv
```

它由 `scripts/generate_demo_data.py` 使用固定随机种子 42 生成，结构参考
公开的 AI4I 2020 预测性维护数据，但所有行均为本项目合成，不含真实设备
或个人数据。

主要字段：

| 字段 | 含义 | 流程中的角色 |
|---|---|---|
| `timestamp` | 采样时间 | 时间字段 |
| `machine_id` | 合成设备编号 | 资产分组字段 |
| `serial_no` | 合成序列号 | ID/泄漏字段，应被排除 |
| `type` | L/M/H 产品类型 | 分类特征 |
| `air_temperature` | 环境温度 | 数值特征 |
| `process_temperature` | 工艺温度 | 数值特征 |
| `rotational_speed` | 转速 | 数值特征，含平线问题 |
| `torque` | 扭矩 | 数值特征，含异常值 |
| `tool_wear` | 磨损量 | 数值特征 |
| `machine_failure` | 是否故障 | 二分类目标 |

数据故意包含少量缺失值、一个重复行、设备 `M0001` 的转速平线，以及
设备 `M0002` 的扭矩异常值，用于验证质量检查不是静态页面。

在 PowerShell 中查看前几行：

```powershell
Get-Content .\data\synthetic\maintai_ai4i_style_demo.csv -TotalCount 6
```

重新生成并核对确定性：

```powershell
.\.venv\Scripts\python.exe scripts\generate_demo_data.py
git diff -- data\synthetic\maintai_ai4i_style_demo.csv
```

正常情况下 `git diff` 没有输出，说明固定种子生成了相同文件。

## 3. 页面操作顺序

建议严格按左侧编号操作。侧栏的 Dataset、Experiment、Model 是当前上下文；
如果选错了对象，可点击 `Reset context` 后重新选择。

### 3.1 Home

1. 查看 API、数据库、MLflow 健康状态。
2. 确认页面没有红色连接错误。
3. 查看流程总览和安全说明。

正确性证据：API 显示可用，Ready 页面返回三个 `ok/ready` 状态。

### 3.2 Dataset & Health

1. 在上传区域选择
   `data/synthetic/maintai_ai4i_style_demo.csv`。
2. 点击上传，记下返回的 Dataset ID。
3. 从数据集列表中选中刚上传的数据。
4. 点击 `Profile dataset`。
5. 查看 Schema、Health score 和 Findings 表格。

预期结果：

- 状态变为 `profiled`。
- 行列数与上传文件一致。
- Findings 中能看到缺失、重复、平线或异常值证据。
- 平线证据指向 `M0001` 的 `rotational_speed`。
- 异常值证据包含 `M0002` 的 `torque`。
- Health score 带有“启发式指标，不是行业标准”的说明。

重复上传同一个文件时，系统按 SHA-256 去重，不应产生一份内容相同但无法
追踪来源的新数据。

### 3.3 Task & Plan

选择当前数据集并填写：

```text
Target column:    machine_failure
Asset ID column:  machine_id
Timestamp column: timestamp
```

点击 `Recommend task`。

预期结果：

- Task type 为 `binary_classification`。
- 泄漏检查 verdict 为 `block`。
- `serial_no` 出现在 excluded features 中。
- 页面给出判断依据，而不是只有一个分类名称。

如果 `serial_no` 仍被用作训练特征，不要继续训练，这表示泄漏防护没有按预期
工作。

### 3.4 Experiments

1. 选择已经完成 Task Recommendation 的数据集。
2. 模型列表留空表示运行默认三个分类模型。
3. 保持 `minimum_recall=0.80`。
4. 创建实验。
5. 点击手动刷新，直到状态从 `pending/running` 变成 `succeeded`。
6. 查看 Comparison、Ranking 和 recommendation notes。

预期结果：

- Logistic Regression、Random Forest、XGBoost 都有 run。
- 实验最终为 `succeeded`，不是长期停在 `running`。
- 页面有推荐模型和排序依据。
- 最低召回率约束为 0.80；未满足时必须显示说明，不能静默忽略。
- MLflow 的 experiment/runs 与页面中的 run 数量和指标对应。

训练是单进程协调执行，不要连续多次点击创建实验。页面不会自动轮询，需要
手动刷新。

### 3.5 Explainability

1. 选择成功实验和其中一个模型 run。
2. 查看 Global feature importance。
3. 做过单条预测后，再查看该记录的 Local explanation。

预期结果：

- 页面显示解释方法，例如 SHAP 或 permutation fallback。
- 每个重要特征有方向或重要度证据。
- 页面明确说明“模型解释不等于已验证的物理根因”。

### 3.6 Registry & Deploy

1. 选择推荐 run。
2. 可填写一个 Registry name。
3. 点击注册，模型首先成为 `candidate`。
4. 点击 `Deploy demo`，状态变为 `demo_deployed`。

这是 P0 演示部署，不是生产发布。P1 的 `challenger/champion` 是独立的人类
审批生命周期，不会改变 P0 的预测资格语义。

### 3.7 Predict

只有 `demo_deployed` 模型可用于 P0 推理。单条 JSON 可使用以下高风险样本：

```json
{
  "type": "L",
  "air_temperature": 305.0,
  "process_temperature": 317.0,
  "rotational_speed": 1050.0,
  "torque": 92.0,
  "tool_wear": 245.0
}
```

预期结果：

- 返回故障类别或等价的高风险判断。
- 故障概率应不低于 0.5。
- 返回置信度和局部驱动因素。
- 审计只保存输入哈希，不保存原始预测输入。

Batch 页签可上传包含同样特征列的 CSV。缺字段、未知字段或类型错误应显示
明确错误，不能补出一个看似正常但无法解释的预测。

### 3.8 Copilot

可以依次使用页面中的快捷问题：

- 数据有哪些质量问题？
- 为什么推荐分类任务？
- 为什么推荐这个模型？
- 解释这条预测。

预期结果：

- 回答包含 evidence，而不是只有自然语言结论。
- 默认 mock LLM 可离线运行。
- 要求直接生产发布或创建工单时，Copilot 只提出建议，不绕过人工审批。

### 3.9 P1 Monitor & Act

P1 页面把后部署决策闭环集中在一个页面内，通常按以下顺序使用：

1. **Monitoring**：选择已注册模型，运行 normal、mild drift、severe drift
   或 increased failure-risk 回放；查看 PSI、KS、均值/标准差变化、异常分数
   和重训练建议。
2. **Cost comparison**：选择成功的二分类实验，填写 FN cost 和 FP cost；
   比较 metric-best 与 cost-best。修改成本后结果应重新计算，不会重新训练。
3. **Promotion & approvals**：把候选模型声明为 challenger，创建 promotion
   approval；审批后再显式执行 champion promotion。审批本身不执行发布。
4. **Feedback & mock CMMS**：对预测记录填写技术员反馈；CMMS 工单先提交
   approval，再生成明确标为 mock 的工单草稿。

审批决策和执行需要 `.env` 中配置的 `APPROVAL_API_TOKEN`。在 UI 中使用密码
输入框填写，不要把 token 放在备注、反馈、URL 或截图中。

P1 正确性证据：

- Severe drift 的结果严重度高于 normal，并附具体特征证据。
- Retraining recommendation 的 `auto_deploy` 永远为 false。
- 调高 FN cost 后，cost-best 允许与 metric-best 不同。
- Pending/rejected approval 无法执行 Champion 或 CMMS 动作。
- Approve 之后仍需单独点击执行；重复执行返回同一结果，不重复创建记录。
- Champion 生命周期不破坏原来的 `demo_deployed` 推理。
- Feedback 可追加查看，但不会触发在线学习。
- 工单始终显示 `mock=true` 或等价的 Mock CMMS 声明。

## 4. 一键结果验证

如果只想确认算法和闭环正确，而不操作 UI：

```powershell
.\.venv\Scripts\python.exe scripts\run_demo_pipeline.py
```

脚本使用临时 SQLite 和临时 MLflow，不读取 Compose 中的数据，也不会把结果
显示到当前 Streamlit。最后一行 JSON 应包含：

- 数据哈希和 Dataset ID；
- `binary_classification`；
- `serial_no` 被排除；
- 三个模型 run；
- 推荐模型；
- candidate 注册和 demo deploy；
- 高风险预测；
- 三个带 evidence 的 Copilot 回答。

本地代码回归检查：

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
```

## 5. 常见问题

### UI 打开但 API unavailable

检查：

```powershell
docker compose ps
docker compose logs api mlflow postgres
```

优先打开 `/health/ready` 判断是数据库还是 MLflow 不可用。

### 页面没有刚才的数据

先点击页面的手动 Refresh。仍没有时确认当前侧栏 Dataset/Experiment/Model ID，
必要时 Reset context 后重新选择。

### 实验一直 running

训练不会自动刷新。等待片刻后点击手动刷新，并查看：

```powershell
docker compose logs api
```

### 修改代码后数据库没有新表

当前演示使用 SQLAlchemy `create_all` 创建缺失表。重新构建并启动 migrate：

```powershell
docker compose up --build
```

它能创建新表，但不会迁移已有列；这是当前演示边界，不是生产迁移方案。

### 想完全清空演示数据

以下命令会永久删除 Compose 持久化卷，只在明确需要从零开始时使用：

```powershell
docker compose down -v
docker compose up --build
```
