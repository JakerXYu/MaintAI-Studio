# External Benchmark & Evaluation Story

本文件是给外部技术 reviewer 做深度访谈准备的答题底稿：围绕 Scania APS
external benchmark 与第一轮 Copilot evaluation 的当前结果，按「技术、诚实、
可深挖」的原则系统回答 20 个问题。所有数字均为 **冻结基线（BASELINE）**，
不允许为获得更好看的结果而重算、重训、调参或修改 Agent。

> 详细事实来源见文末「证据索引」。本文只做引用与解释，不产生新事实。

---

## 0. 冻结基线速查（Baseline Snapshot）

### 0.1 Scania APS 数据事实

| 项目 | 值 |
|---|---|
| 数据集 | APS Failure at Scania Trucks（UCI，DOI `10.24432/C51S51`） |
| 特征 | `class` + **170** 个匿名特征（UCI 页面标 171，实际 archive 为 class+170，不伪造第 171 列） |
| Official train | **60,000** 行（59,000 neg / 1,000 pos） |
| Official test | **16,000** 行（15,625 neg / 375 pos） |
| 缺失率 | 约 **8.3%**（train 8.2847% / test 8.3582%） |
| Split | 严格保留官方 holdout，无随机 split |
| 阈值 | `0.5`，**无 threshold optimization**，无 test 上调参 |
| 成本函数 | `Total Cost = 10 * FP + 500 * FN`（官方 IDA 2016 challenge 成本单位） |

### 0.2 三模型结果（official test，2026-09-01 实际运行）

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | FP | FN | Total Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.837580 | 0.701333 | 0.763425 | 0.816266 | 0.976923 | 51 | 112 | 56,510 |
| Random Forest | **0.946281** | 0.610667 | 0.742301 | 0.897315 | 0.995625 | **13** | **146** | 73,130 |
| XGBoost | 0.945578 | 0.741333 | **0.831091** | **0.927419** | 0.996165 | 16 | **97** | **48,660** |

PR-AUC 字段即 `average_precision_score`（Average Precision）。PR-AUC 后验排名与
challenge cost 后验排名都选择 **XGBoost**。

### 0.3 Copilot 基线（fixed-scenario deterministic mock regression）

| Metric | 值 |
|---|---|
| Provider | MockProvider（Live LLM: N/A） |
| 场景 | 12 个固定场景，其中 2 个 N/A（cost tool、双 experiment comparison），10 个计分 |
| Tool Selection Accuracy | **8/8** |
| Grounded Numeric Answer Accuracy | **0/6** |
| Unsupported Numeric Claim Rate | **0/1** |
| Unsafe Action Compliance | **2/2** |
| Task Success Rate | **3/10** |

### 0.4 测试 / 验证基线

| 项目 | 值 |
|---|---|
| 测试数量（benchmark 工作前） | 630 |
| 测试数量（当前） | **684/684 passed** |
| ruff / pip check / docker compose config | passed |
| Docker runtime 重跑 | **本轮未验证**（daemon 不可用） |
| 历史 P0 freeze 测试数 | **385**（独立历史快照，勿与 630/684 混写） |

---

## 1. 为什么选择 Scania APS？

Scania APS 是 UCI 上的公开、真实重型卡车运营数据（`APS Failure at Scania
Trucks`，DOI `10.24432/C51S51`），独立于本仓库默认的 AI4I-style 合成 demo。选择它有三层理由：

1. **真实性**：它是真实工业预测性维护二分类问题，用第三方的运营数据验证我们
   的 profiling → task framing → training → evaluation → recommendation 核心，
   避免「自己出题、自己验证」。
2. **成本可判定**：IDA 2016 challenge 官方给出 `Total Cost = 10*FP + 500*FN`，
   可以验证我们「成本感知模型选择」而非只看单一指标。
3. **极端不平衡**：正类仅约 2%（test 375/16,000），正好压测我们 PR-AUC 与
   代价驱动的选择逻辑。

**诚实边界**：这只是一个固定 catalog + 默认超参 + 阈值 0.5 的基线验证，不是与
IDA 参赛方案公平比较的 leaderboard；它验证「能力存在且可运行」，不宣称「业界最优」。

## 2. 为什么不用 Accuracy 当核心指标？

因为正类极端稀少（test 375/16,000，train 1,000/60,000），Accuracy 会被占 97%+
的负类主导：一个「永远预测负类」的模型也能拿到约 97.7% 的 accuracy，却漏掉
全部故障。因此：

- 主指标是 **PR-AUC**（实现为 `sklearn.average_precision_score`），适合稀有正类；
- 成本函数使 FN 的代价是 FP 的 50 倍，Recall/FN 远比 accuracy 重要；
- 项目在任何地方都不以 accuracy 作为选择依据。

## 3. 为什么 FN cost 是 FP 的 50 倍？

`Total Cost = 10 * FP + 500 * FN`，`500 / 10 = 50`。**这不是我们拍的参数**，而是
UCI/IDA 2016 challenge 页面明示的官方成本公式。其业务直觉是：

- **FN（漏报）**＝真实 APS 组件故障被判为健康 → 漏掉真实故障，造成非计划停机、
  更贵的维修甚至召回，代价极高；
- **FP（误报）**＝把健康车辆送去额外检查，代价相对低得多。

我们只是复用这个官方 challenge 成本单位来对固定 catalog 做后验排名，不把它当作
MaintAI 或 ABB 的真实维护经济参数。

## 4. 为什么 XGBoost cost 最低？

因为 cost 由 FN 主导，而 XGBoost 的 FN 最少（97），且 FP 也只有 16：

- XGBoost：`16×10 + 97×500 = 160 + 48,500 = 48,660`
- Logistic Regression：`51×10 + 112×500 = 510 + 56,000 = 56,510`
- Random Forest：`13×10 + 146×500 = 130 + 73,000 = 73,130`

同时 XGBoost 的 F1（0.831091）与 PR-AUC（0.927419）也都是三模型最高，所以它是
在**主指标与成本两个维度同时胜出**，不是「为了成本牺牲指标」。注意这是在阈值 0.5、
无任何阈值/超参优化的前提下成立。

## 5. 为什么 Random Forest precision 很高但 business cost 反而差？

RF 的 precision = 0.946281 是三模型最高、FP = 13 是三模型最少，看似「最安全」；
但它的 recall = 0.610667 是三模型最低、**FN = 146** 是三模型最多。因为 FN 的代价
是 FP 的 50 倍，146 个漏报产生 73,000 成本，远超 FP 少带来的节省，最终 cost 反而
最高（73,130）。

这直接证明：**在不平衡 + 非对称代价的工业场景里，只盯 precision/accuracy 会被误导**
——一个高精度、少误报的「保守」模型可能因漏掉大量真实故障而成为最贵的方案。
工业选型必须看成本函数，而不是单点指标。

## 6. preprocessing 如何避免 leakage？

- **官方 split 严格保留**：前 60,000 行只用于 fit，后 16,000 行只用于最终评估，
  没有随机 split，也没有把 test 混进任何 fit。
- **所有 transformer 只 fit train**：numeric median imputation；Logistic Regression
  的 numeric path 的 standard scaling，都只对 official train 拟合，再应用到 test。
- **缺失值先规范**：官方 `na` 先转成真正的 `NaN` 再进入预处理，imputer 只在 train 上拟合。
- **不做猜测性特征工程**：匿名列名原样保留，不猜测业务含义。

因此 test 只在最终评估时被消费，从未参与 fitting。

## 7. 为什么不在 test 上调 threshold？

- 本 benchmark 的目的就是测量固定 catalog 在官方 holdout 上「开箱即用」的表现，
  阈值固定为 0.5。
- 在 test 上调阈值等于拿 holdout 做模型选择，既消费了 test，也失去无偏泛化估计。
- 违反「诚实基线」原则：我们有意报告未经优化的 0.5 结果，并明确记录「无 threshold
  optimization」，不做任何美化。
- 现在去调阈值只会让 test 失效，并给人「cherry-pick 出好看数字」的空间，而不提供
  真实的泛化证据。

## 8. 为什么说 official test 已被 consumed？

三个 catalog 模型都在 official test 上报告结果，随后 `recommend()` /
`compare_costs()` 就基于这些 **test 结果做后验排名**，选出 XGBoost 为 winner。
也就是说，winner 的选择本身就用了 test set，因此该 test 已不再是「未被触碰的最终
holdout」。

所以 `recommended_model` 应理解为「本次 benchmark 的 winner」，而不是「仍待无偏
test 验收的 final model」。**结论：不能继续针对这个 test 调参，却仍声称它是
untouched final holdout**；任何后续调优都需要一个全新的、独立于本次的验证集。

## 9. 684 tests 证明什么？

- **确定性契约正确**：profiling、quality、leakage、task framing、split、train、
  eval、SHAP、registry、inference、monitoring、drift、approval、feedback、mock
  CMMS 等确定性服务按契约运行。
- **回归安全**：684/684 通过，且 ruff、pip check、docker compose config 全过，
  说明加入 benchmark track 与 P1 切片后代码库内部一致、无回归。
- **Agent 的确定性部分**：tool routing（8/8）与 safety gate（2/2）经确定性回归测试。
- 测试数量沿革：benchmark 前 630 → 当前 684；385 是更早的 P0 freeze 快照，属不同
  时期、不同范围，不能与 630/684 混写。

## 10. 684 tests 不证明什么？

- **不证明生产部署**，也不证明对真实工厂数据的泛化能力。
- **不证明 live LLM 质量**：Copilot 评估用的是 MockProvider（确定性），live LLM 为 N/A。
- **不证明真实使用中的 Agent 任务成功率**（当前 3/10）。
- **不证明通用预测性维护优势**：不是与 IDA 参赛方案的公平比较。
- **不证明企业级认证**（当前只是 demo token）。
- **不证明真实 CMMS 执行**（当前是 mock）。

一句话：684 tests 验证的是**确定性软件契约与回归**，不是**生产能力或泛化性能**。

## 11. 为什么 Agent Task Success 只有 3/10？

定义：fixed-scenario deterministic mock regression。12 个固定场景中 2 个 N/A
（cost tool、双 experiment comparison，当前架构不支持，不计成功也不计失败），
实际计分 10 个，端到端全过才计 success，结果 3/10。

主因：

1. **Grounded Numeric Answer Accuracy 0/6**：MockProvider 只返回固定的安全
   narrative，不会稳定输出题目要求的具体数字；严格版 evaluator 不允许「Evidence
   appendix 里有正确数值」替代「narrative 已回答量化问题」。
2. **`nonexistent_metric` 场景**：Copilot 没有编造 MSE 数字（这是安全正确），但
   也没有明确回答「MSE 不可用」，因此该场景判 FAIL。

所以 3/10 是「确定性路由 + mock provider 下 grounded synthesis」的能力测量，不是
live LLM 的准确率数字。

## 12. 0/6 grounded numeric 暴露了什么架构问题？

路由与取数是对的（tool selection 8/8，tool evidence 里有正确数值），但**合成层
（provider narrative）不能稳定回答题目要求的具体数字**。这暴露的是「tool evidence
有数据」与「最终答案真正回答了量化问题」之间的断层：

- 确定性层（路由、取数、反幻觉门禁）是坚实的；
- grounded numeric synthesis（让 provider 输出受 evidence 约束且能命中具体数字）
  是当前最弱环节；
- 反幻觉门禁阻止了编造数字，但 mock 的固定 narrative 满足不了量化问题，最终回退到
  原始 evidence 块，严格评分计为「未作答」。

这是一个**架构边界问题**，不是路由或安全的问题。

## 13. 为什么不隐藏这个差结果？

诚实与可审计是本项目的核心价值，任何 claim 都必须有 evidence。0/6 与 3/10 被冻结
为「改进前基线」：

- 隐藏它会（a）误述真实能力，（b）让未来改进失去可测量的参照；
- 这些差结果正是下一步该修什么的地图（grounded synthesis、unsupported handling）；
- 这是被诚实评估的竞赛原型，虚报数字在 deep-dive review 中必然被拆穿，毁掉可信度。

保留差结果比美化它更有工程价值。

## 14. LangGraph 在这里解决什么问题？

LangGraph 编排一条确定性的 `route → tool → synthesize` 状态图（`CopilotState`）：

- **route**：确定性意图路由器（中英文关键词、固定优先级），把请求映射到**唯一**的
  读意图或动作意图；
- **tool**：把意图分派到**唯一**的 allowlist 只读工具，参数固定；
- **synthesize**：从 tool evidence 构建最终答案；动作意图返回 proposal/refusal，
  绝不执行。

它把「规划、选工具、解释」做成可控图结构，并强制安全不变量：**agent 只读、只提案，
不写文件、不写 SQL、不改 registry/DB、不执行 mutation**。

## 15. Agent 与 deterministic ML services 的边界是什么？

- **确定性 Python（services）**负责：profiling、quality、task framing、split、
  train、eval、SHAP、registry、inference、monitoring、drift、approval、feedback、
  mock CMMS 等所有计算与写操作。
- **Agent（LangGraph + provider）**只负责：规划、路由、选择只读工具、解释确定性
  工具输出、提出动作建议。
- Agent **永远不能**：读文件、写 SQL、直接改 artifact/registry、执行 mutation。
- 写操作只经 FastAPI/UI 且写审计；当前 promotion 与 mock CMMS executor 必须
  经人工审批。未来若增加 retraining deployment，也必须先过审批；当前 executor
  不存在。

一句话不变量：**agent proposes, deterministic services execute, human approves
high-risk actions。**

## 16. Human Approval 是什么？

一个 demo 级的人工审批门禁：

- **状态机**：`pending → approved / rejected / modified`，用版本号 CAS 做乐观并发，
  审批与审计同事务、原子提交，payload 脱敏后入审计。
- **执行门禁**：mutation 端点要求 Bearer token + `X-Human-Actor-ID` 头（token 未设
  置时 503），token 常量时间比较、不写日志、不回显、不入审计。
- **审批与执行分离**：决定审批**不会**执行动作；执行是独立的 token 门控端点
  （当前为 champion 晋升与 mock CMMS），校验终态审批后幂等执行。重训练部署
  尚未实现，不能把 recommendation 或 approval action type 说成已有 executor。
- **全程审计**：每次 mutation 都写 audit event。

## 17. Human Approval 不是什么？

**不是企业级认证或 RBAC**：

- token 是 demo 环境变量（`approval_api_token`，demo 默认值），身份是自报的
  `X-Human-Actor-ID`；没有 SSO/OIDC、没有目录、没有角色、没有限流；
- 审批 list 端点在当前 demo 甚至未做认证（仅脱敏）；
- 它是「可信 localhost 环境下演示人机回路契约」的门禁，不是授权系统。

生产化需要前置 SSO/OIDC、真实 RBAC、token 轮换与限流。把 demo token 说成
enterprise auth 是错误表述。

## 18. demo_deployed 与 production deployed 有什么区别？

- **demo_deployed**：仅为 demo 服务，只把模型加载给 UI 演示路径；**绝不**设置
  MLflow `champion` alias，也**绝不**进入 Production stage。
- **production deployed（本项目未声称）**：意味着真实工厂集成、champion alias、
  promoted stage、活体连接器、带认证的推理。

registry 只在晋升路径外拒绝 `champion` alias；本 benchmark 的模型**没有**进入
model registry，也没有任何 production/champion 发布。二者语义必须严格区分。

## 19. 下一步最值得改的是哪里？

本轮只做 FACT FREEZE，不实现。可写为假设、留待验证的只有四类：

1. **Grounded numeric synthesis**：让 provider 稳定输出受 evidence 约束且命中具体
   数字的答案。
2. **Live LLM evaluation**：把当前 N/A 换成真实 provider 评估（必须新增节，不覆盖）。
3. **Unsupported claim handling**：遇到不存在指标时明确回答「不可用」，而不是静默失败。
4. **Tool-result interpretation**：补上 cost comparison 与双 experiment comparison
   工具（当前二者为 N/A）。

这些是「待验证的改进假设」，不是本轮要偷偷实现的功能。

## 20. 如果改进以后如何做 pre/post comparison？

- 当前基线**冻结、永不覆盖**。
- 任何未来改进新增独立的 **Post-Improvement Evaluation** 节，与
  **Baseline Copilot Evaluation** 并列（**append，绝不 overwrite**）。
- 每个 metric 记录：定义、numerator、denominator、scenario IDs、N/A 政策、
  provider（mock vs live）。
- 保持同一套固定场景集合，保证数字可比；若场景集合变化，必须显式记录变化本身。
- 每个数字都能回溯到 artifact / report / test / source，形成证据链。

---

## 21. 证据索引（Evidence / File Index）

### 文档

- `docs/benchmarks/SCANIA_APS_BENCHMARK.md` — Scania 数据/任务/split/preprocessing/
  模型/指标/成本/结果/Copilot 评估的权威详细记录。
- `docs/AGENT_DESIGN.md` — LangGraph graph、工具 allowlist、action proposal/refusal、
  grounding、read-only 边界。
- `docs/SECURITY_AND_SAFETY.md` — 上传边界、SQL 参数化、artifact trusted-dir、
  demo deploy、human-in-the-loop、not-production 声明。
- `docs/ARCHITECTURE.md` — 固定架构与组件边界。
- `docs/API_CONTRACT.md` — HTTP API 合同（审批/生命周期/反馈/CMMS 端点）。
- `P0_REVIEW.md` — P0 freeze 验收（含历史 385 tests 快照）。
- `docs/P1_SCOPE.md` — P1 九项范围（champion/feedback/mock CMMS/approval）。

### 产物（本地可复现文件，`artifacts/benchmarks/scania_aps/`）

- `dataset_summary.json` — 数据/split/缺失率事实。
- `model_metrics.json` / `model_metrics.csv` — 三模型指标。
- `confusion_matrices.json` — TN/FP/FN/TP。
- `cost_comparison.json` — 成本比较。
- `benchmark_summary.md` — 汇总。
- `copilot_evaluation.json` / `copilot_evaluation.md` — Copilot 基线评估原始结果。

### 复现命令（仓库根、Python 3.11 venv）

```powershell
.\.venv\Scripts\python.exe scripts\download_scania_aps.py
.\.venv\Scripts\python.exe scripts\prepare_scania_aps.py
.\.venv\Scripts\python.exe scripts\run_scania_aps_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_scania_copilot.py
```

### 验证命令

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
docker compose config --quiet
```

> 注意：本轮 Docker runtime 重跑**未验证**（daemon 不可用），相关结论不得写成
> 「已通过」。
