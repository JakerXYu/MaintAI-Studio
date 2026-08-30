# P1 Scope — MaintAI Studio

P1 is implemented **only after P0 Freeze** (see `AGENTS.md`). Do not create
empty P1 modules during P0. Full criteria are in the parent spec (§11, §15).

1. **Anomaly detection [implemented]** — robust Z-score/MAD baseline + IsolationForest; anomaly
   score/flag + top deviating features; no accuracy claim without labels.
2. **Drift detection [implemented]** — numeric PSI + KS (stat + p-value) + mean/std shift;
   categorical PSI/distribution distance; severity LOW/MEDIUM/HIGH with
   configurable thresholds (labeled "not universal industry standards").
3. **Synthetic production replay [implemented]** — normal / mild drift / severe drift /
   increased failure-risk batches, fixed seed.
4. **Cost-aware selection [core implemented]** — FN/FP cost configurable; `Expected Error Cost =
   FN×FN_cost + FP×FP_cost`; show metric-best vs cost-best (demo assumptions).
5. **Human-in-the-loop approval [state/API implemented]** — pending →
   approved/rejected/modified with audit; action executors remain pending.
6. **Champion/challenger [pending]** — candidate → challenger → champion → archived.
7. **Retraining recommendation [implemented]** — HIGH drift / performance drop / anomaly jump /
   schedule / manual; never auto-deploys.
8. **Technician feedback [pending]** — confirmed / false alarm / different issue / no action.
9. **Mock CMMS [pending]** — draft work-order (`POST /api/v1/cmms/work-orders/draft`),
   clearly labeled mock; no real Fiix/SSP endpoints or tokens.

Status: **in progress**. Current full regression gate: **512 tests passed**.
