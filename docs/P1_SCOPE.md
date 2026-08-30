# P1 Scope — MaintAI Studio

P1 is implemented **only after P0 Freeze** (see `AGENTS.md`). Do not create
empty P1 modules during P0. Full criteria are in the parent spec (§11, §15).

1. **Anomaly detection** — robust Z-score/MAD baseline + IsolationForest; anomaly
   score/flag + top deviating features; no accuracy claim without labels.
2. **Drift detection** — numeric PSI + KS (stat + p-value) + mean/std shift;
   categorical PSI/distribution distance; severity LOW/MEDIUM/HIGH with
   configurable thresholds (labeled "not universal industry standards").
3. **Synthetic production replay** — normal / mild drift / severe drift /
   increased failure-risk batches, fixed seed.
4. **Cost-aware selection** — FN/FP cost configurable; `Expected Error Cost =
   FN×FN_cost + FP×FP_cost`; show metric-best vs cost-best (demo assumptions).
5. **Human-in-the-loop approval** — pending → approved/rejected/modified with
   audit; gates model promotion, retraining deployment, CMMS action.
6. **Champion/challenger** — candidate → challenger → champion → archived.
7. **Retraining recommendation** — HIGH drift / performance drop / anomaly jump /
   schedule / manual; never auto-deploys.
8. **Technician feedback** — confirmed / false alarm / different issue / no action.
9. **Mock CMMS** — draft work-order (`POST /api/v1/cmms/work-orders/draft`),
   clearly labeled mock; no real Fiix/SSP endpoints or tokens.

Status: **authorized, not yet implemented** (P0 Freeze passed 2026-08-30).
