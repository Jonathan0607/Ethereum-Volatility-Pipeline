# 🔬 Engineering & Experimentation Timeline

## 📜 Logging Protocol
> **MANDATORY PROTOCOL:** Every completed sprint, architectural refactor, pipeline milestone, or quantitative ablation experiment **MUST** be appended to this file before completing any task. Every entry must adhere to the standardized schema below, including hard quantitative metrics, exact commit hashes, and an actionable verdict.

---

## 🏛️ Version 0: Production Baseline (APEX v45 Champion)

- **Date:** 2026-11-22
- **Commit Hash:** `7badb16023927f43eaab74f2b3214661ed8bd146` (Branch: `main` / `feat/multimodal-risk-mle`)
- **Architecture:** 
  - Frozen 3-sleeve orthogonal quantitative derivatives engine:
    * **25% Directional Sleeve:** GARCH(1,1) dynamic volatility scaling, multi-horizon skewness ensemble (72h, 168h, 336h), and trend momentum.
    * **55% Spread Sleeve:** Market-neutral cross-sectional statistical arbitrage (SOL, AVAX, LINK).
    * **20% Funding Arbitrage Sleeve:** Perpetual funding rate carry harvesting and basis capture.
  - Historical Evaluation Period: 4 Years (2022–2026), 35,040+ hourly bars.
- **Audited Baseline Metrics (CPCV & DSR):**
  - **Mean Out-of-Sample Sharpe Ratio (CPCV):** `+1.47` (Full 4-Year Sharpe: `+1.46`)
  - **Maximum Drawdown (Max DD):** `-13.1%` (compressed from v44b `-16.3%`)
  - **Calmar Ratio:** `11.20`
  - **Sortino Ratio:** `+1.46`
  - **Deflated Sharpe Ratio (DSR):** `96.9%` ($p = 0.0314 < 0.05$, adjusting for 50 trials)
  - **CPCV Inter-Path Variance:** $\sigma = 0.003$
  - **Worst-Case OOS Path Return / Drawdown:** `+147.9%` net return / `-13.1%` Max DD
- **Status:** **FROZEN (Read-Only)** downstream evaluation harness and baseline benchmark for multimodal risk overlays.

---

## 📋 Experiment Entry Template

```markdown
### EXP-XXX: [Title / Sprint Name]
- **Date & Branch/Commit:** YYYY-MM-DD | `<branch>` @ `<commit-hash>`
- **Hypothesis / Objective:** [State the specific engineering bottleneck, causal leakage concern, or ML risk modeling capability targeted]
- **Implementation Summary:** [Key modules, files added/modified, algorithms implemented, training pipelines executed]
- **Metrics & Audit Results:**
  - *Data / Pipeline:* [Funnel drop rates, throughput, deduplication, latency (p50/p99)]
  - *Model Performance:* [PR-AUC, ROC-AUC, Brier Score, Expected Calibration Error (ECE), Log Loss]
  - *Downstream Portfolio Delta:* [Downstream Max DD reduction, Calmar delta, Sharpe delta, VaR / Expected Shortfall impact]
- **Verdict & Next Step:** `[KEEP / ITERATE / DISCARD]` - [Concise technical justification and next development action]
```

---

## 🧪 Experiment Registry

*(New experiment entries starting with `EXP-001` will be appended below)*
