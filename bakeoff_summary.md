# Chop Regime Sub-Agent Bake-Off Summary

We conducted a structured "Bake-Off" comparing three alternative sub-agents to govern the **Chop Regime (Low-Volatility Mean Reversion)** in our Ethereum trading pipeline, replacing the GMM model. 

All models were optimized using a shortened Optuna tuner (50 trials) on the same training set, and tested on the out-of-sample period under identical fee/slippage parameters (10 bps base fee, 20 bps stop-loss/exit slippage).

---

## Performance Comparison Table

| Metric | Baseline GMM (Audited ATR) | Volatility Squeeze | XGBoost Classifier | StatArb / OU Pairs |
| :--- | :---: | :---: | :---: | :---: |
| **Git Branch** | `ATR` | `experiment/squeeze-agent` | `experiment/xgboost-agent` | `experiment/pairs-agent` |
| **Total Strategy Return (%)** | **+35.06%** | -30.86% | -3.49% | +4.59% |
| **Overall Sharpe Ratio** | **1.75** | -2.30 | 0.03 | 0.42 |
| **Overall Max Drawdown (%)** | **-12.91%** | -32.07% | -30.19% | -23.10% |
| **Chop Agent PnL Contribution (%)** | **-13.65%** | -34.13% | -28.43% | -23.24% |
| **Chop Agent Max Drawdown (%)** | **-22.37%** | -31.79% | -30.52% | -28.04% |
| **Breakout Agent PnL Contribution (%)**| +47.38% | -0.46% | +29.00% | +32.13% |
| **Breakout Agent Max Drawdown (%)** | -10.61% | -20.07% | -16.62% | -13.20% |
| **Trade Allocation (Long / Short / Flat)**| 299h / 803h / 3186h | 140h / 500h / 3648h | 0h / 1125h / 3163h | 468h / 831h / 2989h |
| **Optuna Best Sharpe (Shortened)** | N/A | -12.43 | -8.48 | -8.37 |
| **Verdict** | **RETAIN (Ensemble)** | **FAIL** | **FAIL** | **FAIL** |

---

## Sub-Agent Diagnostic Analysis

### 1. Volatility Squeeze Engine (`experiment/squeeze-agent`)
* **Logic**: Uses a 20-period Bollinger Band and Keltner Channel compression filter. Fires a long signal on momentum breakouts from the squeeze compression zone.
* **Analysis**: Suffered the worst performance. In the choppy regime, breakout signals from narrow ranges were almost entirely whipsaws, leading to "death by a thousand cuts." It had the worst Sharpe ratio (-2.30) and dragged down the Breakout sub-agent's performance as well.

### 2. XGBoost Classifier Engine (`experiment/xgboost-agent`)
* **Logic**: Trains a gradient-boosted classifier to predict whether next-hour returns will exceed a threshold, buying only on high-probability positive hours.
* **Analysis**: The model was highly over-constrained, generating **0 long signals** during the entire out-of-sample backtest. The chop agent contribution was negative (-28.43%) due to exit friction/transaction penalties when transitioning back to cash. The breakout sub-agent was left carrying the entire portfolio.

### 3. StatArb / Pairs Trading Engine (`experiment/pairs-agent`)
* **Logic**: Approximates a rolling cointegration spread against a mock BTC asset. Uses Ornstein-Uhlenbeck (OU) half-life estimation to ensure the spread is mean-reverting and trades rolling Z-score deviations.
* **Analysis**: While it produced a positive overall return (+4.59%) and Sharpe (0.42), this was entirely carried by the Breakout sub-agent (+32.13% contribution). The StatArb sub-agent itself bled -23.24% with a deep -28.04% max drawdown. High transaction costs and rapid spread shifts under high volatility led to frequent stop-outs.

---

## Conclusion & Production Recommendation

None of the three alternative sub-agents succeeded in generating positive standalone alpha in the Chop Regime. Mean-reverting range-trading in Ethereum remains a highly difficult task under realistic transaction fees (10 bps) and exit slippage (20 bps). 

### Recommendation:
1. **Short-Term**: Retain the baseline **GMM (Audited ATR)** setup on the `ATR` branch for production. It achieves a **1.75 Sharpe Ratio** and **35.06% overall return** out-of-sample.
2. **Medium-Term**: We should consider a **Single-Agent Breakout model**. Since all Chop Regime agents function as drag (GMM: -13.65%, StatArb: -23.24%, Squeeze: -34.13%), running the Breakout sub-agent alone (with flat position sizing in low-volatility regimes) represents our highest-Sharpe, lowest-risk configuration.
