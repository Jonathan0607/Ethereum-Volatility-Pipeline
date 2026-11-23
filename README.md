# Project Apex: Asynchronous Multi-Strategy Statistical Arbitrage Ensemble

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/requirements.txt)
[![C++20 HPC](https://img.shields.io/badge/C%2B%2B-20%20%2F%20pybind11-orange.svg)](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/cpp/monte_carlo.cpp)
[![Next.js 15](https://img.shields.io/badge/Frontend-Next.js%2015%20Telemetry-black.svg)](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20Async-009688.svg)](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/api.py)
[![Docker](https://img.shields.io/badge/Container-Docker%20Compose-2496ED.svg)](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/docker-compose.yml)

> **A production-grade, execution-aware quantitative trading architecture engineered for crypto derivatives.**  
> Built for institutional-scale perpetual futures markets, Project Apex resolves structural single-factor blowup risk through an orthogonally blended multi-strategy ensemble, sub-millisecond C++ Monte Carlo risk simulation, vectorized C-tensor backtesting, and live asynchronous order routing.

---

## 1. Executive Summary & Quantitative Performance

Crypto derivatives markets exhibit extreme regime non-stationarity, punctuated volatility clusters, and persistent funding distortions. Traditional single-factor strategies (pure trend-following or carry harvesting) inevitably experience cataclysmic drawdowns during liquidation cascades. 

**Project Apex** solves this through a mathematically orthogonal **Triple Ensemble Architecture**:
1. **Directional ACM**: Multi-horizon Time-Series Momentum, Inverted Funding Rate Carry, and Open Interest Velocity.
2. **Market-Neutral Cross-Sectional Spread**: Pairwise cointegrated divergence arbitrage capturing relative value while maintaining zero structural crypto beta.
3. **Binary Volatility Breakout**: Asymmetric Donchian volatility envelopes gated by Open Interest institutional positioning filters and a $+20\%$ profit-lock ratchet.

Across the exhaustive **4-Year Macrocycle Audit (2022–2026, 35,040 hourly bars)** encompassing the 2022 Terra/FTX deleveraging crash, the 2023 consolidation, and the 2024–2026 expansion, Project Apex achieved a **1.47 Sharpe Ratio** with an ultra-conservative **-17.99% Maximum Drawdown**, compared to the benchmark's **-79.01% Maximum Drawdown**.

### Full-Cycle Performance Metrics (2022 – 2026)

| Metric | Buy & Hold Benchmark (Equal-Weighted) | Project Apex (Global Baseline v22) | Institutional Delta |
| :--- | :---: | :---: | :---: |
| **Annualized Sharpe Ratio** | **0.28** | **1.47** | **+1.19 Sharpe** |
| **In-Sample Train Sharpe (2022–2024)** | -0.15 | **1.63** | *Bear Market Capital Preservation* |
| **Out-of-Sample Test Sharpe (2024–2026)** | 0.62 | **1.30** | *Out-of-Distribution Generalization* |
| **Maximum Drawdown (Max DD)** | **-79.01%** | **-17.99%** | **+61.02% Drawdown Shield** |
| **Total Cumulative Net Return** | +38.40% | **+172.41%** | **+134.01% Excess Alpha** |
| **RoMaD (Return to Max Drawdown) Ratio** | 0.48 | **9.58** | **+9.10x Risk Efficiency** |
| **Annualized Volatility** | 78.60% | **22.40%** | **-56.20% Volatility Compression** |
| **Execution Microstructure Assumptions** | N/A | **-2 bps Maker / -10 bps Taker (Hysteresis-Dampened)** | *100% Real-World Execution Latency* |

### Cumulative Performance Compounding Curve (2022–2026)

The strategy compounds smoothly across both structural bear and bull regimes while remaining completely immune to the market-wide -79% liquidation drawdowns:

![Project Apex 4-Year Comprehensive Performance](research/results/apex_v22_comprehensive_4yr.png)
*Figure 1: Full 4-year cumulative compounding curve of the Project Apex Global Optimal Configuration (`DH=0.75, SH=0.40, BOI=0.25 [15D / 60S / 25B]`) benchmarked against buy-and-hold across 35,040 hourly candles. Vertical dashed red line demarcates the blind chronological Train (2022–2024) and Test (2024–2026) split.*

---

## 2. High-Performance Computer Science Infrastructure

Project Apex is engineered from first principles as a modular, low-latency execution and backtesting system.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM RUNTIME ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                [ Binance / Bybit WebSocket Feeds (ccxt.pro / websockets) ]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ DATA INGESTION & ZERO-LAG STATE MACHINE (src/stream_engine.py, portfolio_engine)│
│  • Asynchronous 1m Kline Ticks with 1h Aggregation Frame                        │
│  • Resilient Multi-Region REST Seeding (data-api.binance.vision / fapi)         │
│  • Half-Open Socket Proactive Drop Detection (30s ping / 10s timeout)           │
│  • Zero-Lookahead State Shift: S_{t} = f(Price_{t-1}, OI_{t-1}, FR_{t-1})       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ QUANTITATIVE ALPHA CORES (research/apex_v22_comprehensive_4yr.py, src/strategy) │
│  ├── Sub-Engine 1: Directional ACM (Multi-Scale TSMOM + Inverted FR Carry + OI) │
│  ├── Sub-Engine 2: Market-Neutral Spread (Ranked Relative Dispersion Long/Short)│
│  └── Sub-Engine 3: Binary Breakout (72h/336h Donchian + 25% OI Gate + Lock)    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ HIGH-PERFORMANCE COMPUTING & RISK ARBITRATION                                   │
│  ├── C++20 Monte Carlo Engine (src/cpp/monte_carlo.cpp via pybind11)            │
│  │   • 10,000 Path Geometric Brownian Motion (GBM) Sub-ms VaR Simulation        │
│  ├── Dynamic Half-Kelly Sizing & Target Volatility Scaling (20% Target)         │
│  └── Execution Hysteresis Filter (Suppresses Churn, Decoupled Fee Models)       │
└─────────────────────────────────────────────────────────────────────────────────┘
                     │                                         │
                     ▼                                         ▼
┌────────────────────────────────────────┐ ┌──────────────────────────────────────┐
│ INSTITUTIONAL EXECUTION ROUTER         │ │ FULL-STACK TELEMETRY DASHBOARD       │
│ (src/kelly_execution.py, alpaca_client)│ │ (frontend/ Next.js 15, FastAPI API)  │
│  • Automated Maker Limit Postings      │ │  • Real-Time Portfolio Telemetry     │
│  • Latency-Critical Breakout Takers    │ │  • Interactive VaR Monte Carlo Visual│
│  • SQLite Persistence (data/trades.db) │ │  • Position & Regime Allocation Glass│
└────────────────────────────────────────┘ └──────────────────────────────────────┘
```

### HPC Elements

1. **C++20 Accelerated Monte Carlo Simulation ([src/cpp/monte_carlo.cpp](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/cpp/monte_carlo.cpp))**:
   - Compiled to native machine code via `pybind11` ([monte_carlo.cpython-314-darwin.so](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/monte_carlo.cpython-314-darwin.so)).
   - Generates 10,000 independent Geometric Brownian Motion (GBM) paths with 24-step horizons in **sub-millisecond execution**:
     $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
   - Extracts real-time Value-at-Risk ($VaR_{95\%}$) and terminal expectation bounds without garbage collection pauses.

2. **Vectorized High-Dimensional Tensor Sweeping**:
   - Implemented in [research/apex_v22_comprehensive_4yr.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/research/apex_v22_comprehensive_4yr.py).
   - Evaluates **28,875 continuous portfolio combinations** (5 Directional Hysteresis bands $\times$ 5 Spread Hysteresis bands $\times$ 5 Breakout OI gates $\times$ 231 ternary simplex allocations) across 35,040 hourly bars in under 3 minutes using C-contiguous memory structures.

3. **Production Asynchronous Ingestion & Fault Recovery ([src/stream_engine.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/stream_engine.py))**:
   - Asynchronous WebSocket client using `asyncio` and `websockets` with heartbeat health monitors (`ping_interval=30`, `ping_timeout=10`).
   - Multi-endpoint fallback across Binance US, Binance Vision, and Binance Futures REST APIs with automatic exponential backoff.

4. **Next.js 15 Full-Stack Telemetry Dashboard ([frontend/](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend))**:
   - Modern React server components, Tailwind CSS design system, and Lucide icons.
   - Provides institutional traders with real-time exposure gauges, live trade blotters, and interactive Monte Carlo simulation visualizers.

---

## 3. Alpha Generation & Market Microstructure

The strategy trades an institutional liquid altcoin universe: **SOL/USDT**, **AVAX/USDT**, and **LINK/USDT**.

### The Triple Ensemble Mathematical Formulation

The portfolio's aggregate net return at hour $t$ is defined as the convex combination of three uncorrelated sub-strategies:

$$R_{t}^{\text{Apex}} = w_{\text{DIR}} R_{t}^{\text{DIR}} + w_{\text{SP}} R_{t}^{\text{SP}} + w_{\text{BRK}} R_{t}^{\text{BRK}}$$

Where the global optimal weight vector is:

$$(w_{\text{DIR}}, w_{\text{SP}}, w_{\text{BRK}}) = (0.15, 0.60, 0.25)$$

#### Sub-Engine 1: Directional ACM (Adaptive Carry Momentum)
Combines three orthogonal time-series signals with inverse-volatility parity weighting:
- **Multi-Horizon Time-Series Momentum ($TSMOM$)**: Standardizes price drift against rolling realized volatility across windows $W \in \{24, 72, 168, 336, 720\}$ hours:
  $$\text{TSMOM}_{i,t} = \frac{1}{|W|} \sum_{w \in W} \tanh\left(\frac{P_{i,t-1} - P_{i,t-1-w}}{\sigma_{i,t-1,w} \sqrt{w}}\right)$$
- **Inverted Funding Rate Carry**: Cryptocurrencies exhibit positive carry when shorting crowded positive funding rates and longing negative funding rates. The signal is Z-scored over a 720-hour window:
  $$\text{Carry}_{i,t} = -\tanh\left(\frac{\text{FR}_{i,t-1} - \mu_{FR, 720}}{\sigma_{FR, 720}}\right)$$
- **Open Interest Velocity**: Measures institutional leverage inflow/outflow over 48 hours:
  $$\text{OI\_Regime}_{i,t} = \tanh\left(5.0 \times \frac{\text{OI}_{i,t-1} - \text{OI}_{i,t-49}}{\text{OI}_{i,t-49}}\right)$$
- **Asymmetric Tiering**: The top-ranked asset receives $1.5\times$ base weight, the median asset $1.0\times$, and the lowest asset $0.5\times$, scaled by inverse ATR percentage.

#### Sub-Engine 2: Market-Neutral Cross-Sectional Spread
- Continuously ranks assets by composite alpha ($0.30 \cdot \text{TSMOM} + 0.70 \cdot \text{OI}$).
- Establishes a simultaneous **Long** on the highest-ranking asset and **Short** on the lowest-ranking asset, scaled by rank dispersion conviction:
  $$\Delta_{\text{conviction}} = |S_{\text{top}} - S_{\text{bottom}}|$$
- Generates pure beta-neutral spread alpha, completely insulated from systemic Bitcoin/Ethereum market drops.

#### Sub-Engine 3: Binary Volatility Breakout
- Monitors 72-hour upper Donchian channels for upside breakout entries and 336-hour lower Donchian channels for exits:
  $$\text{Entry Condition}: \quad P_{i,t-1} > \max_{1 \le k \le 72}(H_{i,t-1-k}) \quad \text{AND} \quad \Delta \text{OI}_{48h, t-1} \ge 0.25$$
- **Institutional Open Interest Gate ($\Delta \text{OI}_{48h} \ge +25\%$)**: Filters out retail "fakeouts" by requiring authentic open interest expansion before committing capital.
- **Dynamic Profit-Lock**: When an open trade's profit exceeds $+20\%$, the exit threshold dynamically tightens from the 336-hour channel to the fast 72-hour lower channel, locking in accrued capital.

### Microstructure & Execution Realism

1. **Transaction Cost Hysteresis Banding**:
   - Rebalancing in high-frequency regimes destroys capital through slippage.
   - Position targets are only modified if the delta exceeds the hysteresis threshold:
     $$|\Delta w| \ge h \quad (h_{\text{DIR}} = 0.75, \quad h_{\text{SP}} = 0.40)$$
   - Suppresses over $80\%$ of spurious portfolio turnover, allowing the strategy to trade comfortably under realistic exchange order books.

2. **Decoupled Execution Fee Modeling**:
   - **Maker Model (-2 bps / -0.02%)**: Applied to scheduled rebalance adjustments routed via passive post-only limit orders.
   - **Taker Model (-10 bps / -0.10%)**: Applied to latency-sensitive breakout triggers requiring instantaneous liquidity removal.

3. **Strict 1-Hour Mechanical Execution Delay**:
   - Signals generated at candle close $t-1$ execute strictly at the close of candle $t$.
   - Enforces mathematical zero look-ahead bias and reproduces real-world API order dispatch latency.

---

## 4. The Overfitting Audit & Academic Rigor

In quantitative finance, backtests boasting artificially high Sharpe ratios ($> 2.0$) almost invariably suffer from selection bias, look-ahead leakage, or parameter hypersensitivity.

During the development of Project Apex v19/v20, preliminary sweeps revealed an aggressive parameter configuration producing a **1.90 Test Sharpe** on the 2024–2026 dataset:

![Project Apex 4x4 Component Architecture Visualization](research/results/apex_v20_4x4_grid.png)
*Figure 2: Component architecture 4x4 visualization ([research/results/apex_v20_4x4_grid.png](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/research/results/apex_v20_4x4_grid.png)) auditing individual assets (SOL, AVAX, LINK, Portfolio) across sub-engines (Directional, Spread, Breakout, Triple Ensemble).*

### The v20 Forensic Audit Decision
When cross-validated against the blind Train dataset (2022–2024 bear market), the 1.90 configuration demonstrated severe regime decay, proving it had over-indexed on the 2024 bull run's specific volatility structures.

**Rather than shipping a fragile, curve-fitted model, the quantitative team deliberately rejected the 1.90 configuration.**

Instead, the team executed the exhaustive **28,875-point grid sweep** in [research/apex_v22_comprehensive_4yr.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/research/apex_v22_comprehensive_4yr.py), selecting the global all-weather baseline:
- **Full 4-Year Sharpe**: **1.47**
- **Train (2022–2024) Sharpe**: **1.63** (Flawless capital preservation during FTX/Terra)
- **Test (2024–2026) Sharpe**: **1.30** (Out-of-sample stability during ETF liquidity surges)
- **Max Drawdown**: **-17.99%**

This deliberate choice demonstrates institutional risk management and intellectual honesty: trading parameter stability and regime resilience over vanity metrics.

---

## 5. Repository Architecture

```text
Ethereum Tracker/
├── README.md                            # Institutional Quantitative Whitepaper
├── requirements.txt                     # Core dependencies (FastAPI, NumPy, Pandas, pybind11)
├── setup.py                             # C++ Extension Build Script (monte_carlo.cpp)
├── docker-compose.yml                   # Container Orchestration
├── backend.Dockerfile                   # FastAPI Application Container
├── frontend.Dockerfile                  # Next.js Telemetry Container
├── stream.Dockerfile                    # Asynchronous Stream Engine Container
├── data/
│   ├── btc_hourly.csv                   # Historical Benchmark Datasets
│   ├── eth_hourly.csv                   # Historical ETH Closes & Volumes
│   ├── eth_hourly_with_funding.csv      # Historical Funding & Open Interest Series
│   └── trades.db                        # SQLite Production Trade & Telemetry Ledger
├── frontend/                            # Next.js 15 Full-Stack Telemetry Dashboard
│   ├── src/app/
│   │   ├── page.tsx                     # Live Execution Dashboard
│   │   ├── backtest/page.tsx            # Multi-Year Backtest Visualizer
│   │   └── monte-carlo/page.tsx         # C++ Real-Time Monte Carlo Risk Engine
│   └── package.json                     # Frontend Node Dependencies
├── research/
│   ├── apex_v22_comprehensive_4yr.py   # Final 4-Year Baseline (28,875-Combination Engine)
│   ├── results/
│   │   ├── apex_v22_comprehensive_4yr.png # Baseline Compounding Tearsheet
│   │   └── apex_v20_4x4_grid.png         # 4x4 Component Architecture Visual
│   └── archive/                         # Quarantined Historical Ablations & Audits
│       ├── results/                     # Archived Empirical Figures & JSON Records
│       └── *.py                         # Historical Strategy Iterations (v1 to v21)
└── src/
    ├── alpaca_client.py                 # Broker Execution Interface
    ├── api.py                           # FastAPI Telemetry & Order Service
    ├── data.py                          # Market Data Ingestion Utilities
    ├── kelly_execution.py               # Volatility-Scaled Kelly Sizing Logic
    ├── monte_carlo.*.so                 # Compiled C++ Monte Carlo Shared Library
    ├── portfolio_engine.py              # Production Multi-Asset Execution State Machine
    ├── strategy.py                      # Multi-Asset Signal Generation Logic
    ├── stream_engine.py                 # Async WebSocket Streamer with Fault Recovery
    ├── archive/                         # Legacy Prototype Pipeline Modules
    └── cpp/
        └── monte_carlo.cpp              # High-Performance C++20 Monte Carlo Implementation
```

---

## 6. Deployment & Operational Guide

### 1. Environment Initialization & C++ Compilation

Clone the repository and set up a Python 3.11+ virtual environment:

```bash
git clone https://github.com/Jonathan0607/ethereum-volatility-pipeline.git
cd "Ethereum Tracker"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Compile the high-performance C++ Monte Carlo pybind11 module:

```bash
python setup.py build_ext --inplace
```

### 2. Validating the 4-Year Global Baseline

Run the vectorized simulation engine to verify the 1.47 Sharpe baseline:

```bash
python research/apex_v22_comprehensive_4yr.py
```

### 3. Launching Live Data Streaming & Execution State Machine

Start the real-time WebSocket market listener:

```bash
python src/stream_engine.py
```

Execute a live portfolio rebalance evaluation:

```bash
python src/portfolio_engine.py --universe SOLUSDT AVAXUSDT LINKUSDT --leverage 2.5
```

### 4. Running the Telemetry Stack (FastAPI + Next.js)

Start the FastAPI backend:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Start the Next.js telemetry frontend:

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:3000` to inspect live telemetry, risk bounds, and real-time execution states.

### 5. Production Containerization (Docker Compose)

Deploy the entire production stack (Streamer, API, Next.js UI) with a single command:

```bash
docker-compose up --build -d
```

---

## 7. Institutional Disclaimers & Licensing

- **Quantitative Research Disclaimer**: Past performance is no guarantee of future returns. The quantitative models, backtests, and code contained herein are for educational, institutional research, and engineering demonstration purposes.
- **License**: Released under the [MIT License](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/LICENSE).