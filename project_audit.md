# Codebase Topography & Audit Report

This document presents a comprehensive audit of the files, classes, methods, and functions composing the Ethereum Volatility Strategy Pipeline.

---

## 1. Core Services & Strategies (`src/`)

### 📂 [alpaca_client.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/alpaca_client.py)
- **High-Level Purpose**: Execution client interface to target order routing to the Alpaca Broker REST API, accommodating fractional coin sizes, market orders, and emergency liquidations (CASH).
- **Classes**:
  - `AlpacaExecutionClient`: Manages live API connections and order transmission.
    - `__init__(self)`: Configures the Alpaca REST client with credentials (`APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, `APCA_API_BASE_URL`) and registers the target asset symbol (`ETHUSD`).
    - `execute_trade(self, action: str, target_weight: float, current_price: float)`: Translates strategy decisions into Alpaca REST API orders:
      - If action is `"CASH"`, liquidates all active ETH positions.
      - Retrieves current account equity to compute target fractional quantities based on `target_weight` and `current_price` (rounded to 4 decimals).
      - If action is `"SELL_SHORT"`, closes any long positions and submits a market sell order.

---

### 📂 [api.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/api.py)
- **High-Level Purpose**: Main web server hosting FastAPI REST endpoints and WebSocket ingestion handlers, orchestrating decision-making between strategies and broker connections.
- **Classes**:
  - `TradingState`: Tracks system state variables.
    - `__init__(self)`: Instantiates a default state where `cooldown_until` is set to `datetime.min`.
  - `LiveExecutionPayload` (Pydantic model): Defines schema for incoming WebSocket ticks.
    - Fields: `current_price`, `forecasted_vol`, `prob_high_vol`, `rolling_max`, `rolling_min`, `gmm_z_score`, `gmm_cluster`, `vol_24h`, `vol_168h`, `closes`.
- **Functions / Standalone Methods**:
  - `get_db_path()`: Returns the absolute file path resolving to the SQLite database.
  - `initialize_database()`: Creates the `paper_trades` table if not exists with all required column definitions. Applies schema migrations (via ALTER TABLE) if fields are missing.
  - `log_trade(action, price, pred_vol, regime, lower_bound, upper_bound, position_size, gmm_z_score, gmm_cluster)`: Writes trade logs to SQLite, calculates realized PNL for closed positions, and posts structured embed notifications containing telemetry data to a Discord webhook.
  - `load_artifacts()` (FastAPI startup handler): Runs database initialization and displays server configurations.
  - `execute_live_stream_trade(payload: LiveExecutionPayload)`: Master real-time gateway:
    - Verifies system cooldown constraints.
    - Re-fits a GARCH model on the latest Yahoo Finance data to generate a rolling volatility base.
    - Loads hyperparameters from `best_params.txt`.
    - Evaluates hierarchical routing: Breakout (Short-Only) vs GMM (Long-Only) state conditions.
    - Computes volatility-scaled position sizing and checks it against a friction/rebalance threshold.
    - Invokes the C++ Monte Carlo simulator for price boundaries.
    - Dispatches execution requests to the `AlpacaExecutionClient`.
  - `health_check()`: Returns system engine state.
  - `get_latest_state()`: Reads database records, returning the 10 most recent execution ticks.
  - `get_backtest_data()`: Reads backtest results JSON payload.
  - `get_portfolio_stats()`: Processes SQLite trades to yield total realized/unrealized PNL metrics, win rates, and closed trade counts.
  - `get_monte_carlo_visual()`: Simulates 50 GBM asset price projection paths over 24 steps with baseline volatility (0.05), returning JSON results for charting.

---

### 📂 [backtest.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/backtest.py)
- **High-Level Purpose**: Offline performance simulator executing historical splits, running indicators, applying fees/slippage penalties, and plotting performance diagnostics.
- **Functions / Standalone Methods**:
  - `get_hurst_exponent(ts, max_lag=20)`: Calculates the Hurst exponent using log returns to check if price is mean-reverting (Hurst < 0.5) or trending.
  - `load_data()`: Reads hourly dataset `data/eth_hourly.csv`.
  - `get_lstm_predictions(df)`: Performs forward pass of sequential historical feature arrays through the ProgressiveModel to output 24-hour volatility predictions.
  - `run_backtest(df, target_volatility)`: Main loop executing historical simulations. Evaluates HMM volatility probability, calculates rolling indicators, determines sub-agent activation zones, scales positions according to predicted volatility, implements the friction/rebalance filter, computes transaction fees (with penalties on exits to simulate slippage), and returns a cumulative strategy trajectory.
  - `calculate_metrics(df, verbose)`: Computes annualized return, Sharpe ratios, maximum drawdowns, and sub-agent specific attribution statistics.
  - `export_json(df, metrics)`: Downsamples strategy series and outputs JSON data to `backtest_results.json`.
  - `plot_results(df)`: Plots strategy return against raw Buy & Hold performance, shading regions corresponding to active trading agents.
  - `plot_dashboard(df)`: Generates detailed subplot visualization showing actual volatility vs AI forecast volatility, risk thresholds, and HMM state classification.
  - `run_visualizer()`: Standalone method running features calculations and saving dashboard outputs.

---

### 📂 [data.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/data.py)
- **High-Level Purpose**: Data processing utility that fetches yahoo finance candles, enforces schema cleanliness, and constructs stationary features.
- **Global Variables**:
  - `FEATURE_COLS`: `['volatility', 'log_return', 'rolling_vol_24h', 'ma_20_dist', 'ma_50_dist']`
- **Functions / Standalone Methods**:
  - `split_data(df, verbose)`: Splits data chronologically: 75% for training (history) and 25% for testing (future).
  - `check_integrity(df)`: Cleans data by filling missing NaN cells using forward fill, sorting index dates, and checking for essential price and volume columns.
  - `fetch_data()`: Ingests 2 years of hourly ETH price history via yfinance, deduplicates it against existing datasets, saves it to `data/eth_hourly.csv`, and runs data splits.
  - `run_processing()`: Confirms files exist and checks integrity.
  - `_compute_stationary_features(df)`: Utility computing log returns, rolling volatility, and distances to moving averages.
  - `calculate_features(df, train_df)`: Fits a GARCH model on training returns, extracts conditional volatilities, constructs input features, and appends the target column `fwd_vol_24h`.
  - `calculate_features_test(df)`: Similar to `calculate_features` but executes without fitting a new GARCH model (uses pre-fit coefficients) and skips target column generation.

---

### 📂 [hmm_engine.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/hmm_engine.py)
- **High-Level Purpose**: Fits Gaussian Hidden Markov Models to classify volatile market states.
- **Functions / Standalone Methods**:
  - `get_hmm_features(closes)`: Computes log returns and rolling volatility on a sequence of close prices.
  - `get_high_vol_probability(closes)`: Fits a 2-State Gaussian Hidden Markov Model, identifies the component corresponding to the high volatility regime (based on the component's volatility mean), and computes the posterior probability of the current tick being classified under this high-vol state.

---

### 📂 [pipeline.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/pipeline.py)
- **High-Level Purpose**: End-to-end retraining script running fetches, optimization runs, neural network fits, and backtesting.
- **Functions / Standalone Methods**:
  - `run_pipeline()`: Triggers sequence: `data.fetch_data()`, loads parameters, executes `strategy.train_model()`, runs out-of-sample backtest, generates and exports charts, tables, and JSON metadata.

---

### 📂 [strategy.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/strategy.py)
- **High-Level Purpose**: Neural network architecture definitions, neural training loops, and Gaussian Mixture model classification.
- **Classes**:
  - `ProgressiveModel` (inherits from `nn.Module`): Recurrent neural network mapping input sequences to future volatility targets.
    - `__init__(self, input_dim, hidden_dim, output_dim, dropout)`: Defines RNN layer, GRU layer, LSTM layer, intermediate dropout regularization, and linear projection weights.
    - `forward(self, x)`: Feeds inputs sequentially through the RNN, GRU, and LSTM, extracting the final sequence timestep to predict volatility.
- **Functions / Standalone Methods**:
  - `get_gmm_state(closes, z_window=20)`: Calculates rolling z-score of close prices, fits a 3-component Gaussian Mixture Model, orders component means, and classifies the current market state (0: Oversold, 1: Neutral, 2: Overbought).
  - `create_sequences(features, targets, seq_length)`: Creates sequential feature sliding windows of length `seq_length` matched to their corresponding targets.
  - `load_best_params()`: Reads the optimal configuration dictionary from `best_params.txt`.
  - `train_model(hyperparams)`: Loads historical CSV, applies stationary scaling, creates PyTorch datasets, instantiates the `ProgressiveModel`, runs standard backpropagation, and serializes the model weights along with scaling objects.

---

### 📂 [stream_engine.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/stream_engine.py)
- **High-Level Purpose**: Real-time market tick listener that aggregates live minute data, performs model inference on the hour mark, and hands off outputs to the API engine.
- **Functions / Standalone Methods**:
  - `get_hurst_exponent(ts, max_lag=20)`: Utility function calculating hurst coefficient.
  - `stream_ethereum_data()` (coroutine): Establishes a persistent socket connection to the Binance US feed (`wss://stream.binance.us:9443/ws/ethusdt@kline_1m`). Processes closed 1-minute candles and updates an hourly price buffer. At the turn of the hour, calculates rolling features, invokes the PyTorch `ProgressiveModel` for a volatility forecast, queries the GMM and HMM states, and dispatches the payload to the API microservice (`/execution/live-stream`).

---

### 📂 [cpp/monte_carlo.cpp](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/src/cpp/monte_carlo.cpp)
- **High-Level Purpose**: C++ source compiling to a Python module to compute fast Geometric Brownian Motion simulations.
- **Functions / Standalone Methods**:
  - `simulate_gbm(current_price, predicted_vol, num_sims, steps)`: Runs paths utilizing the normal distribution, sorts terminal prices, and extracts the 95% confidence bounds.
  - `PYBIND11_MODULE(monte_carlo, m)`: Binds methods to Python.

---

## 2. Research & Tuning sandboxes (`research/`)

### 📂 [optuna_tuner.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/research/optuna_tuner.py)
- **High-Level Purpose**: Runs Bayesian hyperparameter optimizations across strategy rules (entry levels, lookbacks, rebalance rules) to maximize Sharpe Ratio under trading constraints.
- **Functions / Standalone Methods**:
  - `load_eth_data()`: Ingests raw historical records.
  - `get_lstm_predictions(df, best_params)`: Generates predictions out of ProgressiveModel.
  - `load_hmm_features(df)`: Manages feature calculation caching.
  - `simulate_sharpe(test_df, params)`: Fast vectorized strategy simulation that applies trading rules, scales size, applies transaction fees, and handles penalties for drawdowns.
  - `objective(trial, test_df)`: Optuna objective wrapper that suggests parameter values and evaluates strategy results.
  - `write_best_params(study, current_params)`: Writes optimized strategy params back to `best_params.txt`.
  - `main()`: Orchestrates the optimization runs.

---

### 📂 [optimize_strategy.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/research/optimize_strategy.py)
- **High-Level Purpose**: Alternative hyperparameter optimizer focused on Hurst filters and moving average trend constraints.
- **Functions / Standalone Methods**:
  - `simulate_strategy_sharpe(df, predictions, params)`: Evaluates strategy returns under MA filter regimes.
  - `objective(trial, df, predictions)`: Handles trials.
  - `main()`: Resolves data structures and runs study loops.

---

## 3. Web Front-End (`frontend/`)

### 📂 [src/app/page.tsx](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend/src/app/page.tsx)
- **High-Level Purpose**: Renders the main dashboard cockpit containing running stats (PnL, Win Rate) and live trade execution feeds.
- **Functions / Components**:
  - `formatTime(ts)`, `formatPnl(val)`, `pnlGradient(val)`, `pnlColorPlain(val)`, `actionBadge(action)`, `statusLED(status)`: Render formatting utilities.
  - `LiveOps()`: Primary client dashboard page. Polls backend telemetry endpoints every 5 seconds.
  - `GlassCard(...)`: UI container display block for metric values.

---

### 📂 [src/app/backtest/page.tsx](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend/src/app/backtest/page.tsx)
- **High-Level Purpose**: Pulls and plots out-of-sample backtest JSON files on a responsive Recharts graphic.
- **Functions / Components**:
  - `BacktestTooltip({ active, payload })`: Custom charts hover tooltips.
  - `BacktestPage()`: Renders historical returns comparison between strategy performance and Buy & Hold ETH.
  - `GlassCard(...)`: Metric card component.

---

### 📂 [src/app/monte-carlo/page.tsx](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend/src/app/monte-carlo/page.tsx)
- **High-Level Purpose**: Renders forward-looking Geometric Brownian Motion simulation paths.
- **Functions / Components**:
  - `MonteCarloTooltip(...)`: Custom simulation tooltips.
  - `MonteCarloPage()`: Renders line charts illustrating 50 simulated prices paths and the median outcome.

---

### 📂 [src/lib/utils.ts](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend/src/lib/utils.ts)
- **High-Level Purpose**: Tailwind CSS class name merging helper.
- **Functions / Components**:
  - `cn(...inputs)`: Combines `clsx` and `twMerge` utility functions.

---

### 📂 [src/components/ui/card.tsx](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend/src/components/ui/card.tsx)
- **High-Level Purpose**: Card rendering primitive.
- **Functions / Components**:
  - `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardAction`, `CardContent`, `CardFooter`: Sub-components styling UI blocks.

---

## 4. Root Configuration & Deployments

- **[setup.py](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/setup.py)**: Extension building configuration for the C++ Monte Carlo simulation.
- **[docker-compose.yml](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/docker-compose.yml)**: Defines application microservices (`api`, `web`, `stream_engine`).
- **[backend.Dockerfile](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/backend.Dockerfile)**: Ingests requirements, builds pybind11 extension modules, and starts FastAPI.
- **[frontend.Dockerfile](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/frontend.Dockerfile)**: NextJS node:20-alpine build environment.
- **[stream.Dockerfile](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/stream.Dockerfile)**: Stream client execution environment.
- **[retrain.sh](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/retrain.sh)**: Executable shell script automated by cron to retrain models weekly and deploy to a production VPS.
- **[requirements.txt](file:///Users/kingebenezer/Desktop/Coding/Projects/Ethereum%20Tracker/requirements.txt)**: Contains Python packages.
