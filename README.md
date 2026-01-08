# Ethereum Volatility Prediction Pipeline

## Overview
This project is an end-to-end quantitative research infrastructure designed to forecast short-term volatility in Ethereum (ETH/USD) markets. Unlike traditional price prediction models, this system focuses on **volatility clustering** and **regime detection** to assess market risk.

## Architecture
The system follows a standard ETL + Inference pattern:
1.  **Data Ingestion:** High-frequency OHLCV tick data fetched via CCXT (Kraken API).
2.  **Feature Engineering:** - Log-Returns transformation for stationarity.
    - Rolling Volatility (GARCH proxies).
    - Augmented Dickey-Fuller (ADF) tests for stationarity checks.
3.  **Regime Detection:** Gaussian Mixture Models (GMM) to classify market states (e.g., *Stable* vs. *Turbulent*).
4.  **Modeling:** LSTM (Long Short-Term Memory) networks for sequence forecasting.

## Results (Preliminary)
Below is the output of the **RegimeGuard** module. It uses unsupervised learning (GMM) to automatically tag market regimes based on volatility clustering.

![Regime Detection](regime_plot.png)

*Red dots indicate high-volatility regimes detected by the GMM, while green dots indicate stable market conditions.*

## Tech Stack
-   **Orchestration:** Python Scripts (Airflow planned)
-   **Data Processing:** Pandas, NumPy, Scikit-Learn
-   **Deep Learning:** PyTorch (LSTM/GRU)
-   **APIs:** CCXT (Kraken)

## Current Status
- [x] Data Fetching Module (Kraken integration)
- [x] Statistical Tests (ADF & Rolling Metrics)
- [x] RegimeGuard Implementation (GMM Clustering)
- [x] LSTM Model Architecture & Training Loop
- [ ] Hyperparameter Tuning

## Setup
```bash
git clone [https://github.com/Jonathan0607/ethereum-volatility-pipeline.git](https://github.com/Jonathan0607/ethereum-volatility-pipeline.git)
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Run pipeline
python3 src/fetch_data.py
python3 src/train.py
python3 src/visualize.py