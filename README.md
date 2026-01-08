# Ethereum Volatility Prediction Pipeline

## Overview
This project is an end-to-end quantitative research infrastructure designed to forecast short-term volatility in Ethereum (ETH/USDT) markets. Unlike traditional price prediction models, this system focuses on **volatility clustering** and **regime detection** to assess market risk.

## Architecture
The system follows a standard ETL + Inference pattern:
1.  **Data Ingestion:** High-frequency OHLCV tick data fetched via CCXT (Binance API).
2.  **Feature Engineering:** - Log-Returns transformation for stationarity.
    - Rolling Volatility (GARCH proxies).
    - Augmented Dickey-Fuller (ADF) tests for stationarity checks.
3.  **Regime Detection:** Gaussian Mixture Models (GMM) to classify market states (e.g., *Stable* vs. *Turbulent*).
4.  **Modeling:** LSTM (Long Short-Term Memory) networks for sequence forecasting.

## Tech Stack
-   **Orchestration:** Apache Airflow
-   **Data Processing:** Pandas, NumPy, Scikit-Learn
-   **Deep Learning:** PyTorch (LSTM/GRU)
-   **APIs:** CCXT

## Current Status (In Progress)
- [x] Data Fetching Module (CCXT integration)
- [x] Statistical Tests (ADF & Rolling Metrics)
- [ ] Airflow DAG configuration
- [ ] LSTM Hyperparameter Tuning
- [ ] RegimeGuard Implementation (GMM)

## Setup
```bash
git clone [https://github.com/jonathan-ebenezer/ethereum-volatility-pipeline.git](https://github.com/jonathan-ebenezer/ethereum-volatility-pipeline.git)
pip install -r requirements.txt
python src/fetch_data.py
