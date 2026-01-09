# Ethereum Volatility & Market Regime Detector

## Overview

A hybrid machine learning pipeline designed to forecast Ethereum volatility and classify market risk regimes in real-time. By combining statistical methods (GARCH) with deep learning (LSTMs) and unsupervised clustering (GMM), this system provides actionable risk signals for algorithmic trading strategies.

## 🚀 Key Features

* **Hybrid Architecture:** Integrates **GARCH** (Generalized Autoregressive Conditional Heteroskedasticity) variance features into an **LSTM** (Long Short-Term Memory) network to capture both linear and non-linear market dependencies.
* **Dynamic Regime Detection:** Utilizes **Gaussian Mixture Models (GMM)** to automatically cluster market conditions into "Stable" vs. "High Volatility" regimes, enabling dynamic risk-off/risk-on logic.
* **Automated Optimization:** Implements Bayesian Optimization via **Optuna** to systematically tune hyperparameters (Learning Rate, Dropout, Hidden Layers), reducing validation loss by **~97%** compared to baseline models.
* **Robust Data Pipeline:** Fetches real-time OHLCV data via Kraken API (`ccxt`), enforcing stationarity through log-return transformations and standard scaling.

## 🛠️ Tech Stack

* **Core:** Python 3.10+, Pandas, NumPy
* **Deep Learning:** PyTorch (MPS/CUDA support enabled)
* **Statistical Modeling:** Arch (GARCH), Scikit-Learn (GMM, Scalers)
* **Optimization:** Optuna
* **Visualization:** Matplotlib, Seaborn

## 📊 Performance & Optimization

The model architecture was optimized using `Optuna` to maximize predictive accuracy on unseen validation data.

* **Optimization Method:** Bayesian Search (TPE Sampler)
* **Search Space:**
    * Learning Rate: `1e-4` to `1e-2` (Log scale)
    * Hidden Dimensions: `[32, 64, 128]`
    * Dropout: `0.1` to `0.5`
* **Results:**
    * *Baseline Loss:* `1.39`
    * *Optimized Loss:* **`0.034`**
    * *Best Config:* Single-layer LSTM (128 units) with moderate dropout (`0.25`) proved most robust against overfitting.

## 📂 Project Structure

```bash
├── data/
│   ├── raw/             # Raw data from Kraken API
│   └── eth_hourly.csv   # Cleaned, stationary data with GARCH features
├── src/
│   ├── fetch_data.py    # Data ingestion script
│   ├── process_data.py  # Cleaning & Log-return transformation
│   ├── tune.py          # Optuna hyperparameter optimization script
│   ├── train.py         # Final model training loop
│   └── inference.py     # (Planned) Real-time prediction script
├── notebooks/           # EDA and Experimentation
└── README.md