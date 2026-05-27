import asyncio
# pyrefly: ignore [missing-import]
import websockets
import json
import torch
import pickle
import numpy as np
import os
import ast
import urllib.request
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy import ProgressiveModel

# --- 1. The Cold Start (REST API Seed) ---
print("=== FETCHING HOURLY MACRO-CONTEXT ===")
rest_url = "https://api.binance.us/api/v3/klines?symbol=ETHUSDT&interval=1h&limit=100"
req = urllib.request.Request(rest_url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        kline_data = json.loads(response.read().decode())
        # Binance klines: [Open time, Open, High, Low, Close, Volume, ...]
        hourly_price_buffer = [float(kline[4]) for kline in kline_data]
    print(f">>> Successfully seeded buffer with {len(hourly_price_buffer)} hourly closes.\n")
except Exception as e:
    print(f"[!] ERROR FETCHING SEED DATA: {e}")
    hourly_price_buffer = []

# --- 2. Model Initialization ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
current_dir = os.path.dirname(os.path.abspath(__file__))

params_path = os.path.join(current_dir, '..', 'best_params.txt')
if os.path.exists(params_path):
    with open(params_path, 'r') as f:
        content = f.read()
        try:
            best_params = json.loads(content)
        except Exception:
            best_params = ast.literal_eval(content)
else:
    best_params = {'hidden_dim': 64, 'input_dim': 5, 'dropout': 0.2}

model = ProgressiveModel(
    input_dim=best_params.get('input_dim', 5),
    hidden_dim=best_params.get('hidden_dim', 64),
    output_dim=1,
    dropout=best_params.get('dropout', 0.2)
)
model_path = os.path.join(current_dir, '..', 'lstm_model.pth')
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
model.to(device)
model.eval()

scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

target_scaler_path = os.path.join(current_dir, '..', 'target_scaler.pkl')
with open(target_scaler_path, 'rb') as f:
    target_scaler = pickle.load(f)


def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# --- 3. The Live WebSocket Stream ---
async def stream_ethereum_data():
    socket_url = "wss://stream.binance.us:9443/ws/ethusdt@kline_1m"
    
    print("=== INITIATING QUANTITATIVE WEBSOCKET UPLINK ===")
    print(f"Connecting to: {socket_url}...\n")

    try:
        async with websockets.connect(socket_url, ping_interval=None) as websocket:
            print(">>> HANDSHAKE SUCCESSFUL. Waiting for market ticks...\n")
            
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                kline = data['k']
                is_candle_closed = kline['x']
                current_price = float(kline['c'])
                timestamp_dt = datetime.fromtimestamp(data['E'] / 1000)
                timestamp_str = timestamp_dt.strftime('%H:%M:%S')

                print(f"[{timestamp_str}] ETH/USDT Live Tick: ${current_price:,.2f}")

                if is_candle_closed:
                    print(f" >>> 1-MINUTE CANDLE CLOSED. Logging ${current_price:,.2f} to Matrix.")
                    
                    # --- 4. The Hourly Trigger & Inference ---
                    if timestamp_dt.minute == 0:
                        hourly_price_buffer.append(current_price)
                        
                        if len(hourly_price_buffer) > 100:
                            hourly_price_buffer.pop(0)
                            
                        if len(hourly_price_buffer) >= 90:
                            closes = np.array(hourly_price_buffer)
                            
                            log_return = np.log(closes[-1] / closes[-2])
                            
                            log_returns = np.log(closes[1:] / closes[:-1])
                            rolling_vol_24h = np.std(log_returns[-24:], ddof=1)
                            
                            ma_20 = np.mean(closes[-20:])
                            ma_20_dist = (closes[-1] - ma_20) / ma_20
                            
                            ma_50 = np.mean(closes[-50:])
                            ma_50_dist = (closes[-1] - ma_50) / ma_50
                            
                            # Scaler expects 5 features (volatility, log_return, rolling_vol_24h, ma_20_dist, ma_50_dist)
                            latest_features = np.array([[rolling_vol_24h, log_return, rolling_vol_24h, ma_20_dist, ma_50_dist]])
                            scaled_features = scaler.transform(latest_features)
                            
                            tensor_features = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                output_scaled = model(tensor_features)
                                
                            volatility_forecast = target_scaler.inverse_transform(output_scaled.cpu().numpy())[0][0]
                            
                            z_window = best_params.get('z_window', 20)
                            window_closes = closes[-z_window:]
                            z_score = (current_price - np.mean(window_closes)) / (np.std(window_closes, ddof=1) + 1e-8)
                            
                            hurst_window = best_params.get('hurst_window', 48)
                            hurst_value = get_hurst_exponent(closes[-hurst_window:])

                            print("\n" + "="*60)
                            print(f"🚨 HOURLY REGIME UPDATE 🚨")
                            print(f"   Real-Time Hourly Close: ${current_price:,.2f}")
                            print(f"   LSTM Volatility Forecast (24h): {volatility_forecast:.4%}")
                            print(f"   Hurst Exponent (48h): {hurst_value:.4f}")
                            print(f"   Z-Score ({z_window}h): {z_score:.4f}")
                            print("="*60 + "\n")

                            # Fire the Microservice Handshake to the FastAPI engine
                            try:
                                api_url = "http://api:8000/execution/live-stream"
                                payload = {
                                    "current_price": float(current_price),
                                    "forecasted_vol": float(volatility_forecast),
                                    "z_score": float(z_score),
                                    "hurst": float(hurst_value)
                                }
                                import requests
                                response = requests.post(api_url, json=payload, timeout=5)
                                print(f">>> Handshake successful. API Server Response: {response.json()}")
                            except Exception as e:
                                print(f">>> [Handshake Failed] Could not route signal to Execution Engine: {e}")

    except Exception as e:
        print(f"\n[!] FATAL CONNECTION ERROR: {e}")
        print("Are you on a restricted University Wi-Fi or behind a VPN?")

if __name__ == "__main__":
    try:
        asyncio.run(stream_ethereum_data())
    except KeyboardInterrupt:
        print("\n=== WEBSOCKET UPLINK SEVERED BY USER ===")