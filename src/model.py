import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Standard LSTM architecture for Time Series Forecasting.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        # .to(x.device) ensures it works on both CPU and GPU/MPS automatically
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the LAST time step
        # We only care about the final prediction
        out = self.fc(out[:, -1, :])
        return out

# --- SELF-TEST BLOCK ---
# This runs only if you execute 'python src/model.py' directly.
# It proves the model architecture is valid without needing data.
if __name__ == "__main__":
    print("Testing LSTM Model Architecture...")
    
    # 1. Create a dummy input (Batch=32, Seq_Len=60, Features=1)
    # This simulates 32 different 60-hour windows of volatility data
    dummy_input = torch.randn(32, 60, 1)
    
    # 2. Initialize Model
    model = LSTMModel(input_dim=1, hidden_dim=64, output_dim=1, dropout=0.2)
    
    # 3. Forward Pass
    try:
        output = model(dummy_input)
        print(f"Input Shape: {dummy_input.shape}")
        print(f"Output Shape: {output.shape}")
        
        # Check if output is (32, 1)
        if output.shape == (32, 1):
            print("✅ SUCCESS: Model dimensions are correct.")
        else:
            print("❌ ERROR: Output shape is wrong.")
            
    except Exception as e:
        print(f"❌ CRASHED: {e}")