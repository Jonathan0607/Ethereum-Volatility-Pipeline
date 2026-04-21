import pandas as pd
import numpy as np

def split_data(df: pd.DataFrame, verbose: bool = True):
    """
    Splits data chronologically:
    - First 75% = Train (History)
    - Last 25%  = Test (Future)
    """
    # Ensure sorted by date
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Calculate the split point (75% mark)
    split_idx = int(len(df) * 0.75)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    if verbose:
        print(f"\n[Data Split] Configuration (Dynamic 80/20):")
        print(f"   - Total Rows:  {len(df)}")
        print(f"   - Train Size:  {len(train_df)} rows")
        print(f"   - Test Size:   {len(test_df)} rows")
    
    return train_df, test_df