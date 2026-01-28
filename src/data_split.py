import pandas as pd

# --- CONFIGURATION ---
TEST_START_DATE = "2025-01-01"

def split_data(df: pd.DataFrame, verbose: bool = False):
    """
    Splits the DataFrame.
    verbose=True: Prints the config (Use this only once per pipeline run).
    verbose=False: Runs silently.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
    mask = (df.index < TEST_START_DATE)
    
    train_df = df.loc[mask].copy()
    test_df = df.loc[~mask].copy()
    
    if verbose:
        print(f"[Data Split] Configuration:")
        print(f"   - Cutoff Date: {TEST_START_DATE}")
        print(f"   - Train Set:   {len(train_df)} rows")
        print(f"   - Test Set:    {len(test_df)} rows\n")
    
    return train_df, test_df