import pandas as pd

TEST_START_DATE = "2025-01-01"

def split_data(df: pd.DataFrame):
    """
    Splits the DataFrame into Training and Testing sets based on a cutoff date.
    
    Args:
        df (pd.DataFrame): The full dataset with a DatetimeIndex.
        
    Returns:
        tuple: (train_df, test_df)
    """
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
    mask = (df.index < TEST_START_DATE)
    
    train_df = df.loc[mask].copy()
    test_df = df.loc[~mask].copy()
    
    print(f"\n[Data Split] Configuration:")
    print(f"   - Cutoff Date: {TEST_START_DATE}")
    print(f"   - Train Set:   {len(train_df)} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"   - Test Set:    {len(test_df)} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df