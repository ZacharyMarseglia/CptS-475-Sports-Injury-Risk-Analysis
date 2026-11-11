import pandas as pd
import numpy as np

def process_and_clean_nba_data(df_gamelogs, df_injuries):
    """
    Cleans and merges NBA game logs with injury records.
    """
    print("[Clean Data] Processing and cleaning NBA game logs and injuries...")
    
    # Standardize column names and dates if necessary
    df_gamelogs['GAME_DATE'] = pd.to_datetime(df_gamelogs['GAME_DATE'], errors='coerce')
    df_injuries['DATE'] = pd.to_datetime(df_injuries['DATE'], errors='coerce')
    
    # Clean player names for matching
    df_gamelogs['PLAYER_NAME_LOWER'] = df_gamelogs['PLAYER_NAME'].str.lower().str.strip()
    df_injuries['PLAYER_LOWER'] = df_injuries['PLAYER'].str.lower().str.strip()
    
    # Create injury indicator flag on exact match date
    df_injuries['is_injured_today'] = 1
    
    # Merge game logs with injuries on player and date
    merged = pd.merge(
        df_gamelogs,
        df_injuries[['PLAYER_LOWER', 'DATE', 'STATUS', 'REASON', 'is_injured_today']],
        left_on=['PLAYER_NAME_LOWER', 'GAME_DATE'],
        right_on=['PLAYER_LOWER', 'DATE'],
        how='left'
    )
    
    merged['is_injured_today'] = merged['is_injured_today'].fillna(0).astype(int)
    
    print(f"[Clean Data] Merged dataset shape: {merged.shape}")
    return merged