import pandas as pd
import numpy as np

def engineer_nba_features(df):
    """
    Engineers rolling workload, rest days, and target labels for injury prediction.
    """
    print("[NBA Feature Engineering] Building rolling workload and rest indicators...")
    
    # Ensure standard column name casing for merging and sorting
    if 'PLAYER_ID' in df.columns and 'player_id' not in df.columns:
        df['player_id'] = df['PLAYER_ID']
    if 'GAME_DATE' in df.columns and 'game_date' not in df.columns:
        df['game_date'] = pd.to_datetime(df['GAME_DATE'])

    # Sort chronologically per player for rolling calculations
    df = df.sort_values(by=["player_id", "game_date"]).copy()
    
    # Calculate days rest between games
    df['prev_game_date'] = df.groupby('player_id')['game_date'].shift(1)
    df['rest_days'] = (df['game_date'] - df['prev_game_date']).dt.days.fillna(7)
    
    # Rolling workload features (e.g., minutes played in last 5 games)
    if 'MIN' in df.columns:
        df['rolling_min_5g'] = df.groupby('player_id')['MIN'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
    else:
        df['rolling_min_5g'] = 0.0
        
    # Define target label: Will the player be injured within the next 14 days?
    # Create future injury window check
    df['future_injury_window'] = df.groupby('player_id')['is_injured_today'].rolling(window=14, min_periods=1).max().shift(-14).reset_index(0, drop=True).fillna(0)
    
    df['target_injured_next_14d'] = df['future_injury_window'].astype(int)
    
    # Clean up intermediate helper columns
    df_cleaned = df.drop(columns=['prev_game_date', 'future_injury_window'], errors='ignore')
    df_model_ready = df_cleaned.dropna(subset=['rolling_min_5g'])
    
    print(f"[NBA Feature Engineering] Completed. Model-ready shape: {df_model_ready.shape}")
    return df_cleaned, df_model_ready