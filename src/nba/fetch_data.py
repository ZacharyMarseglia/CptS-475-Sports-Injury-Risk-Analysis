import os
import time
import pandas as pd
from nba_api.stats.endpoints import playergamelogs

def fetch_nba_game_logs(seasons=["2022-23", "2023-24", "2024-25"]):
    """
    Fetches player game logs from nba_api for the specified seasons.
    """
    all_logs = []
    for season in seasons:
        print(f"[NBA API] Fetching player game logs for season: {season}...")
        try:
            gamelogs = playergamelogs.PlayerGameLogs(season_nullable=season)
            df = gamelogs.get_data_frames()[0]
            df["SEASON"] = season
            all_logs.append(df)
            time.sleep(1) # polite pause to avoid rate limiting
        except Exception as e:
            print(f"[Warning] Failed to fetch season {season}: {e}")
            
    if all_logs:
        combined_df = pd.concat(all_logs, ignore_index=True)
        print(f"[NBA API] Successfully fetched {len(combined_df)} total game log rows.")
        return combined_df
    else:
        raise ValueError("No game logs were fetched successfully.")

def load_nba_injury_data(injury_csv_path="data/nba_injuries.csv"):
    """
    Loads raw NBA injury records from CSV.
    """
    print(f"[Data Loader] Loading injury dataset from {injury_csv_path}...")
    if not os.path.exists(injury_csv_path):
        # Fallback check if it's named with spaces or in root
        alt_path = "Injury Database - Oct 2021 - June 2024.csv"
        if os.path.exists(alt_path):
            injury_csv_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find injury CSV at {injury_csv_path} or {alt_path}")
            
    df = pd.read_csv(injury_csv_path)
    print(f"[Data Loader] Loaded {len(df)} injury records.")
    return df

# Alias for compatibility with pipeline import
load_nba_injuries = load_nba_injury_data