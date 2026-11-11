import argparse
import json
import os
import pandas as pd

from src.nba.fetch_data import fetch_nba_game_logs, load_nba_injuries
from src.nba.clean_data import process_and_clean_nba_data
from src.nba.engineer_features import engineer_nba_features

def main():
    print("[DEBUG] Pipeline started successfully!")
    parser = argparse.ArgumentParser(description="NBA Data Collection and Preprocessing Pipeline")
    parser.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25"], help="NBA seasons to download")
    parser.add_argument("--injury-csv", type=str, default="data/nba_injuries.csv", help="Path to raw NBA injury CSV file")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save output CSVs")
    parser.add_argument("--report-dir", type=str, default="results/nba", help="Directory for quality reports")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    df_gamelogs = fetch_nba_game_logs(seasons=args.seasons)
    df_injuries = load_nba_injuries(injury_csv_path=args.injury_csv)
    df_merged = process_and_clean_nba_data(df_gamelogs, df_injuries)
    df_cleaned, df_model_ready = engineer_nba_features(df_merged)

    cleaned_path = os.path.join(args.output_dir, "nba_cleaned.csv")
    model_ready_path = os.path.join(args.output_dir, "nba_model_ready.csv")

    df_cleaned.to_csv(cleaned_path, index=False)
    df_model_ready.to_csv(model_ready_path, index=False)

    print(f"\n[NBA Pipeline] Success!")
    print(f"  Cleaned dataset retained for audit: {cleaned_path} (Shape: {df_cleaned.shape})")
    print(f"  Model-ready dataset saved: {model_ready_path} (Shape: {df_model_ready.shape})")

    quality_report = {
        "seasons_processed": args.seasons,
        "total_game_log_rows": int(len(df_gamelogs)),
        "total_cleaned_rows": int(len(df_cleaned)),
        "total_model_ready_rows": int(len(df_model_ready)),
        "total_injury_events_matched": int(df_cleaned["is_injured_today"].sum()),
        "positive_target_class_count": int(df_model_ready["target_injured_next_14d"].sum()),
        "positive_target_class_pct": float(round(df_model_ready["target_injured_next_14d"].mean() * 100, 2))
    }

    report_path = os.path.join(args.report_dir, "data_quality_summary.json")
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=4)
    print(f"  Data quality report saved to: {report_path}")

if __name__ == "__main__":
    main()