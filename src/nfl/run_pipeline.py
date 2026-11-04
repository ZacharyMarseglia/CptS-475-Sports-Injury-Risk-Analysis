"""Run Zach's configurable nflverse collection and integration pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.nfl.clean_data import clean_injuries, clean_snap_counts, clean_weekly_stats
from src.nfl.download_data import URLS, download_sources
from src.nfl.feature_engineering import engineer_historical_features
from src.nfl.merge_data import add_gsis_to_snaps, merge_sources


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nfl"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/nfl"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/nfl"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def serializable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def main() -> None:
    args = arguments()
    paths = download_sources(args.raw_dir, args.season, args.force_download) if not args.skip_download else {
        "injuries": args.raw_dir / str(args.season) / f"injuries_{args.season}.csv",
        "snap_counts": args.raw_dir / str(args.season) / f"snap_counts_{args.season}.csv",
        "weekly_stats": args.raw_dir / str(args.season) / f"weekly_stats_{args.season}.csv",
        "players": args.raw_dir / "players.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    injuries, injury_quality = clean_injuries(pd.read_csv(paths["injuries"], low_memory=False))
    snaps, snap_quality = clean_snap_counts(pd.read_csv(paths["snap_counts"], low_memory=False))
    weekly, weekly_quality = clean_weekly_stats(pd.read_csv(paths["weekly_stats"], low_memory=False))
    players = pd.read_csv(paths["players"], low_memory=False)
    snaps, unmatched_ids, id_stats = add_gsis_to_snaps(snaps, players)
    merged, unmatched, join_stats = merge_sources(weekly, snaps, injuries)
    model_ready, feature_dictionary = engineer_historical_features(merged)

    output = args.processed_dir / str(args.season)
    reports = args.report_dir / str(args.season)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "nfl_cleaned.csv", index=False)
    model_ready.to_csv(output / "nfl_model_ready.csv", index=False)
    unmatched_ids.to_csv(output / "unmatched_snap_player_ids.csv", index=False)
    for name, frame in unmatched.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    quality = {
        "season": args.season,
        "sources": {name: URLS[name].format(season=args.season) for name in URLS},
        "injuries": injury_quality,
        "snap_counts": snap_quality,
        "weekly_stats": weekly_quality,
        "id_crosswalk": id_stats,
        "joins": join_stats,
        "limitations": [
            "Injury reports are availability reports, not confirmed injury events.",
            "Rest days were not engineered because these files do not contain actual game dates.",
            "Model-ready workloads and injury history exclude the current week via shift(1).",
        ],
    }
    with (reports / "data_quality_summary.json").open("w", encoding="utf-8") as file:
        json.dump(quality, file, indent=2, default=serializable)
    pd.DataFrame(
        [
            {"source": source, "column": column, "missing_count": count}
            for source, details in [("injuries", injury_quality), ("snap_counts", snap_quality), ("weekly_stats", weekly_quality)]
            for column, count in details["missing_by_column"].items()
        ]
    ).to_csv(reports / "missing_value_report.csv", index=False)
    pd.DataFrame([id_stats | join_stats]).to_csv(reports / "join_match_statistics.csv", index=False)
    pd.DataFrame(feature_dictionary).to_csv(reports / "feature_dictionary.csv", index=False)
    print(json.dumps({"id_crosswalk": id_stats, "joins": join_stats}, indent=2))
    print(f"Wrote cleaned data to {output} and reports to {reports}")


if __name__ == "__main__":
    main()

