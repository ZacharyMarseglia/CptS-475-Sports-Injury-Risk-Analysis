"""Leakage-safe historical NFL workload features."""

from __future__ import annotations

import pandas as pd

WORKLOAD_COLUMNS = [
    "offense_snaps", "offense_pct", "defense_snaps", "defense_pct",
    "carries", "rushing_yards", "receptions", "targets", "receiving_yards",
    "completions", "attempts", "passing_yards", "sacks_suffered",
]


def engineer_historical_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    result = frame.sort_values(["gsis_id", "season", "week"]).copy()
    available = [column for column in WORKLOAD_COLUMNS if column in result]
    dictionary: list[dict[str, str]] = []
    groups = result.groupby(["gsis_id", "season"], dropna=False)
    for column in available:
        numeric = pd.to_numeric(result[column], errors="coerce")
        result[column] = numeric
        shifted = groups[column].shift(1)
        for window in (1, 2, 4):
            feature = f"prior_{window}w_avg_{column}"
            if window == 1:
                result[feature] = shifted
            else:
                result[feature] = shifted.groupby([result["gsis_id"], result["season"]]).transform(
                    lambda values: values.rolling(window, min_periods=1).mean()
                )
            dictionary.append({
                "feature": feature,
                "description": f"Mean {column} over up to {window} prior played weeks; current week excluded.",
                "source": "weekly_stats" if column not in {"offense_snaps", "offense_pct", "defense_snaps", "defense_pct"} else "snap_counts",
                "leakage_control": "shift(1) within player and season",
            })

    reported = pd.to_numeric(result["injury_reported"], errors="coerce").fillna(0)
    shifted_reported = reported.groupby([result["gsis_id"], result["season"]]).shift(1)
    for window in (1, 2, 4):
        feature = f"prior_{window}w_injury_reports"
        result[feature] = shifted_reported.groupby([result["gsis_id"], result["season"]]).transform(
            lambda values: values.rolling(window, min_periods=1).sum()
        )
        dictionary.append({
            "feature": feature,
            "description": f"Count of injury-report appearances over up to {window} prior weeks.",
            "source": "injuries",
            "leakage_control": "shift(1) within player and season",
        })

    # Dates in these sources are report modification timestamps, not game dates.
    # Rest days are intentionally not inferred from them.
    identifiers = ["gsis_id", "season", "week"]
    stable = [column for column in ["player_name", "normalized_player_name", "position", "position_group", "team"] if column in result]
    historical = [item["feature"] for item in dictionary]
    target = ["injury_reported", "report_status"]
    return result[identifiers + stable + historical + [column for column in target if column in result]], dictionary

