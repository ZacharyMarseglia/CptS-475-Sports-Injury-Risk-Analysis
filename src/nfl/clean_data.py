"""Cleaning and source-level quality checks for NFL data."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

TEAM_ALIASES = {"JAX": "JAC", "LAR": "LA", "STL": "LA", "OAK": "LV", "SD": "LAC"}


def normalize_name(value: Any) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9 ]+", " ", text).lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.upper()
    return values.replace(TEAM_ALIASES)


def base_clean(frame: pd.DataFrame, name_column: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = frame.copy()
    before = len(cleaned)
    exact_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    cleaned["season"] = pd.to_numeric(cleaned["season"], errors="coerce").astype("Int64")
    cleaned["week"] = pd.to_numeric(cleaned["week"], errors="coerce").astype("Int64")
    cleaned["team"] = normalize_team(cleaned["team"])
    cleaned["normalized_player_name"] = cleaned[name_column].map(normalize_name).astype("string")
    return cleaned, {
        "rows_raw": before,
        "exact_duplicates_removed": exact_duplicates,
        "rows_clean": len(cleaned),
        "missing_by_column": {key: int(value) for key, value in cleaned.isna().sum().items()},
        "data_types": {key: str(value) for key, value in cleaned.dtypes.items()},
    }


def clean_injuries(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned, quality = base_clean(frame, "full_name")
    cleaned["gsis_id"] = cleaned["gsis_id"].astype("string").str.strip()
    cleaned["date_modified"] = pd.to_datetime(cleaned["date_modified"], errors="coerce", utc=True)
    cleaned = cleaned.sort_values("date_modified").drop_duplicates(
        ["gsis_id", "season", "week"], keep="last"
    )
    cleaned["injury_reported"] = 1
    # An appearance on a league injury report is not proof that an injury occurred.
    cleaned["confirmed_injury"] = pd.Series(pd.NA, index=cleaned.index, dtype="boolean")
    quality["rows_after_player_week_collapse"] = len(cleaned)
    quality["confirmed_injury_note"] = (
        "Not derivable from injury reports; report_status describes availability, not confirmed injury incidence."
    )
    return cleaned, quality


def clean_snap_counts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned, quality = base_clean(frame, "player")
    cleaned["pfr_player_id"] = cleaned["pfr_player_id"].astype("string").str.strip()
    numeric = ["offense_snaps", "offense_pct", "defense_snaps", "defense_pct", "st_snaps", "st_pct"]
    cleaned[numeric] = cleaned[numeric].apply(pd.to_numeric, errors="coerce")
    duplicate_keys = cleaned.duplicated(["pfr_player_id", "season", "week"], keep=False)
    quality["duplicate_player_week_rows"] = int(duplicate_keys.sum())
    return cleaned, quality


def clean_weekly_stats(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned, quality = base_clean(frame, "player_display_name")
    cleaned = cleaned.rename(columns={"player_id": "gsis_id"})
    cleaned["gsis_id"] = cleaned["gsis_id"].astype("string").str.strip()
    duplicate_keys = cleaned.duplicated(["gsis_id", "season", "week"], keep=False)
    quality["duplicate_player_week_rows"] = int(duplicate_keys.sum())
    return cleaned, quality

