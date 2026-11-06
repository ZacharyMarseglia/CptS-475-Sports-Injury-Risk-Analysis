"""Validated one-to-one integration of nflverse NFL sources."""

from __future__ import annotations

from typing import Any

import pandas as pd

KEYS = ["gsis_id", "season", "week"]


def add_gsis_to_snaps(snaps: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    crosswalk = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates()
    ambiguous = crosswalk[crosswalk.duplicated("pfr_id", keep=False)]
    valid = crosswalk[~crosswalk["pfr_id"].isin(ambiguous["pfr_id"])].rename(
        columns={"pfr_id": "pfr_player_id"}
    )
    mapped = snaps.merge(valid, on="pfr_player_id", how="left", validate="many_to_one", indicator="id_match")
    unmatched = mapped[mapped["gsis_id"].isna()].copy()
    stats = {
        "snap_rows": len(mapped),
        "snap_id_matches": int(mapped["gsis_id"].notna().sum()),
        "snap_id_unmatched": len(unmatched),
        "snap_id_match_rate": float(mapped["gsis_id"].notna().mean()),
        "ambiguous_pfr_ids_excluded": int(ambiguous["pfr_id"].nunique()),
    }
    return mapped.drop(columns="id_match"), unmatched, stats


def _assert_unique(frame: pd.DataFrame, label: str) -> None:
    duplicates = frame.duplicated(KEYS, keep=False)
    if duplicates.any():
        raise ValueError(f"{label} has {int(duplicates.sum())} non-unique player-season-week rows")


def merge_sources(
    stats: pd.DataFrame, snaps: pd.DataFrame, injuries: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    stats_reg = stats[stats["season_type"].eq("REG")].copy()
    snaps_reg = snaps[snaps["game_type"].eq("REG") & snaps["gsis_id"].notna()].copy()
    injury_reg = injuries[injuries["game_type"].eq("REG")].copy()
    for frame, label in [(stats_reg, "weekly_stats"), (snaps_reg, "snap_counts"), (injury_reg, "injuries")]:
        _assert_unique(frame, label)

    workload = stats_reg.merge(
        snaps_reg.drop(columns=["normalized_player_name"], errors="ignore"),
        on=KEYS,
        how="outer",
        suffixes=("_stats", "_snaps"),
        validate="one_to_one",
        indicator="stats_snap_match",
    )
    injury_columns = KEYS + [
        "full_name", "position", "team",
        "report_primary_injury", "report_secondary_injury", "report_status",
        "practice_primary_injury", "practice_secondary_injury", "practice_status",
        "date_modified", "injury_reported", "confirmed_injury",
    ]
    injury_for_join = injury_reg[injury_columns].rename(
        columns={"full_name": "player_name_injury", "position": "position_injury", "team": "team_injury"}
    )
    merged = workload.merge(
        injury_for_join, on=KEYS, how="outer", validate="one_to_one", indicator="injury_match"
    )
    merged["injury_reported"] = merged["injury_reported"].fillna(0).astype("Int8")
    merged["player_name"] = merged["player_display_name"].combine_first(merged["player_name_injury"])
    merged["position"] = merged["position_stats"].combine_first(merged["position_snaps"]).combine_first(
        merged["position_injury"]
    )
    merged["team_workload"] = merged["team_stats"].combine_first(merged["team_snaps"])
    merged["team"] = merged["team_workload"].combine_first(merged["team_injury"])

    unmatched = {
        "stats_without_snaps": merged[merged["stats_snap_match"].eq("left_only")].copy(),
        "snaps_without_stats": merged[merged["stats_snap_match"].eq("right_only")].copy(),
        "injuries_without_workload": merged[merged["injury_match"].eq("right_only")].copy(),
    }
    stats_summary = {
        "merged_rows": len(merged),
        "stats_rows_regular_season": len(stats_reg),
        "snap_rows_regular_season_with_gsis": len(snaps_reg),
        "injury_player_week_rows_regular_season": len(injury_reg),
        "stats_with_snap_match": int((merged["stats_snap_match"] == "both").sum()),
        "stats_snap_match_rate_from_stats": float(
            (merged["stats_snap_match"] == "both").sum() / len(stats_reg) if len(stats_reg) else 0
        ),
        "injury_rows_with_workload": int((injury_reg[KEYS].merge(workload[KEYS], on=KEYS, how="inner")).shape[0]),
        "stats_snap_team_agreements": int(
            (merged["team_stats"].notna() & merged["team_snaps"].notna() & merged["team_stats"].eq(merged["team_snaps"])).sum()
        ),
        "injury_workload_team_agreements": int(
            (merged["team_injury"].notna() & merged["team_workload"].notna() & merged["team_injury"].eq(merged["team_workload"])).sum()
        ),
    }
    stats_summary["injury_workload_match_rate"] = (
        stats_summary["injury_rows_with_workload"] / len(injury_reg) if len(injury_reg) else 0
    )
    return merged, unmatched, stats_summary

