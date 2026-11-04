import pandas as pd
import pytest

from src.nfl.clean_data import clean_injuries, normalize_name
from src.nfl.feature_engineering import engineer_historical_features
from src.nfl.merge_data import merge_sources


def test_normalize_name_removes_suffix_and_punctuation():
    assert normalize_name("D'Andre Smith Jr.") == "d andre smith"


def test_injury_cleaning_keeps_latest_report_and_does_not_claim_confirmation():
    raw = pd.DataFrame({
        "season": [2024, 2024], "week": [1, 1], "team": ["JAX", "JAX"],
        "gsis_id": ["00-1", "00-1"], "full_name": ["Test Player", "Test Player"],
        "game_type": ["REG", "REG"], "report_status": ["Questionable", "Out"],
        "date_modified": ["2024-09-01T00:00:00Z", "2024-09-02T00:00:00Z"],
    })
    cleaned, _ = clean_injuries(raw)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["report_status"] == "Out"
    assert pd.isna(cleaned.iloc[0]["confirmed_injury"])
    assert cleaned.iloc[0]["team"] == "JAC"


def test_historical_features_exclude_current_week():
    frame = pd.DataFrame({
        "gsis_id": ["p1", "p1", "p1"], "season": [2024] * 3, "week": [1, 2, 3],
        "attempts": [10, 20, 999], "injury_reported": [0, 1, 0],
    })
    model_ready, _ = engineer_historical_features(frame)
    assert pd.isna(model_ready.loc[0, "prior_1w_avg_attempts"])
    assert model_ready.loc[1, "prior_1w_avg_attempts"] == 10
    assert model_ready.loc[2, "prior_2w_avg_attempts"] == 15
    assert model_ready.loc[2, "prior_2w_injury_reports"] == 1


def test_merge_rejects_many_to_many_keys():
    stats = pd.DataFrame({
        "gsis_id": ["p1", "p1"], "season": [2024, 2024], "week": [1, 1],
        "season_type": ["REG", "REG"],
    })
    snaps = pd.DataFrame({
        "gsis_id": ["p1"], "season": [2024], "week": [1], "game_type": ["REG"]
    })
    injuries = pd.DataFrame({
        "gsis_id": ["p1"], "season": [2024], "week": [1], "game_type": ["REG"]
    })
    with pytest.raises(ValueError, match="non-unique"):
        merge_sources(stats, snaps, injuries)


def test_model_ready_keeps_position_and_team_when_available():
    frame = pd.DataFrame({
        "gsis_id": ["p1"], "season": [2024], "week": [1], "injury_reported": [0],
        "player_name": ["Player One"], "position": ["RB"], "team": ["SEA"],
    })
    model_ready, _ = engineer_historical_features(frame)
    assert model_ready.loc[0, "position"] == "RB"
    assert model_ready.loc[0, "team"] == "SEA"

