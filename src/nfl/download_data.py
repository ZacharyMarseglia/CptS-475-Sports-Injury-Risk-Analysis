"""Download and validate nflverse source files."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

URLS = {
    "injuries": "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv",
    "snap_counts": "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv",
    "weekly_stats": "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv",
    "players": "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv",
}

REQUIRED_COLUMNS = {
    "injuries": {"season", "week", "team", "gsis_id", "full_name", "report_status"},
    "snap_counts": {"season", "week", "team", "pfr_player_id", "player", "offense_snaps", "defense_snaps"},
    "weekly_stats": {"season", "week", "team", "player_id", "player_display_name", "position"},
    "players": {"gsis_id", "pfr_id", "display_name"},
}


def source_paths(raw_dir: Path, season: int) -> dict[str, Path]:
    return {
        "injuries": raw_dir / str(season) / f"injuries_{season}.csv",
        "snap_counts": raw_dir / str(season) / f"snap_counts_{season}.csv",
        "weekly_stats": raw_dir / str(season) / f"weekly_stats_{season}.csv",
        "players": raw_dir / "players.csv",
    }


def validate_schema(path: Path, source: str) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    missing = REQUIRED_COLUMNS[source].difference(columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")
    return columns


def download_sources(raw_dir: Path, season: int, force: bool = False) -> dict[str, Path]:
    """Download official CSV assets and fail immediately on schema drift."""
    paths = source_paths(raw_dir, season)
    for source, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if force or not path.exists():
            url = URLS[source].format(season=season)
            temporary = path.with_suffix(".download")
            try:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        validate_schema(path, source)
    return paths

