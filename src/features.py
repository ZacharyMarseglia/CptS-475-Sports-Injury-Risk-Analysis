"""Conservative feature engineering for common athlete datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_column(columns: pd.Index, aliases: set[str]) -> str | None:
    normalized = {str(column).strip().lower().replace(" ", "_"): column for column in columns}
    return normalized.get(next((alias for alias in aliases if alias in normalized), ""))


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add only features supported by recognizable columns in the input data."""
    result = frame.copy()
    added: list[str] = []

    weight = _find_column(result.columns, {"weight_kg", "weight"})
    height = _find_column(result.columns, {"height_m", "height"})
    if weight and height:
        height_values = pd.to_numeric(result[height], errors="coerce")
        weight_values = pd.to_numeric(result[weight], errors="coerce")
        # Treat values over 3 as centimeters, a common dataset convention.
        height_m = height_values.where(height_values <= 3, height_values / 100)
        result["engineered_bmi"] = weight_values / height_m.pow(2).replace(0, np.nan)
        added.append("engineered_bmi")

    acute = _find_column(result.columns, {"acute_workload", "weekly_workload", "current_workload"})
    chronic = _find_column(result.columns, {"chronic_workload", "average_workload", "baseline_workload"})
    if acute and chronic:
        denominator = pd.to_numeric(result[chronic], errors="coerce").replace(0, np.nan)
        result["engineered_acute_chronic_ratio"] = (
            pd.to_numeric(result[acute], errors="coerce") / denominator
        )
        added.append("engineered_acute_chronic_ratio")

    return result, added

