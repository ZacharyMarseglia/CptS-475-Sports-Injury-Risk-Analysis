# Zach's NFL Data Contribution — Sideline Science

## Sources

The configurable pipeline uses official CSV releases from
[`nflverse/nflverse-data`](https://github.com/nflverse/nflverse-data):

- season injury reports (`injuries_<season>.csv`)
- season snap counts (`snap_counts_<season>.csv`)
- weekly player statistics (`stats_player_week_<season>.csv`)
- the players ID crosswalk (`players.csv`)

The crosswalk is necessary because injury reports and weekly statistics use
GSIS IDs while snap counts use Pro Football Reference IDs.

## Cleaning and integration

The pipeline removes exact duplicates; standardizes names, team abbreviations,
seasons, and weeks; collapses changing injury reports to the latest report per
player-week; and validates one-to-one player-season-week merges. Ambiguous or
missing PFR-to-GSIS mappings and unmatched source records are saved separately
for review. Missing values, schemas, and match rates are written to reports.

Injury reports describe a player's listed injury and availability status. They
do **not** establish that a new injury occurred, so `confirmed_injury` remains
unknown rather than being inferred from `report_status`.

## Features

Where the 2024 schemas support them, the cleaned dataset includes offensive and
defensive snaps and percentages, rushing and receiving workload, passing
workload, and player position. The model-ready output calculates prior 1-, 2-,
and 4-week workload averages and prior injury-report counts. Every historical
feature is shifted by one player-week to prevent current-week leakage.

Rest days are not engineered because the selected files contain report update
timestamps but not authoritative game dates.

## Results and remaining challenges

The successful November 3, 2026 pipeline run produced 27,356 cleaned and 27,356
model-ready player-week rows from the 2024 data. The source files contained
6,263 injury-report rows, 26,615 snap-count rows, and 18,983 weekly-stat rows
before regular-season filtering and player-week consolidation.

- 26,573 of 26,615 snap rows (99.8422%) mapped from PFR to GSIS IDs; 42 did not.
- 18,098 of 18,130 regular-season weekly-stat rows (99.8235%) matched a snap row.
- 3,987 of 5,952 consolidated regular-season injury reports (66.9859%) matched
  a workload row.
- All 18,098 matched stat/snap rows agreed on standardized team; 3,985 of 3,987
  injury/workload matches agreed on team and the two disagreements remain
  available for review.
- No ambiguous PFR IDs were admitted into the crosswalk.

Missingness is position/stat dependent: for example, passing efficiency is
normally absent for non-passers and kicking fields are absent for non-kickers.
The complete per-column counts are in `missing_value_report.csv`; the pipeline
does not replace structurally inapplicable values indiscriminately.

Remaining unmatched records are retained under `data/processed/nfl/2024/` for
manual review. The main interpretation limitation is that injury-report status
is not a confirmed injury outcome; a future team decision should identify a
defensible target before modeling.

