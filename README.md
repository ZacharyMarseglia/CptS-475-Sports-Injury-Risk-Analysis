# Sports Injury Risk Analysis

CptS 475 data science project for identifying workload, training, performance,
and injury-history factors associated with sports injuries, then comparing
classification models.

> **Current status:** the repository did not contain a dataset when this project
> was initialized. The code is ready for a CSV, but no results are reported
> until a real dataset is added and the target column is specified.

## Project structure

```text
data/                 # Raw CSV input and generated cleaned data (ignored by Git)
notebooks/            # Optional exploratory notebooks
results/              # Metrics, reports, and confusion matrices
src/                  # Reproducible cleaning, EDA, and modeling code
visualizations/       # EDA and ROC charts
CHECK_IN_TEMPLATE.md  # CptS 475 team check-in
```

## Local setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the real CSV in `data/` and run:

```bash
python -m src.run_analysis --data data/your_dataset.csv --target injury_column
```

Use `--positive-label` if the injured class is not naturally encoded as `1`
(for example, `--positive-label Yes`). Use `--test-size 0.2` and
`--random-state 42` to override the defaults. Run `python -m src.run_analysis
--help` for all options.

## Google Colab

```python
!git clone https://github.com/ZacharyMarseglia/CptS-475-Sports-Injury-Risk-Analysis.git
%cd CptS-475-Sports-Injury-Risk-Analysis
!pip install -r requirements.txt
```

Upload the CSV into the Colab session's `data/` directory, then run:

```python
!python -m src.run_analysis --data data/your_dataset.csv --target injury_column
```

Download `results/` and `visualizations/` before the Colab session ends.

## What the pipeline does

1. Reports shape, types, missing values, duplicates, and target distribution.
2. Removes exact duplicates and rows with missing targets.
3. Performs the train/test split **before** fitting imputers, encoders, or
   scalers to prevent leakage.
4. Median-imputes and standardizes numeric features; most-frequent-imputes and
   one-hot encodes categorical features.
5. Adds BMI or acute-to-chronic workload ratio only when recognizable source
   columns exist; otherwise leaves the original features unchanged.
6. Compares Logistic Regression, Decision Tree, and Random Forest using
   accuracy, precision, recall, F1, confusion matrices, and ROC-AUC when a
   probability score is available.
7. Saves descriptive statistics, model results, predictions, and plots.

The pipeline supports binary classification. Common ID columns (and unique text
labels) are removed automatically when every non-null value is unique; inspect
the saved `data_profile.json` to confirm the selected features.

## Expected outputs

- `results/data_profile.json`
- `results/descriptive_statistics.csv`
- `results/model_comparison.csv`
- `results/classification_reports.json`
- `results/test_predictions.csv`
- `results/confusion_matrix_*.png`
- `visualizations/injury_distribution.png`
- `visualizations/numeric_feature_correlations.png`
- `visualizations/workload_vs_injury.png` when a workload-like column exists
- `visualizations/roc_curves.png`

Generated data and outputs are ignored by Git so private or large datasets are
not accidentally committed. Add only intentionally shareable artifacts.

