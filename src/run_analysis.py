"""Command-line pipeline for sports injury EDA and model comparison."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import add_engineered_features

MODEL_DEFAULTS = {
    "logistic_regression": {
        "C": 1.0, "max_iter": 2000, "class_weight": "balanced",
    },
    "decision_tree": {
        "max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2,
        "ccp_alpha": 0.0, "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 300, "max_depth": None, "min_samples_leaf": 1,
        "min_samples_split": 2, "max_features": "sqrt",
        "class_weight": "balanced", "n_jobs": -1,
    },
}


def model_value(text: str) -> Any:
    # CLI values may be numbers, strings, or none/null.
    if text.lower() == "none":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def validate_model_options(model: str, options: dict[str, Any]) -> None:
    for key, value in options.items():
        name = f"{model}.{key}"
        if key in {"max_depth", "max_iter", "n_estimators",
                   "min_samples_leaf", "min_samples_split"}:
            if key == "max_depth" and value is None:
                continue
            minimum = 2 if key == "min_samples_split" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}: expected integer >= {minimum}")
        elif key in {"C", "ccp_alpha"}:
            if (type(value) not in (int, float)
                    or not float("-inf") < value < float("inf")
                    or (value <= 0 if key == "C" else value < 0)):
                raise ValueError(f"{name}: expected finite {'positive' if key == 'C' else 'nonnegative'} number")
        elif key == "class_weight":
            allowed = [None, "balanced"]
            if model == "random_forest":
                allowed.append("balanced_subsample")
            if value not in allowed:
                raise ValueError(f"{name}: expected one of {allowed}")
        elif key == "max_features":
            if not (value is None or value in ("sqrt", "log2")
                    or (type(value) is int and value >= 1)
                    or (type(value) is float and 0 < value <= 1)):
                raise ValueError(f"{name}: expected sqrt, log2, null, positive integer, or fraction in (0, 1]")
        elif key == "n_jobs":
            if type(value) is not int or value == 0:
                raise ValueError(f"{name}: expected nonzero integer")


def load_model_options(path: Path | None) -> dict[str, dict[str, Any]]:
    models = {name: defaults.copy() for name, defaults in MODEL_DEFAULTS.items()}
    if path is None:
        return models

    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    unknown = config.keys() - {"models"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    file_models = config.get("models", {})
    if not isinstance(file_models, dict):
        raise ValueError("models must be a JSON object")

    for name, options in file_models.items():
        if name not in models:
            raise ValueError(f"Unknown model: {name}")
        if not isinstance(options, dict):
            raise ValueError(f"{name} must be a JSON object")
        unknown = options.keys() - models[name].keys()
        if unknown:
            raise ValueError(f"Unknown options for {name}: {sorted(unknown)}")
        models[name].update(options)
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to the input CSV")
    parser.add_argument("--target", required=True, help="Binary injury target column")
    parser.add_argument("--positive-label", help="Value to treat as the injured/positive class")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--visualizations-dir", type=Path, default=Path("visualizations"))
    parser.add_argument("--config", type=Path, help="Optional JSON config file")
    parser.add_argument("--lr-c", help="Regularization parameter")
    parser.add_argument("--lr-max-iter", help="Maximum iterations")
    parser.add_argument("--lr-class-weight", help="balanced or none")
    parser.add_argument("--dt-max-depth", help="Maximum depth or none")
    parser.add_argument("--dt-min-samples-leaf", help="Minimum samples per leaf")
    parser.add_argument("--dt-min-samples-split", help="Minimum samples to split")
    parser.add_argument("--dt-ccp-alpha", help="Pruning parameter")
    parser.add_argument("--dt-class-weight", help="balanced or none")
    parser.add_argument("--rf-n-estimators", help="Number of trees")
    parser.add_argument("--rf-max-depth", help="Maximum depth or none")
    parser.add_argument("--rf-min-samples-leaf", help="Minimum samples per leaf")
    parser.add_argument("--rf-min-samples-split", help="Minimum samples to split")
    parser.add_argument("--rf-max-features", help="Features considered per split")
    parser.add_argument("--rf-class-weight", help="balanced, balanced_subsample, or none")
    parser.add_argument("--rf-n-jobs", help="Number of parallel jobs")

    args = parser.parse_args()
    try:
        args.models = load_model_options(args.config)

        if args.lr_c is not None:
            args.models["logistic_regression"]["C"] = model_value(args.lr_c)

        if args.lr_max_iter is not None:
            args.models["logistic_regression"]["max_iter"] = model_value(args.lr_max_iter)

        if args.lr_class_weight is not None:
            args.models["logistic_regression"]["class_weight"] = model_value(args.lr_class_weight)

        if args.dt_max_depth is not None:
            args.models["decision_tree"]["max_depth"] = model_value(args.dt_max_depth)

        if args.dt_min_samples_leaf is not None:
            args.models["decision_tree"]["min_samples_leaf"] = model_value(args.dt_min_samples_leaf)

        if args.dt_min_samples_split is not None:
            args.models["decision_tree"]["min_samples_split"] = model_value(args.dt_min_samples_split)

        if args.dt_ccp_alpha is not None:
            args.models["decision_tree"]["ccp_alpha"] = model_value(args.dt_ccp_alpha)

        if args.dt_class_weight is not None:
            args.models["decision_tree"]["class_weight"] = model_value(args.dt_class_weight)

        if args.rf_n_estimators is not None:
            args.models["random_forest"]["n_estimators"] = model_value(args.rf_n_estimators)

        if args.rf_max_depth is not None:
            args.models["random_forest"]["max_depth"] = model_value(args.rf_max_depth)

        if args.rf_min_samples_leaf is not None:
            args.models["random_forest"]["min_samples_leaf"] = model_value(args.rf_min_samples_leaf)

        if args.rf_min_samples_split is not None:
            args.models["random_forest"]["min_samples_split"] = model_value(args.rf_min_samples_split)

        if args.rf_max_features is not None:
            args.models["random_forest"]["max_features"] = model_value(args.rf_max_features)

        if args.rf_class_weight is not None:
            args.models["random_forest"]["class_weight"] = model_value(args.rf_class_weight)

        if args.rf_n_jobs is not None:
            args.models["random_forest"]["n_jobs"] = model_value(args.rf_n_jobs)

        for name, options in args.models.items():
            validate_model_options(name, options)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    return args


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_and_profile(path: Path, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Target {target!r} not found. Available columns: {list(frame.columns)}")

    profile = {
        "source_file": str(path),
        "rows_before_cleaning": len(frame),
        "columns": list(frame.columns),
        "data_types": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(count) for column, count in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_distribution_before_cleaning": {
            str(key): int(value) for key, value in frame[target].value_counts(dropna=False).items()
        },
    }
    return frame, profile


def prepare_target(series: pd.Series, positive_label: str | None) -> tuple[pd.Series, dict[str, int]]:
    values = series.astype(str)
    classes = sorted(values.unique())
    if len(classes) != 2:
        raise ValueError(f"Binary classification requires exactly 2 target classes; found {classes}")

    if positive_label is not None:
        matches = [value for value in classes if value == positive_label]
        if not matches:
            raise ValueError(f"Positive label {positive_label!r} is not one of {classes}")
        negative = next(value for value in classes if value != positive_label)
        mapping = {negative: 0, positive_label: 1}
    else:
        encoder = LabelEncoder()
        encoder.fit(classes)
        mapping = {str(label): int(code) for code, label in enumerate(encoder.classes_)}
        print(f"No --positive-label supplied; using deterministic mapping {mapping}")
    return values.map(mapping).astype(int), mapping


def save_eda(
    frame: pd.DataFrame, target: str, output_dir: Path, results_dir: Path
) -> list[str]:
    sns.set_theme(style="whitegrid")
    written: list[str] = []

    counts = frame[target].astype(str).value_counts()
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=axis, color="#3b82f6")
    axis.set(title="Injury target distribution", xlabel=target, ylabel="Athletes/records")
    fig.tight_layout()
    path = output_dir / "injury_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    numeric = frame.select_dtypes(include=np.number)
    numeric.describe().T.to_csv(results_dir / "descriptive_statistics.csv")
    if numeric.shape[1] >= 2:
        fig, axis = plt.subplots(figsize=(max(8, numeric.shape[1] * 0.7), 7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=axis)
        axis.set_title("Numeric feature correlations")
        fig.tight_layout()
        path = output_dir / "numeric_feature_correlations.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    workload = next(
        (column for column in numeric.columns if "workload" in str(column).lower() and column != target),
        None,
    )
    if workload:
        fig, axis = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=frame, x=target, y=workload, ax=axis, color="#93c5fd")
        axis.set_title(f"{workload} by injury outcome")
        fig.tight_layout()
        path = output_dir / "workload_vs_injury.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.columns.difference(numeric).tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.visualizations_dir.mkdir(parents=True, exist_ok=True)

    frame, profile = load_and_profile(args.data, args.target)
    frame = frame.drop_duplicates().dropna(subset=[args.target]).copy()
    profile["rows_after_cleaning"] = len(frame)
    if len(frame) < 10:
        raise ValueError("At least 10 non-duplicate rows with a target are required")

    save_eda(frame, args.target, args.visualizations_dir, args.results_dir)
    y, mapping = prepare_target(frame[args.target], args.positive_label)
    X = frame.drop(columns=[args.target])
    identifier_columns = [
        column
        for column in X
        if X[column].notna().sum()
        and X[column].nunique() == X[column].notna().sum()
        and (
            str(column).lower() in {"id", "athlete_id", "player_id", "record_id"}
            or X[column].dtype == "object"
        )
    ]
    X = X.drop(columns=identifier_columns)
    X, engineered = add_engineered_features(X)
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after removing identifiers")

    profile.update(
        {
            "target_mapping": mapping,
            "identifier_columns_removed": identifier_columns,
            "engineered_features": engineered,
            "model_feature_columns": list(X.columns),
        }
    )
    with (args.results_dir / "data_profile.json").open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, default=json_value)

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError("Each target class needs at least 2 rows for a stratified train/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(
            C=args.models["logistic_regression"]["C"],
            max_iter=args.models["logistic_regression"]["max_iter"],
            class_weight=args.models["logistic_regression"]["class_weight"],
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=args.random_state,
            max_depth=args.models["decision_tree"]["max_depth"],
            min_samples_leaf=args.models["decision_tree"]["min_samples_leaf"],
            min_samples_split=args.models["decision_tree"]["min_samples_split"],
            ccp_alpha=args.models["decision_tree"]["ccp_alpha"],
            class_weight=args.models["decision_tree"]["class_weight"],
        ),
        "Random Forest": RandomForestClassifier(
            random_state=args.random_state,
            n_estimators=args.models["random_forest"]["n_estimators"],
            max_depth=args.models["random_forest"]["max_depth"],
            min_samples_leaf=args.models["random_forest"]["min_samples_leaf"],
            min_samples_split=args.models["random_forest"]["min_samples_split"],
            max_features=args.models["random_forest"]["max_features"],
            class_weight=args.models["random_forest"]["class_weight"],
            n_jobs=args.models["random_forest"]["n_jobs"],
        ),
    }
    with (args.results_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump({"models": args.models}, file, indent=2)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    predictions = pd.DataFrame({"actual": y_test}, index=y_test.index)
    roc_data: list[tuple[str, np.ndarray, np.ndarray, float]] = []

    for name, estimator in models.items():
        model = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else np.nan
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1": f1_score(y_test, predicted, zero_division=0),
                "roc_auc": auc,
            }
        )
        reports[name] = classification_report(y_test, predicted, output_dict=True, zero_division=0)
        predictions[f"{slug(name)}_prediction"] = predicted
        predictions[f"{slug(name)}_probability"] = probabilities
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        display = ConfusionMatrixDisplay(matrix, display_labels=["non-injury (0)", "injury (1)"])
        display.plot(cmap="Blues", colorbar=False)
        display.ax_.set_title(f"{name} confusion matrix")
        plt.tight_layout()
        plt.savefig(args.results_dir / f"confusion_matrix_{slug(name)}.png", dpi=160)
        plt.close()
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        roc_data.append((name, fpr, tpr, auc))

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comparison.to_csv(args.results_dir / "model_comparison.csv", index=False)
    predictions.sort_index().to_csv(args.results_dir / "test_predictions.csv", index_label="source_row")
    with (args.results_dir / "classification_reports.json").open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, default=json_value)

    fig, axis = plt.subplots(figsize=(7, 6))
    for name, fpr, tpr, auc in roc_data:
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Model ROC curves")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.visualizations_dir / "roc_curves.png", dpi=160)
    plt.close(fig)
    print(comparison.to_string(index=False))
    print(f"Saved results to {args.results_dir} and charts to {args.visualizations_dir}")


if __name__ == "__main__":
    main()

