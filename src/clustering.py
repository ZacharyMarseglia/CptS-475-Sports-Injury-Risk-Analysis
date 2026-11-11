"""Clustering pipeline for sports injury datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import add_engineered_features

def log_step(step_name, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[TIMING] {step_name}: {elapsed:.2f} seconds")


def parse_args():
    parser = argparse.ArgumentParser(description="Sports injury clustering pipeline")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("results/clustering"))

    return parser.parse_args()


def load_data(path: Path, target: str):

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)

    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} not found")

    frame = frame.drop_duplicates()
    frame = frame.dropna(subset=[target]).copy()

    if len(frame) < 10:
        raise ValueError("At least 10 rows are required")

    return frame


def prepare_features(frame: pd.DataFrame, target: str):

    injury_target = frame[target].copy()

    X = frame.drop(columns=[target]).copy()

    identifier_columns = [
        column
        for column in X.columns
        if (
            str(column).lower() in {
                "id",
                "athlete_id",
                "player_id",
                "record_id",
            }
            and X[column].nunique(dropna=True) == X[column].notna().sum()
        )
    ]

    X = X.drop(columns=identifier_columns)

    X, engineered = add_engineered_features(X)

    if X.shape[1] == 0:
        raise ValueError("No usable clustering features found")

    return X, injury_target, identifier_columns, engineered


def build_preprocessor(features):

    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = features.columns.difference(numeric).tolist()

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    ),
                ]),
                categorical,
            )
        )

    return ColumnTransformer(transformers=transformers)


def find_optimal_clusters(X, max_clusters, random_state):

    results = []

    cluster_range = range(2, min(max_clusters, len(X) - 1) + 1)

    for k in cluster_range:

        cluster_start = time.perf_counter()

        print(f"[INFO] Evaluating K-Means with k={k}...")

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=5,
            max_iter=300,
            algorithm="lloyd",
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(
            X,
            labels,
            sample_size=min(2000, len(X)),
            random_state=random_state,
        )

        davies_bouldin = davies_bouldin_score(X, labels)

        calinski_harabasz = calinski_harabasz_score(X, labels)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
        })

        log_step(f"K-Means evaluation (k={k})", cluster_start)

    results_df = pd.DataFrame(results)

    return results_df


def save_cluster_visualizations(X, labels, cluster_range, inertias, silhouette_scores, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.plot(cluster_range, inertias, marker="o")

    axis.set(
        title="K-Means Elbow Method",
        xlabel="Number of clusters",
        ylabel="Inertia",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "elbow_plot.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))

    axis.plot(cluster_range, silhouette_scores, marker="o", color="green")

    axis.set(
        title="Silhouette Score by Cluster Count",
        xlabel="Number of clusters",
        ylabel="Silhouette score",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "silhouette_scores.png", dpi=160)
    plt.close(fig)

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(X)

    plot_data = pd.DataFrame({
        "PC1": reduced[:, 0],
        "PC2": reduced[:, 1],
        "Cluster": labels.astype(str),
    })

    fig, axis = plt.subplots(figsize=(9, 7))

    sns.scatterplot(
        data=plot_data,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="tab10",
        s=65,
        alpha=0.8,
        ax=axis,
    )

    variance = pca.explained_variance_ratio_

    axis.set(
        title="Athlete Clusters (PCA)",
        xlabel=f"PC1 ({variance[0]:.1%} variance)",
        ylabel=f"PC2 ({variance[1]:.1%} variance)",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "pca_clusters.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))

    sns.countplot(
        x=labels,
        hue=labels,
        palette="tab10",
        legend=False,
        ax=axis,
    )

    axis.set(
        title="Number of Athletes in Each Cluster",
        xlabel="Cluster",
        ylabel="Number of records",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "cluster_distribution.png", dpi=160)
    plt.close(fig)


def analyze_clusters(original_features, labels, injury_target, output_dir):

    data = original_features.copy()

    data["cluster"] = labels
    data["injury_target"] = injury_target.to_numpy()

    numeric = data.select_dtypes(include=np.number)

    profiles = numeric.groupby("cluster").mean()

    profiles.to_csv(output_dir / "cluster_profiles.csv")

    injury_data = data.copy()

    injury_data["injury_binary"] = (
        injury_data["injury_target"]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes", "injured"])
        .astype(int)
    )

    injury_summary = injury_data.groupby("cluster").agg(
        athlete_count=("injury_binary", "size"),
        injury_count=("injury_binary", "sum"),
        injury_rate=("injury_binary", "mean"),
    )

    injury_summary.to_csv(output_dir / "cluster_injury_summary.csv")

    fig, axis = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=injury_summary.reset_index(),
        x="cluster",
        y="injury_rate",
        color="#3b82f6",
        ax=axis,
    )

    axis.set(
        title="Injury Prevalence Across Clusters",
        xlabel="Cluster",
        ylabel="Injury prevalence",
    )

    axis.set_ylim(0, 1)

    fig.tight_layout()

    fig.savefig(output_dir / "injury_rate_by_cluster.png", dpi=160)

    plt.close(fig)

    return profiles, injury_summary


def plot_cluster_heatmap(original_features, labels, output_dir, top_n=10):

    output_dir.mkdir(parents=True, exist_ok=True)

    numeric = original_features.select_dtypes(include=np.number).copy()

    if numeric.shape[1] == 0:
        return

    numeric["cluster"] = labels

    cluster_means = numeric.groupby("cluster").mean()

    overall_mean = original_features.select_dtypes(include=np.number).mean()

    overall_std = original_features.select_dtypes(include=np.number).std().replace(0, np.nan)

    standardized = (cluster_means - overall_mean) / overall_std

    standardized = standardized.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    if standardized.empty:
        return

    feature_scores = standardized.var(axis=0)

    selected_features = (feature_scores.sort_values(ascending=False).head(top_n).index)

    heatmap_data = standardized[selected_features]

    heatmap_data = heatmap_data[feature_scores[selected_features].sort_values(ascending=False).index]

    feature_scores.sort_values(
        ascending=False
    ).to_csv(
        output_dir / "cluster_feature_relevance.csv",
        header=["relevance_score"],
    )

    fig, axis = plt.subplots(figsize=(10, max(6, len(heatmap_data.columns) * 0.5)))

    sns.heatmap(
        heatmap_data.T,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Standard deviations from overall mean"},
        ax=axis,
    )

    axis.set_title(
        f"Top {len(selected_features)} Most Distinguishing Cluster Features",
        fontsize=14,
        fontweight="bold",
    )

    axis.set_xlabel("Cluster")
    axis.set_ylabel("Features")

    fig.tight_layout()
    fig.savefig(output_dir / "cluster_feature_heatmap.png", dpi=160)

    plt.close(fig)

    print(f"[INFO] Heatmap generated using {len(selected_features)} features.")


def main():

    pipeline_start = time.perf_counter()

    args = parse_args()

    if args.max_clusters < 2:
        raise ValueError("--max-clusters must be at least 2")
    
    if args.k is not None:
        if args.k < 2:
            raise ValueError("--k must be at least 2")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    step_start = time.perf_counter()

    print("[INFO] Loading dataset...")

    frame = load_data(args.data, args.target)

    print(f"[INFO] Dataset shape: {frame.shape}")

    log_step("Data loading", step_start)

    # Feature engineering
    step_start = time.perf_counter()

    print("\n[INFO] Preparing features...")

    X, injury_target, identifiers, engineered = (prepare_features(frame, args.target))

    print(f"[INFO] Features available: {X.shape[1]}")
    print(f"[INFO] Identifier columns removed: {identifiers}")
    print(f"[INFO] Engineered features: {engineered}")

    log_step("Feature preparation and engineering", step_start)

    # Build preprocessing pipeline
    step_start = time.perf_counter()

    print("\n[INFO] Building preprocessing pipeline...")

    preprocessor = build_preprocessor(X)

    log_step("Preprocessor construction", step_start)

    # Transform features
    step_start = time.perf_counter()

    print("\n[INFO] Applying preprocessing...")

    transformed = preprocessor.fit_transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessed data contains non-finite values")

    if transformed.shape[1] < 2:
        raise ValueError("At least two encoded features are required")

    print(f"[INFO] Transformed dataset shape: {transformed.shape}")

    log_step("Feature preprocessing and transformation", step_start)

    
    # Find or manually set optimal cluster count
    step_start = time.perf_counter()

    if args.k is not None:

        best_k = args.k

        if best_k >= len(transformed):
            raise ValueError("--k must be smaller than the number of samples")

        print(f"\n[INFO] Using manually specified k={best_k}")

    else:
        print("\n[INFO] Searching for optimal number of clusters...")

        results_df = find_optimal_clusters(transformed, args.max_clusters, args.random_state)

        best_k = int(results_df.loc[results_df["silhouette"].idxmax(), "k"])

        print("\n[INFO] K selection results:")
        print(results_df.to_string(index=False))

        print(f"\n[INFO] Automatically selected number of clusters: {best_k}")

        results_df.to_csv(args.results_dir / "k_selection_results.csv", index=False)

    log_step("Cluster count selection", step_start)

    # Fit final K-Means model
    step_start = time.perf_counter()

    print("\n[INFO] Training final K-Means model...")

    model = KMeans(
        n_clusters=best_k,
        random_state=args.random_state,
        n_init=10,
    )

    labels = model.fit_predict(transformed)

    log_step("Final K-Means model training", step_start)

    # Save cluster assignments
    step_start = time.perf_counter()

    print("\n[INFO] Saving cluster assignments...")

    assignments = pd.DataFrame({
        "source_row": frame.index,
        "cluster": labels,
        "injury_target": injury_target.to_numpy(),
    })

    assignments.to_csv(args.results_dir / "cluster_assignments.csv", index=False)

    log_step("Saving cluster assignments", step_start)

    # Generate visualizations
    step_start = time.perf_counter()

    print("\n[INFO] Generating cluster visualizations...")

    if args.k is None:
        cluster_range = results_df["k"].tolist()
        inertias = results_df["inertia"].tolist()
        silhouette_scores = results_df["silhouette"].tolist()

    else:
        cluster_range = []
        inertias = []
        silhouette_scores = [] 

    save_cluster_visualizations(
        transformed,
        labels,
        cluster_range,
        inertias,
        silhouette_scores,
        args.results_dir / "cluster_visualizations",
    )

    plot_cluster_heatmap(X, labels, args.results_dir / "cluster_visualizations")

    log_step("Generating visualizations", step_start)

    # Analyze cluster characteristics
    step_start = time.perf_counter()

    print("\n[INFO] Analyzing cluster characteristics...")

    profiles, injury_summary = analyze_clusters(X, labels, injury_target, args.results_dir)

    log_step("Cluster profile and injury analysis", step_start)

    # Display final results
    print("\nFinal Results")

    print("\nCluster profiles:")
    print(profiles)

    print("\nInjury prevalence by cluster:")
    print(injury_summary)

    print(f"\nResults saved to {args.results_dir}")

    # Total execution time
    total_time = time.perf_counter() - pipeline_start

    print(f"[TIMING] Total execution time: {total_time:.2f} seconds / {total_time / 60:.2f} minutes")

if __name__ == "__main__":
    main()