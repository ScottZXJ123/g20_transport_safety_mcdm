from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class YearConfig:
    year: int
    path: Path
    normalizations: Sequence[str]  # e.g., ("vector", "minmax", "max")

YEAR_CONFIG: Iterable[YearConfig] = [
    # Fill these with your actual files
    # YearConfig(2023, Path("data/road_safety_2023.xlsx"), ("vector", "minmax", "max")),
]

NEGATIVE_INDICATORS = [
    "Road_fatalities_per_100_000_inhabitants",
    "Road_fatalities_per_10_000_registered_vehicles",
    "Change_in_number_of_road_deaths",
]

OUTPUT_FILE = Path("cluster_assignments.xlsx")

# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

SCALERS = {
    "vector": Normalizer(),
    "minmax": MinMaxScaler(),
    "max": MaxAbsScaler(),
}

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df

def normalize_features(df: pd.DataFrame, method: str) -> pd.DataFrame:
    scaler = SCALERS[method]
    features = df.drop(columns=["Country"])
    scaled = scaler.fit_transform(features)
    return pd.DataFrame(scaled, index=df.index, columns=features.columns)

def reorder_clusters(df: pd.DataFrame, raw_labels) -> pd.Series:
    perf = df["Road_fatalities_per_100_000_inhabitants"]
    tmp = pd.DataFrame({"cluster": raw_labels, "perf": perf})
    means = tmp.groupby("cluster")["perf"].mean()
    ranking = means.sort_values().index.tolist()
    mapping = {raw: rank for rank, raw in enumerate(ranking, start=1)}
    return pd.Series([mapping[label] for label in raw_labels], index=df.index)

def cluster_single_config(year: int, df: pd.DataFrame, method: str) -> pd.Series:
    X_scaled = normalize_features(df, method)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    coords = tsne.fit_transform(X_scaled)
    gmm = GaussianMixture(n_components=4, random_state=42)
    labels = gmm.fit_predict(coords)
    ordered = reorder_clusters(df, labels)
    ordered.name = f"{year}_{method}"
    return ordered

def run():
    results = []

    for config in YEAR_CONFIG:
        df = load_dataset(config.path).copy()
        df = df.set_index("Code")
        for col in NEGATIVE_INDICATORS + ["Country"]:
            if col not in df.columns and col != "Country":
                raise ValueError(f"Column '{col}' missing in {config.path}")

        for method in config.normalizations:
            if method not in SCALERS:
                raise ValueError(f"Unsupported normalization '{method}'")
            series = cluster_single_config(config.year, df, method)
            results.append(series)

    if not results:
        raise RuntimeError("YEAR_CONFIG is empty—add your datasets first.")

    assignments = pd.concat(results, axis=1)
    assignments.to_excel(OUTPUT_FILE)
    print("Cluster assignments:\n", assignments)
    print(f"\nSaved {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    run()