from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
import skfuzzy as fuzz


@dataclass
class YearConfig:
    year: int
    path: Path
    columns: Sequence[str]


YEAR_CONFIG: Iterable[YearConfig] = [
    
]


OUTPUT = Path("cluster_kmeans_fcm.xlsx")
METRIC = "Road_fatalities_per_100_000_inhabitants"


def load_matrix(config: YearConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(config.path)
    df = pd.read_excel(config.path) if config.path.suffix.lower() in {".xls", ".xlsx"} else pd.read_csv(config.path)
    missing = {"Country", "Code", METRIC}.difference(df.columns)
    if missing:
        raise ValueError(f"{config.path} missing columns: {missing}")
    subset = df[["Country", "Code", *config.columns]].dropna()
    subset = subset.set_index("Code")
    return subset


def reorder_labels(df: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    temp = pd.DataFrame({"label": labels, METRIC: df[METRIC]})
    means = temp.groupby("label")[METRIC].mean().sort_values()
    mapping = {old: rank for rank, old in enumerate(means.index, start=1)}
    return np.array([mapping[z] for z in labels])


def cluster_year(config: YearConfig) -> pd.DataFrame:
    df = load_matrix(config)
    features = df[config.columns].drop(columns=["Country"], errors="ignore")
    scaler = Normalizer()
    X_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=4, random_state=42)
    km_labels = reorder_labels(df, km.fit_predict(X_scaled))
    df["Cluster_KMeans"] = km_labels

    data_for_fcm = X_scaled.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_for_fcm,
        c=4,
        m=2,
        error=0.005,
        maxiter=1000,
        seed=42,
    )
    fcm_labels = reorder_labels(df, np.argmax(u, axis=0))
    df["Cluster_FCM"] = fcm_labels

    return df[["Country", "Cluster_KMeans", "Cluster_FCM"]]


def main() -> None:
    if not YEAR_CONFIG:
        raise RuntimeError("YEAR_CONFIG is empty. Add your input files first.")
    results = []
    for config in YEAR_CONFIG:
        result = cluster_year(config)
        result.insert(0, "Year", config.year)
        results.append(result.reset_index())
        print(f"\nClusters for {config.year}:\n{result}")

    combined = pd.concat(results, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(OUTPUT, index=False)
    print(f"\nSaved {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()