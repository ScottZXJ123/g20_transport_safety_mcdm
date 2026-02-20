from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer


@dataclass
class YearConfig:
    year: int
    path: Path
    feature_columns: Sequence[str]


YEAR_CONFIG: Iterable[YearConfig] = []

OUTPUT = Path("cluster_kmeans_fcm.xlsx")
METRIC = "Road_fatalities_per_100_000_inhabitants"


def load_matrix(config: YearConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(config.path)

    if config.path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(config.path)
    else:
        df = pd.read_csv(config.path)

    required = {"Country", "Code", METRIC, *config.feature_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{config.path} missing columns: {sorted(missing)}")

    subset = df[["Country", "Code", METRIC, *config.feature_columns]].dropna().set_index("Code")
    return subset


def reorder_labels(metric: pd.Series, labels: np.ndarray) -> np.ndarray:
    temp = pd.DataFrame({"label": labels, METRIC: metric.to_numpy()})
    means = temp.groupby("label", as_index=True)[METRIC].mean().sort_values()
    mapping = {old: rank for rank, old in enumerate(means.index, start=1)}
    return np.array([mapping[label] for label in labels])


def cluster_year(config: YearConfig) -> pd.DataFrame:
    df = load_matrix(config)
    features = df[list(config.feature_columns)]

    scaler = Normalizer()
    x_scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=4, random_state=42, n_init="auto")
    km_labels = reorder_labels(df[METRIC], km.fit_predict(x_scaled))

    _, memberships, *_ = fuzz.cluster.cmeans(
        x_scaled.T,
        c=4,
        m=2,
        error=0.005,
        maxiter=1000,
        seed=42,
    )
    fcm_labels = reorder_labels(df[METRIC], np.argmax(memberships, axis=0))

    return pd.DataFrame(
        {
            "Country": df["Country"],
            "Cluster_KMeans": km_labels,
            "Cluster_FCM": fcm_labels,
        },
        index=df.index,
    )


def main() -> None:
    if not YEAR_CONFIG:
        raise RuntimeError("YEAR_CONFIG is empty. Add your input files first.")

    outputs: list[pd.DataFrame] = []
    for config in YEAR_CONFIG:
        clusters = cluster_year(config)
        with_year = clusters.reset_index().assign(Year=config.year)
        outputs.append(with_year[["Year", "Code", "Country", "Cluster_KMeans", "Cluster_FCM"]])
        print(f"\nClusters for {config.year}:\n{clusters}")

    combined = pd.concat(outputs, ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(OUTPUT, index=False)
    print(f"\nSaved {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
