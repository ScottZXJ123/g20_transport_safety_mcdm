"""K-means and fuzzy C-means clustering (benchmark methods).

Used for the lateral-reliability grouping comparison in Tables 11-12.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

METRIC = "Road_fatalities_per_100_000_inhabitants"


@dataclass
class YearConfig:
    """Configuration for one year of clustering data."""
    year: int
    path: Path
    feature_columns: Sequence[str]


def load_matrix(config: YearConfig) -> pd.DataFrame:
    if not config.path.exists():
        raise FileNotFoundError(config.path)
    if config.path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(config.path)
    else:
        df = pd.read_csv(config.path)

    required = {"Country", "Code", METRIC, *config.feature_columns}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{config.path} missing columns: {sorted(missing)}")
    return df[["Country", "Code", METRIC, *config.feature_columns]].dropna().set_index("Code")


def reorder_labels(metric: pd.Series, labels: np.ndarray) -> np.ndarray:
    """Re-label clusters 1..K ordered by ascending mean fatality rate."""
    temp = pd.DataFrame({"label": labels, METRIC: metric.to_numpy()})
    means = temp.groupby("label", as_index=True)[METRIC].mean().sort_values()
    mapping = {old: rank for rank, old in enumerate(means.index, start=1)}
    return np.array([mapping[label] for label in labels])


def cluster_year(config: YearConfig) -> pd.DataFrame:
    df = load_matrix(config)
    features = df[list(config.feature_columns)]
    x_scaled = Normalizer().fit_transform(features)

    km = KMeans(n_clusters=4, random_state=42, n_init="auto")
    km_labels = reorder_labels(df[METRIC], km.fit_predict(x_scaled))

    _, memberships, *_ = fuzz.cluster.cmeans(
        x_scaled.T, c=4, m=2, error=0.005, maxiter=1000, seed=42,
    )
    fcm_labels = reorder_labels(df[METRIC], np.argmax(memberships, axis=0))

    return pd.DataFrame(
        {"Country": df["Country"], "Cluster_KMeans": km_labels, "Cluster_FCM": fcm_labels},
        index=df.index,
    )


def run(configs: list[YearConfig], output: Path) -> None:
    if not configs:
        raise RuntimeError("No year configurations provided.")

    outputs: list[pd.DataFrame] = []
    for config in configs:
        clusters = cluster_year(config)
        with_year = clusters.reset_index().assign(Year=config.year)
        outputs.append(with_year[["Year", "Code", "Country", "Cluster_KMeans", "Cluster_FCM"]])
        print(f"\nClusters for {config.year}:\n{clusters}")

    combined = pd.concat(outputs, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_excel(output, index=False)
    print(f"\nSaved {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="+",
                   help="One or more Excel/CSV files (one per year)")
    p.add_argument("--years", type=int, nargs="+",
                   help="Year label for each data file")
    p.add_argument("--features", nargs="+", required=True,
                   help="Column names to use as clustering features")
    p.add_argument("-o", "--output", type=Path, default=Path("cluster_kmeans_fcm.xlsx"))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    years = args.years or list(range(len(args.data)))
    if len(years) != len(args.data):
        raise ValueError("Number of --years must match number of data files")
    cfgs = [YearConfig(year=y, path=p, feature_columns=args.features) for y, p in zip(years, args.data)]
    run(cfgs, args.output)
