from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer


@dataclass
class YearConfig:
    year: int
    path: Path
    normalizations: Sequence[str]  # e.g., ("vector", "minmax", "max")


YEAR_CONFIG: Iterable[YearConfig] = []

NEGATIVE_INDICATORS = [
    "Road_fatalities_per_100_000_inhabitants",
    "Road_fatalities_per_10_000_registered_vehicles",
    "Change_in_number_of_road_deaths",
]

OUTPUT_FILE = Path("cluster_assignments.xlsx")

SCALERS = {
    "vector": Normalizer(),
    "minmax": MinMaxScaler(),
    "max": MaxAbsScaler(),
}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_features(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if "Country" not in df.columns:
        raise ValueError("Input must include a 'Country' column.")
    features = df.drop(columns=["Country"])
    scaled = SCALERS[method].fit_transform(features)
    return pd.DataFrame(scaled, index=df.index, columns=features.columns)


def reorder_clusters(df: pd.DataFrame, raw_labels) -> pd.Series:
    metric = "Road_fatalities_per_100_000_inhabitants"
    tmp = pd.DataFrame({"cluster": raw_labels, "metric": df[metric]})
    means = tmp.groupby("cluster", as_index=True)["metric"].mean().sort_values()
    mapping = {old: rank for rank, old in enumerate(means.index, start=1)}
    return pd.Series([mapping[label] for label in raw_labels], index=df.index)


def cluster_single_config(year: int, df: pd.DataFrame, method: str) -> pd.Series:
    x_scaled = normalize_features(df, method)
    perplexity = max(2, min(5, len(x_scaled) - 1))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(x_scaled)
    labels = GaussianMixture(n_components=4, random_state=42).fit_predict(coords)

    ordered = reorder_clusters(df, labels)
    ordered.name = f"{year}_{method}"
    return ordered


def run() -> None:
    results = []

    for config in YEAR_CONFIG:
        df = load_dataset(config.path).copy().set_index("Code")
        required = {"Country", *NEGATIVE_INDICATORS}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{config.path} missing columns: {sorted(missing)}")

        for method in config.normalizations:
            if method not in SCALERS:
                raise ValueError(f"Unsupported normalization '{method}'")
            results.append(cluster_single_config(config.year, df, method))

    if not results:
        raise RuntimeError("YEAR_CONFIG is empty—add your datasets first.")

    assignments = pd.concat(results, axis=1)
    assignments.to_excel(OUTPUT_FILE)
    print("Cluster assignments:\n", assignments)
    print(f"\nSaved {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    run()
