"""t-SNE + GMM clustering (Steps 6-9).

Applies Gaussian Mixture Model clustering after t-SNE dimensionality
reduction, for each year and normalization method combination.

Used for Tables 2, 9-10 and Figure 3 in the paper.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer

REORDER_METRIC = "Road_fatalities_per_100_000_inhabitants"

SCALERS = {
    "vector": Normalizer,
    "minmax": MinMaxScaler,
    "max": MaxAbsScaler,
}


@dataclass
class YearConfig:
    """Configuration for one year of clustering data."""
    year: int
    path: Path
    normalizations: Sequence[str]  # e.g., ("vector", "minmax", "max")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset that must contain 'Code' and 'Country' columns."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_features(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Normalize numeric features using *method* ('vector', 'minmax', or 'max')."""
    if "Country" not in df.columns:
        raise ValueError("Input must include a 'Country' column.")
    features = df.drop(columns=["Country"])
    scaler = SCALERS[method]()
    scaled = scaler.fit_transform(features)
    return pd.DataFrame(scaled, index=df.index, columns=features.columns)


def reorder_clusters(df: pd.DataFrame, raw_labels) -> pd.Series:
    """Re-label clusters 1..K ordered by ascending mean fatality rate."""
    tmp = pd.DataFrame({"cluster": raw_labels, "metric": df[REORDER_METRIC]})
    means = tmp.groupby("cluster", as_index=True)["metric"].mean().sort_values()
    mapping = {old: rank for rank, old in enumerate(means.index, start=1)}
    return pd.Series([mapping[label] for label in raw_labels], index=df.index)


def cluster_single_config(year: int, df: pd.DataFrame, method: str,
                          n_components: int = 4) -> pd.Series:
    """Run t-SNE + GMM for one year/normalization combination."""
    x_scaled = normalize_features(df, method)
    perplexity = max(2, min(5, len(x_scaled) - 1))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(x_scaled)
    labels = GaussianMixture(n_components=n_components, random_state=42).fit_predict(coords)

    ordered = reorder_clusters(df, labels)
    ordered.name = f"{year}_{method}"
    return ordered


def run(configs: list[YearConfig], output: Path) -> None:
    """Execute clustering for all year/normalization combos and save results."""
    if not configs:
        raise RuntimeError("No year configurations provided.")

    results = []
    for config in configs:
        df = load_dataset(config.path).copy().set_index("Code")
        required = {"Country", REORDER_METRIC}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{config.path} missing columns: {sorted(missing)}")

        for method in config.normalizations:
            if method not in SCALERS:
                raise ValueError(f"Unsupported normalization '{method}'. Choose from: {list(SCALERS)}")
            results.append(cluster_single_config(config.year, df, method))

    assignments = pd.concat(results, axis=1)
    assignments.to_excel(output)
    print("Cluster assignments:\n", assignments)
    print(f"\nSaved {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="+",
                   help="One or more Excel/CSV files (one per year)")
    p.add_argument("--years", type=int, nargs="+",
                   help="Year label for each data file (must match count of data args)")
    p.add_argument("--methods", nargs="+", default=["vector", "minmax", "max"],
                   choices=list(SCALERS), help="Normalization methods (default: all three)")
    p.add_argument("-o", "--output", type=Path, default=Path("cluster_assignments.xlsx"),
                   help="Output Excel file (default: cluster_assignments.xlsx)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    years = args.years or list(range(len(args.data)))
    if len(years) != len(args.data):
        raise ValueError("Number of --years must match number of data files")
    cfgs = [YearConfig(year=y, path=p, normalizations=args.methods) for y, p in zip(years, args.data)]
    run(cfgs, args.output)
