"""Spearman and Pearson correlation generator for ranking comparisons.

Used for Tables 4, 6, 8 (ranking correlations) and Tables 10, 12
(V-measure grouping similarity) in the paper.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class CorrelationTask:
    """One ranking-comparison task."""
    name: str
    path: Path
    sheet: str | None = None
    usecols: str | None = None


def load_rank_table(task: CorrelationTask) -> pd.DataFrame:
    """Load a ranking table from Excel or CSV."""
    if not task.path.exists():
        raise FileNotFoundError(task.path)

    if task.path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(task.path, sheet_name=task.sheet, usecols=task.usecols)
    else:
        csv_usecols = None if task.usecols is None else [x.strip() for x in task.usecols.split(",")]
        df = pd.read_csv(task.path, usecols=csv_usecols)

    cleaned = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if cleaned.empty:
        raise ValueError(f"{task.path} produced an empty table.")
    if cleaned.shape[1] < 2:
        raise ValueError(f"{task.path} must contain at least two ranking columns.")
    return cleaned.reset_index(drop=True)


def pairwise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Spearman and Pearson correlations for all column pairs."""
    rows: list[dict[str, float | str]] = []
    cols = df.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            s, _ = spearmanr(df[c1], df[c2])
            p, _ = pearsonr(df[c1], df[c2])
            rows.append({"Comparison": f"{c1} vs {c2}", "Spearman": round(float(s), 4), "Pearson": round(float(p), 4)})
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, task: CorrelationTask, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task.name}_correlations.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved {out_path}")


def run(tasks: list[CorrelationTask], output_dir: Path) -> None:
    if not tasks:
        raise RuntimeError("No correlation tasks provided.")
    for task in tasks:
        table = load_rank_table(task)
        results = pairwise_correlations(table)
        save_results(results, task, output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path,
                   help="Excel/CSV file containing ranking columns")
    p.add_argument("--name", default="rankings",
                   help="Label used in the output filename (default: rankings)")
    p.add_argument("--sheet", default=None,
                   help="Excel sheet name (default: first sheet)")
    p.add_argument("--usecols", default=None,
                   help="Column range, e.g. 'A:D' for Excel or comma-separated names for CSV")
    p.add_argument("-o", "--output-dir", type=Path, default=Path("correlation_outputs"),
                   help="Output directory (default: correlation_outputs/)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    task = CorrelationTask(name=args.name, path=args.data, sheet=args.sheet, usecols=args.usecols)
    run([task], args.output_dir)
