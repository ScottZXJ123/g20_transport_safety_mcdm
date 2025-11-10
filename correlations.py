from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class CorrelationTask:
    name: str
    path: Path
    sheet: Optional[str] = None
    usecols: Optional[str] = None


CORRELATION_TASKS: Iterable[CorrelationTask] = [
    # Examples – replace with your actual files
]


OUTPUT_DIR = Path("correlation_outputs")


def load_rank_table(task: CorrelationTask) -> pd.DataFrame:
    if not task.path.exists():
        raise FileNotFoundError(task.path)
    if task.path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        df = pd.read_excel(task.path, sheet_name=task.sheet, usecols=task.usecols)
    else:
        df = pd.read_csv(task.path, usecols=None if task.usecols is None else task.usecols.split(","))
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"{task.path} produced an empty table.")
    return df.reset_index(drop=True)


def pairwise_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            s, _ = spearmanr(df[c1], df[c2])
            p, _ = pearsonr(df[c1], df[c2])
            rows.append({"Comparison": f"{c1} vs {c2}", "Spearman": s, "Pearson": p})
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, task: CorrelationTask) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{task.name}_correlations.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved {out_path}")


def run(tasks: Iterable[CorrelationTask]) -> None:
    if not tasks:
        raise RuntimeError("CORRELATION_TASKS is empty. Add your ranking files first.")
    for task in tasks:
        table = load_rank_table(task)
        results = pairwise_correlations(table)
        save_results(results, task)


if __name__ == "__main__":
    run(CORRELATION_TASKS)