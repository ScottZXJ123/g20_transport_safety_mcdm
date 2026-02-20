from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr


@dataclass
class CorrelationTask:
    name: str
    path: Path
    sheet: str | None = None
    usecols: str | None = None


CORRELATION_TASKS: list[CorrelationTask] = [
    # Example:
    # CorrelationTask(name="table_5", path=Path("rankings.xlsx"), sheet="Sheet1", usecols="A:D"),
]

OUTPUT_DIR = Path("correlation_outputs")


def load_rank_table(task: CorrelationTask) -> pd.DataFrame:
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
    rows: list[dict[str, float | str]] = []
    cols = df.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            s, _ = spearmanr(df[c1], df[c2])
            p, _ = pearsonr(df[c1], df[c2])
            rows.append({"Comparison": f"{c1} vs {c2}", "Spearman": float(s), "Pearson": float(p)})
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, task: CorrelationTask) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{task.name}_correlations.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Saved {out_path}")


def run(tasks: list[CorrelationTask]) -> None:
    if not tasks:
        raise RuntimeError("CORRELATION_TASKS is empty. Add your ranking files first.")
    for task in tasks:
        table = load_rank_table(task)
        results = pairwise_correlations(table)
        save_results(results, task)


if __name__ == "__main__":
    run(CORRELATION_TASKS)
