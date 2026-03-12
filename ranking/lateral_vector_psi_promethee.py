"""Lateral reliability: PSI + PROMETHEE II aggregation.

Normalization: vector only.
Weighting:     PSI (Eqs. 7-11).
Scoring:       PROMETHEE II (net flow).

Used for the lateral-reliability comparison in Table 7 ("PROMETHEE" column).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aroman_core import (
    AromanConfig,
    load_decision_matrix,
    promethee_scores,
    psi_weights,
    ranking_frame,
    vector_normalization,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="Path to Excel/CSV decision-matrix file")
    p.add_argument("--negative", type=int, nargs="+", default=[0, 1, 2],
                   help="Zero-based column indices of cost (negative) indicators (default: 0 1 2)")
    return p


def main(config: AromanConfig) -> None:
    df = load_decision_matrix(config.data_path)
    matrix = df.to_numpy()
    negative = set(config.negative_indicators)

    normalized = vector_normalization(matrix, negative)
    # PSI weights on the normalized matrix (Eqs. 7-8).
    weights = psi_weights(normalized, negative_indicators=set())
    scores = promethee_scores(normalized, weights)

    rankings = ranking_frame(df.index, scores, "Net Flow")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = AromanConfig(
        data_path=args.data,
        negative_indicators=tuple(args.negative),
    )
    main(cfg)
