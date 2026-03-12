"""Lateral reliability: PSI + COPRAS aggregation.

Normalization: linear + vector, aggregated (Eq. 6).
Weighting:     PSI (Eqs. 7-11).
Scoring:       COPRAS.

Used for the lateral-reliability comparison in Table 7 ("COPRAS" column).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aroman_core import (
    AromanConfig,
    aggregate_normalizations,
    copras_scores,
    linear_normalization,
    load_decision_matrix,
    psi_weights,
    ranking_frame,
    vector_normalization,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="Path to Excel/CSV decision-matrix file")
    p.add_argument("--negative", type=int, nargs="+", default=[0, 1, 2],
                   help="Zero-based column indices of cost (negative) indicators (default: 0 1 2)")
    p.add_argument("--beta", type=float, default=0.5)
    return p


def main(config: AromanConfig) -> None:
    df = load_decision_matrix(config.data_path)
    matrix = df.to_numpy()
    negative = set(config.negative_indicators)

    norm_linear = linear_normalization(matrix, negative)
    norm_vector = vector_normalization(matrix, negative)
    aggregated = aggregate_normalizations(norm_linear, norm_vector, config.beta)

    # PSI weights on the aggregated normalized matrix (Eqs. 7-8).
    weights = psi_weights(aggregated, negative_indicators=set())
    weighted = aggregated * weights
    scores = copras_scores(weighted, negative)

    rankings = ranking_frame(df.index, scores, "Ui")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = AromanConfig(
        data_path=args.data,
        negative_indicators=tuple(args.negative),
        beta=args.beta,
    )
    main(cfg)
