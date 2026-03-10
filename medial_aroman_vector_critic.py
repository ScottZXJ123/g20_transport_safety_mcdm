"""Medial stability: AROMAN with CRITIC weighting.

Normalization: linear + vector, aggregated (Eq. 6).
Weighting:     CRITIC (intercriteria correlation-based).
Scoring:       AROMAN (Eqs. 13-14).

Used for the medial-stability comparison in Table 5 ("CRITIC" column).
"""
import argparse
from pathlib import Path

from aroman_core import (
    AromanConfig,
    aggregate_normalizations,
    aroman_scores,
    critic_weights,
    linear_normalization,
    load_decision_matrix,
    ranking_frame,
    vector_normalization,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help="Path to Excel/CSV decision-matrix file")
    p.add_argument("--negative", type=int, nargs="+", default=[0, 1, 2],
                   help="Zero-based column indices of cost (negative) indicators (default: 0 1 2)")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lambda-value", type=float, default=0.5)
    return p


def main(config: AromanConfig) -> None:
    df = load_decision_matrix(config.data_path)
    matrix = df.to_numpy()
    negative = set(config.negative_indicators)

    norm_linear = linear_normalization(matrix, negative)
    norm_vector = vector_normalization(matrix, negative)
    aggregated = aggregate_normalizations(norm_linear, norm_vector, config.beta)

    # CRITIC performs its own internal normalization on raw values.
    weights = critic_weights(matrix, negative)
    weighted = aggregated * weights
    scores = aroman_scores(weighted, negative, config.lambda_value)

    rankings = ranking_frame(df.index, scores, "R_i")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = AromanConfig(
        data_path=args.data,
        negative_indicators=tuple(args.negative),
        beta=args.beta,
        lambda_value=args.lambda_value,
    )
    main(cfg)
