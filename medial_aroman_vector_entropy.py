from pathlib import Path

import numpy as np

from aroman_core import (
    AromanConfig,
    aggregate_normalizations,
    aroman_scores,
    entropy_weights,
    linear_normalization,
    load_decision_matrix,
    ranking_frame,
    vector_normalization,
)

CONFIG = AromanConfig(data_path=Path("*****.xlsx"), negative_indicators=(0, 1, 2), beta=0.5, lambda_value=0.5)


def main() -> None:
    df = load_decision_matrix(CONFIG.data_path)
    matrix = df.to_numpy()
    negative = set(CONFIG.negative_indicators)

    norm_linear = linear_normalization(matrix, negative)
    norm_vector = vector_normalization(matrix, negative)
    aggregated = aggregate_normalizations(norm_linear, norm_vector, CONFIG.beta)

    # Entropy weights are computed on non-negative values. Shift if required.
    matrix_for_entropy = matrix.copy()
    min_by_col = matrix_for_entropy.min(axis=0)
    matrix_for_entropy -= np.minimum(min_by_col, 0)
    weights = entropy_weights(matrix_for_entropy)

    weighted = aggregated * weights
    scores = aroman_scores(weighted, negative, CONFIG.lambda_value)

    rankings = ranking_frame(df.index, scores, "R_i")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
