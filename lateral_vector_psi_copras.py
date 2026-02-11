from pathlib import Path

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

CONFIG = AromanConfig(data_path=Path("*****.xlsx"), negative_indicators=(0, 1, 2), beta=0.5)


def main() -> None:
    df = load_decision_matrix(CONFIG.data_path)
    matrix = df.to_numpy()
    negative = set(CONFIG.negative_indicators)

    norm_linear = linear_normalization(matrix, negative)
    norm_vector = vector_normalization(matrix, negative)
    aggregated = aggregate_normalizations(norm_linear, norm_vector, CONFIG.beta)

    weights = psi_weights(matrix, negative)
    weighted = aggregated * weights
    scores = copras_scores(weighted, negative)

    rankings = ranking_frame(df.index, scores, "Ui")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
