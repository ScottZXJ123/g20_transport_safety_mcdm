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

CONFIG = AromanConfig(data_path=Path("*****.xlsx"), negative_indicators=(0, 1, 2), beta=0.5, lambda_value=0.5)


def main() -> None:
    df = load_decision_matrix(CONFIG.data_path)
    matrix = df.to_numpy()
    negative = set(CONFIG.negative_indicators)

    norm_linear = linear_normalization(matrix, negative)
    norm_vector = vector_normalization(matrix, negative)
    aggregated = aggregate_normalizations(norm_linear, norm_vector, CONFIG.beta)

    weights = critic_weights(matrix, negative)
    weighted = aggregated * weights
    scores = aroman_scores(weighted, negative, CONFIG.lambda_value)

    rankings = ranking_frame(df.index, scores, "R_i")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
