from pathlib import Path

from aroman_core import (
    AromanConfig,
    load_decision_matrix,
    promethee_scores,
    psi_weights,
    ranking_frame,
    vector_normalization,
)

CONFIG = AromanConfig(data_path=Path("*****.xlsx"), negative_indicators=(0, 1, 2))


def main() -> None:
    df = load_decision_matrix(CONFIG.data_path)
    matrix = df.to_numpy()
    negative = set(CONFIG.negative_indicators)

    normalized = vector_normalization(matrix, negative)
    weights = psi_weights(matrix, negative)
    scores = promethee_scores(normalized, weights)

    rankings = ranking_frame(df.index, scores, "Net Flow")
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
