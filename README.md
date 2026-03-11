# PSI-AROMAN-GMM: A Machine Learning Nested MCDM Framework for Transport Safety Engineering

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.rineng.2025.108543-blue)](https://doi.org/10.1016/j.rineng.2025.108543)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

Open-source implementation accompanying the following peer-reviewed publication:

> Xingjian Zhang, Nanbo (Aaron) Zhang, Jialin Li, Qintao Li, Xingze Liu, Chuanpu (Lukas) Cao, Hao Mao, Ruikang Yan, Yunlong Qi, Xinyi (Chenny) Yang, Jialun Li, Aaron Kaiqiang Zhou, Xu Yan, Hanrui Feng, & Faan Chen. (2026). Machine learning nested MCDM model to enhance decision reliability for transport safety engineering. *Results in Engineering*, 29, 108543. https://doi.org/10.1016/j.rineng.2025.108543

**Paper:** [ScienceDirect (Open Access)](https://www.sciencedirect.com/science/article/pii/S2590123025045876)

## Overview

This repository provides a reproducible implementation of the **PSI-AROMAN-GMM with t-SNE** framework, a hybrid multi-criteria decision-making (MCDM) model designed for transport safety engineering applications. The framework integrates:

- **PSI** (Preference Selection Index; [Maniya & Bhatt, 2010](https://doi.org/10.1016/j.matdes.2009.11.020)) for objective criteria weighting
- **AROMAN** (Alternative Ranking Order Method Accounting for two-step Normalization; [Boskovic et al., 2023]([https://doi.org/10.1109/ACCESS.2023.3332357](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10097712))) for multi-criteria aggregation
- **GMM** (Gaussian Mixture Model) with **t-SNE** (t-distributed Stochastic Neighbor Embedding) for unsupervised clustering

The framework supports ranking, grouping, and robustness analysis of alternatives (e.g., countries) across multiple safety performance indicators.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+.

## Quick Start

### Ranking (proposed method)

```bash
python proposed_aroman_vector_psi.py data_2023.xlsx --negative 0 1 2
```

### Clustering (t-SNE + GMM)

```bash
python grouping_gmm_tsne.py data_2023.xlsx --years 2023 --methods vector minmax max
```

### Correlation analysis

```bash
python correlations.py rankings.xlsx --name table_4
```

## Repository Structure

| File | Paper Reference | Description |
|------|----------------|-------------|
| `aroman_core.py` | Section 4 | Core library: normalization (Eqs. 2-6), weighting (Eqs. 7-11), scoring (Eqs. 13-14) |
| `proposed_aroman_vector_psi.py` | Table 1 | **Proposed method**: linear + vector normalization, PSI weights, AROMAN scores |
| `initial_aroman_max_psi.py` | Table 3 (Max) | Initial sensitivity analysis: max normalization variant |
| `initial_aroman_minmax_psi.py` | Table 3 (MinMax) | Initial sensitivity analysis: min-max normalization variant |
| `medial_aroman_vector_critic.py` | Table 5 (CRITIC) | Medial stability analysis: CRITIC weighting variant |
| `medial_aroman_vector_entropy.py` | Table 5 (Entropy) | Medial stability analysis: entropy weighting variant |
| `lateral_vector_psi_copras.py` | Table 7 (COPRAS) | Lateral reliability analysis: COPRAS aggregation variant |
| `lateral_vector_psi_promethee.py` | Table 7 (PROMETHEE) | Lateral reliability analysis: PROMETHEE II aggregation variant |
| `grouping_gmm_tsne.py` | Tables 2, 9-10, Fig. 3 | t-SNE dimensionality reduction + GMM clustering |
| `grouping_kmeans_cmeans.py` | Tables 11-12 | K-means and fuzzy C-means benchmark clustering |
| `correlations.py` | Tables 4, 6, 8 | Spearman and Pearson correlation analysis |

## Methodology

The framework follows a nine-step pipeline:

1. **Construct** the decision matrix (Eq. 1)
2. **Normalize** using linear (Eqs. 2-3) and vector (Eqs. 4-5) methods
3. **Aggregate** the two normalizations (Eq. 6)
4. **Weight** criteria using PSI (Eqs. 7-11)
5. **Score** alternatives using AROMAN (Eqs. 13-14)
6. **Cluster** using GMM (Eq. 15)
7. **Reduce dimensions** with t-SNE (Eq. 16)
8. **Standardize** features prior to t-SNE
9. **Visualize** and interpret clusters

Robustness is validated through three tiers of sensitivity analysis:

- **Initial sensitivity**: alternative normalization schemes (Table 3)
- **Medial stability**: alternative weighting methods (Table 5)
- **Lateral reliability**: alternative aggregation techniques (Table 7)

## Input Data Format

Prepare your data as an Excel (`.xlsx`) or CSV file:

- **Ranking scripts**: First column = country names/codes (used as index); remaining columns = numeric indicator values.
- **Clustering scripts**: Must include `Country`, `Code`, and `Road_fatalities_per_100_000_inhabitants` columns.

The paper uses indicators A11-A33 (traffic risk), B11-B22 (road user behavior), and C41-C44 (enforcement).

## CLI Reference

All ranking scripts accept the following arguments:

```
positional arguments:
  data                  Path to Excel/CSV decision-matrix file

options:
  --negative N [N ...]  Zero-based column indices of cost indicators (default: 0 1 2)
  --beta BETA           Normalization aggregation weight (default: 0.5)
  --lambda-value LAMBDA AROMAN scoring parameter (default: 0.5)
```

Clustering and correlation scripts have their own arguments; run with `--help` for details.

## Using as a Library

```python
from aroman_core import (
    load_decision_matrix,
    linear_normalization,
    vector_normalization,
    aggregate_normalizations,
    psi_weights,
    aroman_scores,
    ranking_frame,
)

df = load_decision_matrix("data.xlsx")
matrix = df.to_numpy()
negative = {0, 1, 2}

norm_lin = linear_normalization(matrix, negative)
norm_vec = vector_normalization(matrix, negative)
aggregated = aggregate_normalizations(norm_lin, norm_vec, beta=0.5)

weights = psi_weights(aggregated, negative_indicators=set())
weighted = aggregated * weights
scores = aroman_scores(weighted, negative, lambda_value=0.5)

print(ranking_frame(df.index, scores, "R_i"))
```

## Algorithm References

Each algorithm implemented in `aroman_core.py` follows its canonical definition from the original literature:

| Algorithm | Original Reference |
|-----------|--------------------|
| Linear (min-max) normalization | Weitendorf (1976) |
| Max normalization | Stopp (1975) |
| Vector (Euclidean) normalization | [Hwang & Yoon (1981)](https://doi.org/10.1007/978-3-642-48318-9) |
| PSI weighting | [Maniya & Bhatt (2010)](https://doi.org/10.1016/j.matdes.2009.11.020) |
| CRITIC weighting | [Diakoulaki et al. (1995)](https://doi.org/10.1016/0305-0548(94)00059-H) |
| Shannon entropy weighting | [Shannon (1948)](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x); Zeleny (1982) |
| AROMAN scoring | [Boskovic et al. (2023)](https://doi.org/10.1109/ACCESS.2023.3332357) |
| COPRAS scoring | Zavadskas et al. (1994) |
| PROMETHEE II scoring | [Brans et al. (1986)](https://doi.org/10.1016/0377-2217(86)90044-5) |

## Citation

If you use this software in your research, please cite:

```bibtex
@article{zhang2026machine,
  title   = {Machine learning nested {MCDM} model to enhance decision
             reliability for transport safety engineering},
  author  = {Zhang, Xingjian and Zhang, Nanbo (Aaron) and Li, Jialin and
             Li, Qintao and Liu, Xingze and Cao, Chuanpu (Lukas) and
             Mao, Hao and Yan, Ruikang and Qi, Yunlong and
             Yang, Xinyi (Chenny) and Li, Jialun and
             Zhou, Aaron Kaiqiang and Yan, Xu and Feng, Hanrui and
             Chen, Faan},
  journal = {Results in Engineering},
  volume  = {29},
  pages   = {108543},
  year    = {2026},
  publisher = {Elsevier},
  doi     = {10.1016/j.rineng.2025.108543}
}
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license. See [LICENSE](LICENSE) for details.
