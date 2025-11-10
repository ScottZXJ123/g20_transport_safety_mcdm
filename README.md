# PSI–AROMAN–GMM Toolkit

This folder contains all executable code referenced in the paper *“Machine learning nested MCDM model to enhance decision reliability for transport safety engineering”*.  
Each script maps one-to-one to the analyses reported in the manuscript (ranking, robustness, and clustering).

## Contents

| Script | Purpose |
| --- | --- |
| `initial_aroman_*.py` | PSI–AROMAN ranking with vector / max / min-max dual normalization (initial-sensitivity checks). |
| `medial_aroman_*.py` | PSI–AROMAN ranking with PSI / CRITIC / Entropy weighting (medial-stability checks). |
| `lateral_vector_psi_*.py` | AROMAN, COPRAS, PROMETHEE aggregators used for lateral reliability. |
| `grouping_gmm_tsne.py` | t-SNE + GMM clustering for every year/normalization combo in Table 9 / Fig. 3. |
| `grouping_kmeans_cmeans.py` | K-means and fuzzy C-means clustering for Table 11 comparisons. |
| `correlations.py` | Spearman/Pearson correlation generator for all ranking comparison tables. |
| `requirements.txt` | Minimal Python dependencies. |

## Usage

1. Create a virtual environment and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Update the file paths/configs inside each script to point at your local data.
3. Run the scripts you need, e.g.:
   ```bash
   python initial_aroman_vector_psi.py
   python grouping_gmm_tsne.py
   python correlations.py
   ```
4. Outputs (rankings, cluster assignments, correlation tables) are written as console logs and/or Excel files in the working directory.

## Notes

- The code assumes the column names shown in the paper (A31…C44 / full indicator names). Keep them consistent when preparing your spreadsheets.