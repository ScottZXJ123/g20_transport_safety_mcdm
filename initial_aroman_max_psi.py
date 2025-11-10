import numpy as np
import pandas as pd

# Load the Excel file
filename = '2015.xlsx'
df = pd.read_excel(filename, index_col=0)

# Decision matrix and criteria names
decision_matrix = df.values
criteria_names = df.columns[:]

# Number of alternatives (m) and criteria (n)
m, n = decision_matrix.shape

# Indices of negative (cost) indicators
negative_indicators = [0, 1, 2]

# Step 2.1 Normalization 1 (Linear)
normalized_decision_matrix_1 = np.zeros_like(decision_matrix, dtype=float)
for j in range(n):
    if j in negative_indicators:
        normalized_decision_matrix_1[:, j] = (np.max(decision_matrix[:, j]) - decision_matrix[:, j]) / (np.max(decision_matrix[:, j]) - np.min(decision_matrix[:, j]))
    else:
        normalized_decision_matrix_1[:, j] = (decision_matrix[:, j] - np.min(decision_matrix[:, j])) / (np.max(decision_matrix[:, j]) - np.min(decision_matrix[:, j]))

# Step 2.2 Normalization 2 (Max)
normalized_decision_matrix_2 = np.zeros_like(decision_matrix)
for j in range(n):
    if j in negative_indicators:
        normalized_decision_matrix_2[:, j] = 1 - decision_matrix[:, j] / np.max(decision_matrix[:, j])
    else:
        normalized_decision_matrix_2[:, j] = decision_matrix[:, j] / np.max(decision_matrix[:, j])
        
# Step 2.3 Aggregated averaged normalization
beta = 0.5 # Weighing factor
aggregated_averaged_normalization = (beta * normalized_decision_matrix_1 + (1 - beta) * normalized_decision_matrix_2) / 2

# Calculate weights using the Preference Selection Index (PSI) method
data = df.to_numpy()
m, n = data.shape

# Calculate the preference variation value (PVj)
PVj = np.zeros((m, n))  # Create a 2D array to store the element-wise division results
for j in range(n):
    if j in negative_indicators:
        x_min = np.min(data[:, j])
        PVj[:, j] = x_min / data[:, j]  # Perform element-wise division and store the result in PVj[:, j]
    else:
        x_max = np.max(data[:, j])
        PVj[:, j] = data[:, j] / x_max

# Calculate the deviation of the preference value (DPVj)
DPVj = np.mean(np.abs(PVj - np.mean(PVj, axis=0)), axis=0)

# Calculate the weights wj using the Preference Selection Index (PSIj)
PSIj = DPVj / np.sum(DPVj)
weights = PSIj / np.sum(PSIj)

# Step 3. Multiply the Aggregated Averaged Normalized decision-making matrix with the criteria weights to obtain a weighted DM matrix
weighted_dm_matrix = aggregated_averaged_normalization * weights

# Step 4. Separately summarize the normalized weighted values of the criteria type min (L_i) and the normalized weighted values of the max type (A_i)
L_i = np.sum(weighted_dm_matrix[:, negative_indicators], axis=1)
A_i = np.sum(weighted_dm_matrix[:, ~np.isin(range(n), negative_indicators)], axis=1)

# Step 5. Calculate the final ranking of the alternatives
lambda_value = 0.5 # Coefficient degree of the criterion type
R_i = L_i + lambda_value * A_i

# Rank the alternatives based on R_i values
ranking_order = R_i.argsort()[::-1]
ranked_countries = df.index[ranking_order]
rankings = np.arange(1, len(df) + 1)

# Create a DataFrame for the rankings
rankings_df = pd.DataFrame({
    'Rank': rankings,
    'Country': ranked_countries,
    'R_i': R_i[ranking_order]
})

# Reset index to start from 1 and show as the first column
rankings_df.index = np.arange(1, m + 1)
rankings_str = rankings_df.to_string(index=False)

# Output the rankings DataFrame
print(rankings_str)
print("Scores in original order:")
for i in R_i:
    print(i)
# Create the rankings array in the original order
rankings_original_order = np.arange(1, len(df) + 1)[np.argsort(ranking_order)]

# Print the rankings array in the original order
print("Rankings in original order:")
for i in rankings_original_order:
    print(i)
