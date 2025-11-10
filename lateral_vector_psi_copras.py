import numpy as np
import pandas as pd

# Load the Excel file
filename = '*****.xlsx'
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

# Step 2.2 Normalization 2 (Vector)
normalized_decision_matrix_2 = np.zeros_like(decision_matrix, dtype=float)
for j in range(n):
    if j in negative_indicators:
        normalized_decision_matrix_2[:, j] = (1 / decision_matrix[:, j]) / np.sqrt(np.sum(1/decision_matrix[:, j] ** 2))
    else:
        normalized_decision_matrix_2[:, j] = decision_matrix[:, j] / np.sqrt(np.sum(decision_matrix[:, j] ** 2))

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

# Step 4: Calculation of Pi, Ri and Qi using COPRAS method
Pi = np.sum(weighted_dm_matrix[:, ~np.isin(range(n), negative_indicators)], axis=1)
Ri = np.sum(weighted_dm_matrix[:, negative_indicators], axis=1)
Qi = Pi + (Ri.min() * Ri.sum()) / (Ri * m)

# The priority of ranks from alternatives can be calculated from significance Qi.
# Greater significance (relative weight of alternative) Qi, the higher is the priority (rank) of the alternative.
Ui = (Qi / Qi.max()) * 100
ranking_order = Ui.argsort()[::-1]

# Create a DataFrame for the rankings
rankings_df = pd.DataFrame({
    'Alternative': df.index,
    'Qi': Qi,
    'Ui': Ui
})

# Sort the DataFrame by Ui in descending order
rankings_df = rankings_df.sort_values('Ui', ascending=False)

# Add the rank column
rankings_df['Rank'] = range(1, m + 1)

# Reset index and display the rankings DataFrame
rankings_df = rankings_df.reset_index(drop=True)
rankings_str = rankings_df.to_string(index=False)

# Output the rankings DataFrame
print(rankings_str)

print("Scores in original order:")
for i in Ui:
    print(i)

# Create the rankings array in the original order
rankings_original_order = np.arange(1, len(df) + 1)[np.argsort(ranking_order)]

# Print the rankings array in the original order
print("Rankings in original order:")
for i in rankings_original_order:
    print(i) 