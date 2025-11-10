import sys
print(sys.executable)
print(sys.version)
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

# Create a boolean mask for the beneficial criteria
beneficial_mask = np.ones(n, dtype=bool)
beneficial_mask[negative_indicators] = False

# Normalization of decision matrix
normalized_decision_matrix = np.zeros_like(decision_matrix, dtype=float)
for j in range(n):
    if j in negative_indicators:
        normalized_decision_matrix[:, j] = (1 / decision_matrix[:, j]) / np.sqrt(np.sum(1/decision_matrix[:, j] ** 2))
    else:
        normalized_decision_matrix[:, j] = decision_matrix[:, j] / np.sqrt(np.sum(decision_matrix[:, j] ** 2))
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
# Define the preference function (e.g., linear preference function)
def preference_function(d):
    return np.where(d <= 0, 0, np.where(d <= 1, d, 1))

# Calculate the preference indices
preference_indices = np.zeros((m, m, n))
for k in range(n):
    for i in range(m):
        for j in range(m):
            d = normalized_decision_matrix[i, k] - normalized_decision_matrix[j, k]
            preference_indices[i, j, k] = preference_function(d)

# Calculate the aggregated preference indices
aggregated_preference_indices = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        aggregated_preference_indices[i, j] = np.sum(preference_indices[i, j, :] * weights)

# Calculate the positive and negative outranking flows
positive_flow = np.mean(aggregated_preference_indices, axis=1)
negative_flow = np.mean(aggregated_preference_indices, axis=0)
# Calculate the net outranking flow
net_flow = positive_flow - negative_flow

# Rank the alternatives based on the net outranking flow
ranking_order = net_flow.argsort()[::-1]
ranked_countries = df.index[ranking_order]
rankings = np.arange(1, len(df) + 1)

# Create a DataFrame for the rankings
rankings_df = pd.DataFrame({
    'Rank': rankings,
    'Country': ranked_countries,
    'Net Flow': net_flow[ranking_order]
})

# Reset index to start from 1 and show as the first column
rankings_df.index = np.arange(1, m + 1)
rankings_str = rankings_df.to_string(index=False)

# Output the rankings DataFrame
print(rankings_str)
rankings_original_order = np.arange(1, len(df) + 1)[np.argsort(ranking_order)]
# Print the rankings array in the original order
print("Rankings in original order:")
for i in rankings_original_order:
    print(i)

for i in net_flow:
    print(i)
