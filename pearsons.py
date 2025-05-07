# import numpy as np
from scipy.stats import pearsonr
import pandas as pd

# --- Data Compilation ---
# Data extracted from the thesis report (Sections 4.1, 4.2, 4.3)
# Model Configurations:
# 1. COSFIRE 48-bit (C48)
# 2. COSFIRE 72-bit (C72)
# 3. DenseNet 56-bit (D56)
# 4. DenseNet 72-bit (D72)

data = {
    'Model': ['COSFIRE_48', 'COSFIRE_72', 'DenseNet_56', 'DenseNet_72'],
    'mAP (%)': [90.1, 88.69, 89.48, 86.96],
    'Code Length (bits)': [48, 72, 56, 72],
    'Avg. Hamming Wt.': [24.04, 26.52, 27.22, 34.83],
    'Avg. Sparsity Ratio': [0.50, 0.63, 0.51, 0.52], # C48: (48-24.04)/48 = 0.4991 => 0.50. C72: (72-26.52)/72 = 0.6316 => 0.63. D56: (56-27.22)/56 = 0.5139 => 0.51. D72: (72-34.83)/72 = 0.5162 => 0.52
    'Avg. Hashing Time (s)': [0.00003053, 0.00003024, 0.00104862, 0.00106079]
}

df = pd.DataFrame(data)

# --- Correlation Pairs for Investigation ---

# Define the pairs of metrics to correlate
# Format: ('Metric 1 Name', 'Metric 2 Name')
correlation_pairs = [
    # Accuracy (mAP) vs. Sparsity Metrics
    ('mAP (%)', 'Code Length (bits)'),
    ('mAP (%)', 'Avg. Hamming Wt.'),
    ('mAP (%)', 'Avg. Sparsity Ratio'),

    # Efficiency (Hashing Time) vs. Sparsity Metrics
    ('Avg. Hashing Time (s)', 'Code Length (bits)'),
    ('Avg. Hashing Time (s)', 'Avg. Hamming Wt.'),
    ('Avg. Hashing Time (s)', 'Avg. Sparsity Ratio'),

    # Accuracy (mAP) vs. Efficiency (Hashing Time)
    ('mAP (%)', 'Avg. Hashing Time (s)')
]

print("--- Pearson's Correlation Analysis (N=4) ---")

results = []

for metric1_name, metric2_name in correlation_pairs:
    metric1_values = df[metric1_name]
    metric2_values = df[metric2_name]

    # Calculate Pearson's correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(metric1_values, metric2_values)

    results.append({
        'Metric 1': metric1_name,
        'Metric 2': metric2_name,
        'Pearson_r': correlation_coefficient,
        'p_value': p_value
    })

    print(f"Correlation between '{metric1_name}' and '{metric2_name}':")
    print(f"  Pearson's r = {correlation_coefficient:.4f}")
    print(f"  p-value     = {p_value:.4f}\n")

results_df = pd.DataFrame(results)
print("\n--- Summary of Correlation Results ---")
print(results_df.to_string())
