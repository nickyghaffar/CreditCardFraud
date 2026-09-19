import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure plots directory exists
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# 1. Load the dataset with pandas
print("Loading dataset from data/creditcard.csv...")
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# 2. Calculate the correlation matrix for all numerical columns
print("Calculating correlation matrix...")
corr = df.corr()

# 3 & 4 & 5. Create correlation heatmap using seaborn and save to plots/correlation_heatmap.png
print("Generating correlation heatmap...")
plt.figure(figsize=(16, 13))
heatmap = sns.heatmap(
    corr,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.3,
    annot=False,
    cbar_kws={'label': 'Pearson Correlation Coefficient', 'shrink': 0.8}
)

plt.title('Correlation Heatmap of Credit Card Features', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

heatmap_path = os.path.join(plots_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved heatmap: {heatmap_path}\n")

# 6 & 7. Find top 10 feature pairs with highest absolute correlation (excluding self and duplicate pairs)
columns = corr.columns
pairs = []

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        feat1 = columns[i]
        feat2 = columns[j]
        val = corr.iloc[i, j]
        pairs.append({
            'Feature 1': feat1,
            'Feature 2': feat2,
            'Correlation': val,
            'Absolute Correlation': abs(val)
        })

pairs_df = pd.DataFrame(pairs)
top_10_pairs = pairs_df.sort_values(by='Absolute Correlation', ascending=False).head(10).reset_index(drop=True)

print("=== Top 10 Feature Pairs with Highest Absolute Correlation ===")
print(top_10_pairs.to_string(index=False, formatters={
    'Correlation': '{:+.6f}'.format,
    'Absolute Correlation': '{:.6f}'.format
}))
