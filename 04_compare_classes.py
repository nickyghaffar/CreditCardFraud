import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# Ensure plots folder exists
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# 1. Load the dataset with pandas
print("Loading dataset from data/creditcard.csv...")
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# 2. Separate the data into Normal (Class == 0) and Fraud (Class == 1)
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]

# 3 & 4. Compare Amount column and print count, mean, median, min, max
print("=== Amount Column Comparison ===")
print("--- Normal Transactions (Class 0) ---")
print(f"Count:   {len(normal_df):,}")
print(f"Mean:    ${normal_df['Amount'].mean():.4f}")
print(f"Median:  ${normal_df['Amount'].median():.4f}")
print(f"Minimum: ${normal_df['Amount'].min():.4f}")
print(f"Maximum: ${normal_df['Amount'].max():.4f}")
print()

print("--- Fraud Transactions (Class 1) ---")
print(f"Count:   {len(fraud_df):,}")
print(f"Mean:    ${fraud_df['Amount'].mean():.4f}")
print(f"Median:  ${fraud_df['Amount'].median():.4f}")
print(f"Minimum: ${fraud_df['Amount'].min():.4f}")
print(f"Maximum: ${fraud_df['Amount'].max():.4f}")
print()

# 5 & 6. For V1 to V28, calculate Normal Mean, Fraud Mean, and Absolute Difference
v_features = [f'V{i}' for i in range(1, 29)]
feature_diffs = []

for feat in v_features:
    normal_mean = normal_df[feat].mean()
    fraud_mean = fraud_df[feat].mean()
    abs_diff = abs(normal_mean - fraud_mean)
    feature_diffs.append({
        'Feature': feat,
        'Normal Mean': normal_mean,
        'Fraud Mean': fraud_mean,
        'Absolute Difference': abs_diff
    })

table_df = pd.DataFrame(feature_diffs)

# 7. Sort the table by Absolute Difference from largest to smallest
table_df_sorted = table_df.sort_values(by='Absolute Difference', ascending=False).reset_index(drop=True)

# 8. Print the top 10 features with the largest difference
print("=== Top 10 Features with Largest Difference Between Normal and Fraud ===")
print(table_df_sorted.head(10).to_string(index=False, formatters={
    'Normal Mean': '{:.6f}'.format,
    'Fraud Mean': '{:.6f}'.format,
    'Absolute Difference': '{:.6f}'.format
}))
print()

# 9 & 10. Create boxplot comparing Amount for Normal vs Fraud and save to plots/amount_normal_vs_fraud.png
print("Generating boxplot comparing Amount for Normal vs Fraud...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Subplot 1: With outliers (Full scale)
sns.boxplot(
    x='Class',
    y='Amount',
    data=df,
    ax=axes[0],
    palette=['#2b5c8f', '#d9534f'],
    hue='Class',
    legend=False
)
axes[0].set_title('Amount by Class (All Outliers Included)', fontsize=12, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Normal (Class 0)', 'Fraud (Class 1)'], fontsize=11)
axes[0].set_xlabel('Transaction Class', fontsize=11)
axes[0].set_ylabel('Transaction Amount ($)', fontsize=11)

# Subplot 2: Zoomed-in boxplot without outliers to clearly view quartiles and median
sns.boxplot(
    x='Class',
    y='Amount',
    data=df,
    ax=axes[1],
    palette=['#2b5c8f', '#d9534f'],
    hue='Class',
    legend=False,
    showfliers=False
)
axes[1].set_title('Amount by Class (Outliers Hidden to View Quartiles)', fontsize=12, fontweight='bold')
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Normal (Class 0)', 'Fraud (Class 1)'], fontsize=11)
axes[1].set_xlabel('Transaction Class', fontsize=11)
axes[1].set_ylabel('Transaction Amount ($)', fontsize=11)

plt.suptitle('Transaction Amount Comparison: Normal vs Fraud Transactions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

boxplot_path = os.path.join(plots_dir, 'amount_normal_vs_fraud.png')
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved boxplot: {boxplot_path}")
