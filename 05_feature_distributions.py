import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual theme
sns.set_theme(style="whitegrid")

# Ensure plots folder exists
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# 1. Load the dataset with pandas
print("Loading dataset from data/creditcard.csv...")
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# 2. Separate Normal (Class=0) and Fraud (Class=1) transactions
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]
print(f"Normal transactions (Class 0): {len(normal_df):,}")
print(f"Fraud transactions (Class 1):  {len(fraud_df):,}\n")

# 3. Features to investigate based on largest mean difference
features = ['V3', 'V14', 'V17', 'V12', 'V10']

# 4 - 8. Create one separate image for each feature with alpha transparency
for feat in features:
    print(f"Generating distribution plot for {feat}...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate common range and bins for consistent comparison
    min_val = min(normal_df[feat].min(), fraud_df[feat].min())
    max_val = max(normal_df[feat].max(), fraud_df[feat].max())
    bins = np.linspace(min_val, max_val, 60)
    
    # Density=True normalizes the areas so both classes are comparable despite severe imbalance (492 vs 284,315)
    ax.hist(
        normal_df[feat],
        bins=bins,
        density=True,
        alpha=0.5,
        color='#2b5c8f',
        label='Normal (Class 0)',
        edgecolor='black',
        linewidth=0.5
    )
    ax.hist(
        fraud_df[feat],
        bins=bins,
        density=True,
        alpha=0.6,
        color='#d9534f',
        label='Fraud (Class 1)',
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_title(f'Distribution of {feat}: Normal vs Fraud Transactions', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(f'{feat} Value', fontsize=12, labelpad=10)
    ax.set_ylabel('Density (Normalized Frequency)', fontsize=12, labelpad=10)
    ax.legend(fontsize=11, frameon=True)
    
    plt.tight_layout()
    output_path = os.path.join(plots_dir, f'{feat}_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

print("\nAll feature distribution plots have been successfully created and saved in the 'plots' folder!")
