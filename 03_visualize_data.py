import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# Create plots folder
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# 1. Load dataset with pandas
print("Loading dataset from data/creditcard.csv...")
df = pd.read_csv('data/creditcard.csv')
print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.\n")

# 2. Bar chart showing the number of Normal (Class=0) and Fraud (Class=1) transactions
print("1/4: Generating Class Distribution bar chart...")
fig, ax = plt.subplots(figsize=(7, 5))
class_counts = df['Class'].value_counts()
classes = ['Normal (Class 0)', 'Fraud (Class 1)']
counts = [class_counts[0], class_counts[1]]
colors = ['#2b5c8f', '#d9534f']
bars = ax.bar(classes, counts, color=colors, width=0.5)

ax.set_title('Transaction Class Distribution (Normal vs Fraud)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Transaction Class', fontsize=12, labelpad=10)
ax.set_ylabel('Number of Transactions', fontsize=12, labelpad=10)

# Annotate counts on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plot1_name = 'class_distribution.png'
plot1_path = os.path.join(plots_dir, plot1_name)
plt.savefig(plot1_path, dpi=300)
plt.close()
print(f"Saved: {plot1_path}")

# 3. Histogram of the Amount column
print("2/4: Generating Amount histogram...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['Amount'], bins=50, color='#2b5c8f', edgecolor='black', alpha=0.75)
ax.set_title('Distribution of Transaction Amounts', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Transaction Amount ($)', fontsize=12, labelpad=10)
ax.set_ylabel('Frequency (Number of Transactions)', fontsize=12, labelpad=10)

plt.tight_layout()
plot2_name = 'amount_distribution.png'
plot2_path = os.path.join(plots_dir, plot2_name)
plt.savefig(plot2_path, dpi=300)
plt.close()
print(f"Saved: {plot2_path}")

# 4. Histogram of Amount using a logarithmic x-axis
print("3/4: Generating Amount histogram with logarithmic x-axis...")
fig, ax = plt.subplots(figsize=(8, 5))
# Using Amount + 1 with log-spaced bins so 0-amount transactions are safely included
bins = np.logspace(0, np.log10(df['Amount'].max() + 1), 60)
ax.hist(df['Amount'] + 1, bins=bins, color='#17a2b8', edgecolor='black', alpha=0.75)
ax.set_xscale('log')
ax.set_title('Distribution of Transaction Amounts (Logarithmic X-Axis)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Transaction Amount + 1 ($) [Log Scale]', fontsize=12, labelpad=10)
ax.set_ylabel('Frequency (Number of Transactions)', fontsize=12, labelpad=10)

plt.tight_layout()
plot3_name = 'amount_distribution_log.png'
plot3_path = os.path.join(plots_dir, plot3_name)
plt.savefig(plot3_path, dpi=300)
plt.close()
print(f"Saved: {plot3_path}")

# 5. Histogram of the Time column
print("4/4: Generating Time histogram...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['Time'], bins=50, color='#28a745', edgecolor='black', alpha=0.75)
ax.set_title('Distribution of Transaction Time', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Time (Seconds elapsed from first transaction)', fontsize=12, labelpad=10)
ax.set_ylabel('Frequency (Number of Transactions)', fontsize=12, labelpad=10)

plt.tight_layout()
plot4_name = 'time_distribution.png'
plot4_path = os.path.join(plots_dir, plot4_name)
plt.savefig(plot4_path, dpi=300)
plt.close()
print(f"Saved: {plot4_path}")

print("\nAll plots have been successfully created and saved in the 'plots' folder!")
