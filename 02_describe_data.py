import pandas as pd

# 1. Load the dataset with pandas
df = pd.read_csv('data/creditcard.csv')

# 2. Print df.describe() for all numerical columns
print("=== 1. Summary Statistics (df.describe()) ===")
# Set display options so columns and rows are clearly visible
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.describe())
print()

# 3. Print the percentage of each class in the Class column
print("=== 2. Class Percentages ===")
class_percentages = df['Class'].value_counts(normalize=True) * 100
print(class_percentages)
print()

# 4. Print the minimum and maximum value of Amount
print("=== 3. Amount: Minimum and Maximum ===")
print(f"Min Amount: {df['Amount'].min()}")
print(f"Max Amount: {df['Amount'].max()}")
print()

# 5. Print the mean and median of Amount
print("=== 4. Amount: Mean and Median ===")
print(f"Mean Amount: {df['Amount'].mean():.4f}")
print(f"Median Amount: {df['Amount'].median():.4f}")
print()

# 6. Print the minimum and maximum value of Time
print("=== 5. Time: Minimum and Maximum ===")
print(f"Min Time: {df['Time'].min()}")
print(f"Max Time: {df['Time'].max()}")
