import pandas as pd

# 2. Load data/creditcard.csv into a pandas DataFrame named df
df = pd.read_csv('data/creditcard.csv')

# 3. Print the shape of the dataset
print("=== 1. Shape of the Dataset ===")
print(df.shape)
print()

# 4. Print the first 5 rows using df.head()
print("=== 2. First 5 Rows (df.head()) ===")
print(df.head())
print()

# 5. Print the column names
print("=== 3. Column Names ===")
print(df.columns.tolist())
print()

# 6. Print the data types using df.dtypes
print("=== 4. Data Types (df.dtypes) ===")
print(df.dtypes)
print()

# 7. Print the number of missing values in each column
print("=== 5. Missing Values in Each Column ===")
print(df.isnull().sum())
print()

# 8. Print the number of samples in each class using the Class column
print("=== 6. Number of Samples in Each Class ===")
print(df['Class'].value_counts())
