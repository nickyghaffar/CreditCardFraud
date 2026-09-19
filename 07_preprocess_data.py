import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Force UTF-8 output encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 60)
print("Step 2: Data Preprocessing & Stratified Splitting")
print("=" * 60)

# 1. Load Dataset
data_path = 'data/creditcard.csv'
print(f"1. Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
print(f"   Total records: {df.shape[0]:,} rows | Columns: {df.shape[1]}")

# 2. Separate Features (X) and Target (y = Class)
print("\n2. Separating Features (X) and Target (y = Class)...")
X = df.drop(columns=['Class'])
y = df['Class']

# 3. Stratified Train-Test Split (80% Train, 20% Test)
# stratify=y guarantees exact same proportion of fraud cases in both sets
print("\n3. Performing Stratified Train/Test Split (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Scale Time and Amount using RobustScaler
# Crucial: Fit on X_train ONLY, then transform X_train and X_test to prevent Data Leakage
print("\n4. Scaling 'Time' and 'Amount' features using RobustScaler (outlier-resistant)...")
scaler_amount = RobustScaler()
scaler_time = RobustScaler()

X_train = X_train.copy()
X_test = X_test.copy()

X_train['scaled_amount'] = scaler_amount.fit_transform(X_train[['Amount']])
X_test['scaled_amount'] = scaler_amount.transform(X_test[['Amount']])

X_train['scaled_time'] = scaler_time.fit_transform(X_train[['Time']])
X_test['scaled_time'] = scaler_time.transform(X_test[['Time']])

# Remove raw unscaled columns
X_train = X_train.drop(columns=['Time', 'Amount'])
X_test = X_test.drop(columns=['Time', 'Amount'])

print("   Scaling complete. Raw 'Time' and 'Amount' replaced with 'scaled_time' and 'scaled_amount'.")

# 5. Class balance verification
print("\n5. Verifying Class Proportions:")
print("-" * 55)
train_total = len(y_train)
train_fraud = int((y_train == 1).sum())
train_normal = int((y_train == 0).sum())
print(f"Train Set: {train_total:,} samples")
print(f"  - Normal (Class 0): {train_normal:,} ({train_normal / train_total * 100:.3f}%)")
print(f"  - Fraud  (Class 1): {train_fraud:,} ({train_fraud / train_total * 100:.3f}%)")

print("-" * 55)
test_total = len(y_test)
test_fraud = int((y_test == 1).sum())
test_normal = int((y_test == 0).sum())
print(f"Test Set:  {test_total:,} samples")
print(f"  - Normal (Class 0): {test_normal:,} ({test_normal / test_total * 100:.3f}%)")
print(f"  - Fraud  (Class 1): {test_fraud:,} ({test_fraud / test_total * 100:.3f}%)")
print("-" * 55)

# 6. Save processed datasets for subsequent modeling scripts
output_path = 'data/processed_data.joblib'
print(f"\n6. Saving preprocessed datasets to: {output_path} ...")
processed_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time
}
joblib.dump(processed_data, output_path)
filesize_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"   Successfully saved '{output_path}' ({filesize_mb:.2f} MB).")
print("\nPre-processing phase finished successfully!")
