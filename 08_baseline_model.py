import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score
)

# Force UTF-8 output encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure output directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 65)
print("Step 3: Training Baseline Model (Logistic Regression)")
print("=" * 65)

# 1. Load preprocessed data
data_file = 'data/processed_data.joblib'
print(f"1. Loading preprocessed data from: {data_file} ...")
data = joblib.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"   Train samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")
print(f"   Number of features: {X_train.shape[1]}")

# 2. Initialize and train baseline Logistic Regression model
print("\n2. Training Logistic Regression model (Baseline)...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("   Model training completed.")

# 3. Predict on Test set
print("\n3. Evaluating model performance on Test Set (56,962 unseen transactions)...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 4. Detailed Metrics
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print("\n" + "=" * 65)
print("EVALUATION RESULTS (BASELINE LOGISTIC REGRESSION)")
print("=" * 65)

print("\n--- Confusion Matrix ---")
print(f"True Negatives  (Legitimate correctly identified) : {tn:,} / {tn + fp:,}")
print(f"False Positives (False Alarms - Normal called Fraud): {fp:,}")
print(f"False Negatives (MISSED FRAUDS - Danger!)          : {fn:,} / {fn + tp:,}")
print(f"True Positives  (Fraud correctly caught)          : {tp:,} / {fn + tp:,}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Normal (Class 0)', 'Fraud (Class 1)'], digits=4))

print("--- Overall Imbalanced Metrics ---")
print(f"ROC-AUC Score                  : {roc_auc:.4f}")
print(f"PR-AUC (Average Precision) Score: {pr_auc:.4f}")

# 5. Visualizations
# A. Confusion Matrix Plot
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt=',d',
    cmap='Blues',
    xticklabels=['Normal (0)', 'Fraud (1)'],
    yticklabels=['Normal (0)', 'Fraud (1)']
)
plt.title('Baseline Logistic Regression: Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual True Label', fontsize=11)
plt.tight_layout()
cm_plot_path = 'plots/baseline_confusion_matrix.png'
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"\nSaved Confusion Matrix plot: {cm_plot_path}")

# B. Precision-Recall Curve Plot
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='#2b5c8f', lw=2, label=f'Baseline LogReg (PR-AUC = {pr_auc:.4f})')
plt.title('Precision-Recall Curve (Baseline Model)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Recall (Detection Rate of Frauds)', fontsize=11)
plt.ylabel('Precision (True Frauds / All Predicted Frauds)', fontsize=11)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
pr_plot_path = 'plots/baseline_pr_curve.png'
plt.savefig(pr_plot_path, dpi=300)
plt.close()
print(f"Saved Precision-Recall curve plot: {pr_plot_path}")

# 6. Save trained model
model_path = 'models/baseline_logistic_regression.joblib'
joblib.dump(model, model_path)
print(f"Saved trained model to: {model_path}")
print("\nStep 3 completed successfully!")
