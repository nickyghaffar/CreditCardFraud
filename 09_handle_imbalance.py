import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score
)

# Force UTF-8 encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 70)
print("Step 4: Handling Class Imbalance (Class Weighting vs SMOTE)")
print("=" * 70)

# 1. Load preprocessed data
data_file = 'data/processed_data.joblib'
print(f"1. Loading preprocessed data from: {data_file} ...")
data = joblib.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"   Train samples: {len(X_train):,} (Frauds: {(y_train == 1).sum()})")
print(f"   Test samples:  {len(X_test):,} (Frauds: {(y_test == 1).sum()})\n")

# Dictionary to store performance results
results = []
models = {}
curves = {}

# -------------------------------------------------------------
# Model A: Baseline Logistic Regression (Unweighted)
# -------------------------------------------------------------
print("2. Evaluating Model A: Baseline Logistic Regression...")
model_base = joblib.load('models/baseline_logistic_regression.joblib')
y_pred_base = model_base.predict(X_test)
y_prob_base = model_base.predict_proba(X_test)[:, 1]

cm_base = confusion_matrix(y_test, y_pred_base)
tn_b, fp_b, fn_b, tp_b = cm_base.ravel()

results.append({
    'Model': '1. Baseline LogReg',
    'Precision (Fraud)': precision_score(y_test, y_pred_base),
    'Recall (Fraud)': recall_score(y_test, y_pred_base),
    'F1-Score': f1_score(y_test, y_pred_base),
    'PR-AUC': average_precision_score(y_test, y_prob_base),
    'Missed Frauds (FN)': fn_b,
    'False Alarms (FP)': fp_b
})
curves['Baseline'] = precision_recall_curve(y_test, y_prob_base)

# -------------------------------------------------------------
# Model B: Weighted Logistic Regression (class_weight='balanced')
# -------------------------------------------------------------
print("3. Training Model B: Weighted Logistic Regression (class_weight='balanced')...")
model_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
y_prob_weighted = model_weighted.predict_proba(X_test)[:, 1]

cm_weighted = confusion_matrix(y_test, y_pred_weighted)
tn_w, fp_w, fn_w, tp_w = cm_weighted.ravel()

results.append({
    'Model': '2. Weighted LogReg',
    'Precision (Fraud)': precision_score(y_test, y_pred_weighted),
    'Recall (Fraud)': recall_score(y_test, y_pred_weighted),
    'F1-Score': f1_score(y_test, y_pred_weighted),
    'PR-AUC': average_precision_score(y_test, y_prob_weighted),
    'Missed Frauds (FN)': fn_w,
    'False Alarms (FP)': fp_w
})
curves['Weighted'] = precision_recall_curve(y_test, y_prob_weighted)
joblib.dump(model_weighted, 'models/weighted_logistic_regression.joblib')

# -------------------------------------------------------------
# Model C: SMOTE (Synthetic Minority Over-sampling) + Logistic Regression
# -------------------------------------------------------------
print("4. Applying SMOTE to Training Set (Synthesizing Fraud Samples)...")
smote = SMOTE(random_state=42)
# Crucial: SMOTE is applied ONLY on training data to prevent leakage into test set!
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"   Original train fraud count: {(y_train == 1).sum():,}")
print(f"   After SMOTE fraud count:    {(y_train_smote == 1).sum():,}")

print("   Training Logistic Regression on SMOTE-balanced data...")
model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

cm_smote = confusion_matrix(y_test, y_pred_smote)
tn_s, fp_s, fn_s, tp_s = cm_smote.ravel()

results.append({
    'Model': '3. SMOTE + LogReg',
    'Precision (Fraud)': precision_score(y_test, y_pred_smote),
    'Recall (Fraud)': recall_score(y_test, y_pred_smote),
    'F1-Score': f1_score(y_test, y_pred_smote),
    'PR-AUC': average_precision_score(y_test, y_prob_smote),
    'Missed Frauds (FN)': fn_s,
    'False Alarms (FP)': fp_s
})
curves['SMOTE'] = precision_recall_curve(y_test, y_prob_smote)
joblib.dump(model_smote, 'models/smote_logistic_regression.joblib')

# -------------------------------------------------------------
# Comparison Table
# -------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("COMPARATIVE RESULTS: HANDLING IMBALANCE TECHNIQUES")
print("=" * 70)
print(results_df.to_string(index=False, formatters={
    'Precision (Fraud)': '{:.2%}'.format,
    'Recall (Fraud)': '{:.2%}'.format,
    'F1-Score': '{:.4f}'.format,
    'PR-AUC': '{:.4f}'.format,
    'Missed Frauds (FN)': '{:d}'.format,
    'False Alarms (FP)': '{:d}'.format
}))
print("=" * 70)

# -------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------
# 1. 3 Confusion Matrices Side-by-Side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cms = [
    (cm_base, '1. Baseline LogReg', '#2b5c8f'),
    (cm_weighted, '2. Weighted LogReg', '#e67e22'),
    (cm_smote, '3. SMOTE + LogReg', '#27ae60')
]

for idx, (matrix, title, _) in enumerate(cms):
    sns.heatmap(
        matrix,
        annot=True,
        fmt=',d',
        cmap='Blues',
        ax=axes[idx],
        xticklabels=['Normal', 'Fraud'],
        yticklabels=['Normal', 'Fraud']
    )
    axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    axes[idx].set_ylabel('Actual Label', fontsize=10)

plt.suptitle('Confusion Matrix Comparison: Baseline vs Weighted vs SMOTE', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
cm_cmp_path = 'plots/imbalance_comparison_confusion_matrices.png'
plt.savefig(cm_cmp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved Confusion Matrices comparison plot: {cm_cmp_path}")

# 2. Comparative Precision-Recall Curves
plt.figure(figsize=(8, 6))
colors = {'Baseline': '#2b5c8f', 'Weighted': '#e67e22', 'SMOTE': '#27ae60'}

for name, (p, r, _) in curves.items():
    auc_val = [res['PR-AUC'] for res in results if name.lower() in res['Model'].lower()][0]
    plt.plot(r, p, color=colors[name], lw=2.2, label=f'{name} (PR-AUC = {auc_val:.4f})')

plt.title('Precision-Recall Curves Comparison', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Recall (Fraud Detection Rate)', fontsize=11)
plt.ylabel('Precision (Accuracy of Flagged Frauds)', fontsize=11)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

pr_cmp_path = 'plots/imbalance_pr_curves.png'
plt.savefig(pr_cmp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved PR Curves comparison plot: {pr_cmp_path}")

print("\nStep 4 completed successfully!")
