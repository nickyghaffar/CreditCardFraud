import os
import sys
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
print("Step 5: Training Advanced Non-Linear Model (Random Forest)")
print("=" * 70)

# 1. Load preprocessed data
data_file = 'data/processed_data.joblib'
print(f"1. Loading preprocessed data from: {data_file} ...")
data = joblib.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"   Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

# 2. Train Random Forest Classifier
print("\n2. Training Random Forest Classifier (100 Trees, balanced subsample, parallelized)...")
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
elapsed = time.time() - start_time
print(f"   Random Forest training completed in {elapsed:.1f} seconds.")

# 3. Evaluate on Test Set
print("\n3. Evaluating Random Forest on unseen Test Set (56,962 transactions)...")
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

cm_rf = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = cm_rf.ravel()
prec = precision_score(y_test, y_pred_rf)
rec = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
pr_auc = average_precision_score(y_test, y_prob_rf)
roc_auc = roc_auc_score(y_test, y_prob_rf)

print("\n" + "=" * 70)
print("EVALUATION RESULTS (RANDOM FOREST)")
print("=" * 70)
print(f"True Negatives  (Normal correctly classified)   : {tn:,} / {tn + fp:,}")
print(f"False Positives (False Alarms - Normal as Fraud): {fp:,}")
print(f"False Negatives (MISSED FRAUDS)                 : {fn:,} / {fn + tp:,}")
print(f"True Positives  (Fraud correctly caught)        : {tp:,} / {fn + tp:,}")
print("-" * 70)
print(f"Precision : {prec:.2%}")
print(f"Recall    : {rec:.2%}")
print(f"F1-Score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("=" * 70)

# 4. Feature Importance Analysis
print("\n4. Analyzing Feature Importances...")
importances = rf_model.feature_importances_
feature_names = X_train.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print("Top 10 Most Important Features:")
print(fi_df.head(10).to_string(index=False, formatters={'Importance': '{:.4f}'.format}))

# Plot Top 15 Feature Importances
plt.figure(figsize=(10, 6))
top_fi = fi_df.head(15).sort_values(by='Importance', ascending=True)
plt.barh(top_fi['Feature'], top_fi['Importance'], color='#2b5c8f', edgecolor='black', alpha=0.8)
plt.title('Top 15 Features for Fraud Detection (Random Forest)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Gini Importance (Relative Predictive Power)', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.tight_layout()
fi_plot_path = 'plots/rf_feature_importance.png'
plt.savefig(fi_plot_path, dpi=300)
plt.close()
print(f"Saved Feature Importance plot: {fi_plot_path}")

# 5. Confusion Matrix Visualization
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt=',d',
    cmap='Blues',
    xticklabels=['Normal', 'Fraud'],
    yticklabels=['Normal', 'Fraud']
)
plt.title('Random Forest: Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Predicted Label', fontsize=11)
plt.ylabel('Actual True Label', fontsize=11)
plt.tight_layout()
rf_cm_path = 'plots/rf_confusion_matrix.png'
plt.savefig(rf_cm_path, dpi=300)
plt.close()
print(f"Saved Random Forest Confusion Matrix: {rf_cm_path}")

# 6. Overall PR Curve Comparison (Baseline vs Weighted vs Random Forest)
model_base = joblib.load('models/baseline_logistic_regression.joblib')
y_prob_base = model_base.predict_proba(X_test)[:, 1]
pr_auc_base = average_precision_score(y_test, y_prob_base)
p_base, r_base, _ = precision_recall_curve(y_test, y_prob_base)

model_weighted = joblib.load('models/weighted_logistic_regression.joblib')
y_prob_weighted = model_weighted.predict_proba(X_test)[:, 1]
pr_auc_weighted = average_precision_score(y_test, y_prob_weighted)
p_wt, r_wt, _ = precision_recall_curve(y_test, y_prob_weighted)

p_rf, r_rf, _ = precision_recall_curve(y_test, y_prob_rf)

plt.figure(figsize=(9, 6))
plt.plot(r_base, p_base, label=f'Baseline LogReg (PR-AUC = {pr_auc_base:.4f})', color='#7f8c8d', lw=2, linestyle='--')
plt.plot(r_wt, p_wt, label=f'Weighted LogReg (PR-AUC = {pr_auc_weighted:.4f})', color='#e67e22', lw=2)
plt.plot(r_rf, p_rf, label=f'Random Forest (PR-AUC = {pr_auc:.4f})', color='#27ae60', lw=2.5)

plt.title('Comprehensive PR Curves: Logistic Regression vs Random Forest', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Recall (Fraud Detection Rate)', fontsize=11)
plt.ylabel('Precision (True Frauds / All Flagged)', fontsize=11)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
final_pr_path = 'plots/final_pr_comparison.png'
plt.savefig(final_pr_path, dpi=300)
plt.close()
print(f"Saved Final PR Curves Comparison: {final_pr_path}")

# 7. Save Random Forest Model
model_save_path = 'models/random_forest_model.joblib'
joblib.dump(rf_model, model_save_path)
print(f"Saved trained Random Forest model: {model_save_path}")

print("\nStep 5 completed successfully!")
