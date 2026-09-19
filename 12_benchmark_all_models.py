import os
import sys
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Force UTF-8 output encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import ML Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# Ensure directories
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=" * 80)
print("Step 7: Comprehensive Benchmark of All 12 Major Machine Learning Models")
print("=" * 80)

# 1. Load preprocessed data
data_file = 'data/processed_data.joblib'
print(f"Loading preprocessed data from: {data_file} ...")
data = joblib.load(data_file)
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"Train Set: {len(X_train):,} samples (Frauds: {(y_train == 1).sum()})")
print(f"Test Set:  {len(X_test):,} samples (Frauds: {(y_test == 1).sum()})\n")

# Calculate positive scale weight for boosting models: negative / positive
pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

# Define All Models
models_dict = {
    "1. Logistic Regression (Base)": LogisticRegression(max_iter=1000, random_state=42),
    "2. Logistic Regression (Weighted)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "3. Gaussian Naive Bayes": GaussianNB(),
    "4. Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight='balanced', random_state=42),
    "5. Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced_subsample', n_jobs=-1, random_state=42),
    "6. Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42),
    "7. AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "8. HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, class_weight='balanced', random_state=42),
    "9. XGBoost": XGBClassifier(n_estimators=100, max_depth=5, scale_pos_weight=pos_weight, eval_metric='logloss', n_jobs=-1, random_state=42),
    "10. LightGBM": LGBMClassifier(n_estimators=100, max_depth=6, scale_pos_weight=pos_weight, n_jobs=-1, verbose=-1, random_state=42),
    "11. CatBoost": CatBoostClassifier(iterations=100, depth=5, auto_class_weights='Balanced', verbose=0, thread_count=-1, random_state=42),
    "12. Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=40, early_stopping=True, random_state=42)
}

benchmark_records = []
trained_models = {}

print("Starting training and evaluation for each model...\n")
for idx, (name, clf) in enumerate(models_dict.items(), 1):
    print(f"[{idx}/12] Training {name} ...", end=" ", flush=True)
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predict on test
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        # Fallback for models without predict_proba
        y_prob = clf.decision_function(X_test)
        
    # Evaluate
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    benchmark_records.append({
        'Model': name,
        'Train Time (s)': round(train_time, 2),
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'PR-AUC': pr_auc,
        'ROC-AUC': roc_auc,
        'Missed (FN)': fn,
        'False Alarms (FP)': fp
    })
    
    trained_models[name] = clf
    print(f"Done in {train_time:.1f}s | PR-AUC: {pr_auc:.4f} | F1: {f1:.4f} | Missed: {fn}")

# Add 13. Voting Ensemble (Soft Voting of Top 3: Random Forest + XGBoost + LightGBM)
print("\n[13/13] Building Ensemble Voting Classifier (Random Forest + XGBoost + LightGBM)...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', trained_models['5. Random Forest']),
        ('xgb', trained_models['9. XGBoost']),
        ('lgbm', trained_models['10. LightGBM'])
    ],
    voting='soft',
    n_jobs=-1
)
# Pre-fitted voting estimation
voting_probs = (
    trained_models['5. Random Forest'].predict_proba(X_test)[:, 1] +
    trained_models['9. XGBoost'].predict_proba(X_test)[:, 1] +
    trained_models['10. LightGBM'].predict_proba(X_test)[:, 1]
) / 3.0
voting_preds = (voting_probs >= 0.5).astype(int)

cm_v = confusion_matrix(y_test, voting_preds)
tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
prec_v = precision_score(y_test, voting_preds)
rec_v = recall_score(y_test, voting_preds)
f1_v = f1_score(y_test, voting_preds)
pr_auc_v = average_precision_score(y_test, voting_probs)
roc_auc_v = roc_auc_score(y_test, voting_probs)

benchmark_records.append({
    'Model': '13. Super Ensemble (RF+XGB+LGB)',
    'Train Time (s)': 0.0,
    'Precision': prec_v,
    'Recall': rec_v,
    'F1-Score': f1_v,
    'PR-AUC': pr_auc_v,
    'ROC-AUC': roc_auc_v,
    'Missed (FN)': fn_v,
    'False Alarms (FP)': fp_v
})

# Save all trained models bundle
all_models_path = 'models/all_trained_models.joblib'
joblib.dump(trained_models, all_models_path)
print(f"\nAll models saved in: {all_models_path}")

# Display Final Benchmark Table Sorted by PR-AUC
bench_df = pd.DataFrame(benchmark_records)
bench_df_sorted = bench_df.sort_values(by='PR-AUC', ascending=False).reset_index(drop=True)

print("\n" + "=" * 105)
print("FINAL BENCHMARK: RANKED COMPARISON OF ALL 13 MACHINE LEARNING MODELS")
print("=" * 105)
print(bench_df_sorted.to_string(index=False, formatters={
    'Train Time (s)': '{:.1f}'.format,
    'Precision': '{:.2%}'.format,
    'Recall': '{:.2%}'.format,
    'F1-Score': '{:.4f}'.format,
    'PR-AUC': '{:.4f}'.format,
    'ROC-AUC': '{:.4f}'.format,
    'Missed (FN)': '{:d}'.format,
    'False Alarms (FP)': '{:d}'.format
}))
print("=" * 105)

# Save Benchmark Table to CSV
csv_path = 'models/benchmark_results.csv'
bench_df_sorted.to_csv(csv_path, index=False)
print(f"Saved benchmark results table to: {csv_path}")

# -------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------
# 1. Bar Chart of PR-AUC
plt.figure(figsize=(12, 7))
plot_df = bench_df_sorted.sort_values(by='PR-AUC', ascending=True)
colors = ['#27ae60' if 'Ensemble' in m or 'Random Forest' in m or 'XGB' in m or 'Light' in m else '#2b5c8f' for m in plot_df['Model']]
bars = plt.barh(plot_df['Model'], plot_df['PR-AUC'], color=colors, edgecolor='black', alpha=0.85)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f'{w:.4f}', va='center', ha='left', fontsize=10, fontweight='bold')

plt.xlim(0, 1.0)
plt.title('All 13 Machine Learning Models Ranked by PR-AUC', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('PR-AUC Score (Area Under Precision-Recall Curve)', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.tight_layout()
pr_plot_path = 'plots/all_models_pr_auc_comparison.png'
plt.savefig(pr_plot_path, dpi=300)
plt.close()
print(f"Saved PR-AUC comparison plot: {pr_plot_path}")

# 2. Precision vs Recall Trade-off Scatter Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=bench_df,
    x='Recall',
    y='Precision',
    hue='Model',
    s=140,
    palette='tab20',
    edgecolor='black'
)
for _, row in bench_df.iterrows():
    plt.text(row['Recall'] + 0.008, row['Precision'] + 0.01, row['Model'].split('.')[1].strip().split('(')[0], fontsize=9)

plt.title('Precision vs Recall Trade-off Across All Models', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Recall (Fraud Catch Rate)', fontsize=12)
plt.ylabel('Precision (Alert Correctness)', fontsize=12)
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
plt.tight_layout()
scatter_path = 'plots/all_models_tradeoff.png'
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Precision-Recall Trade-off plot: {scatter_path}")

print("\nBenchmark of all models completed successfully!")
