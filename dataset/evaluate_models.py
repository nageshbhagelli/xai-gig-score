import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
import os

print("Loading dataset...")
df = pd.read_csv('realistic_gig_worker_dataset.csv')

# Feature engineering
df['risk_segment'] = 'Medium'
df.loc[(df['volatility_ratio'] > 1.5) | (df['complaints_30d'] > 3), 'risk_segment'] = 'High Risk'
df.loc[(df['reliability_score'] > 0.8) & (df['active_days_30d'] > 25), 'risk_segment'] = 'Low Risk'

df['financial_stress_ratio'] = df['volatility_ratio'] / (df['net_payout_30d'] / 1000 + 1)
df['earnings_consistency'] = df['active_days_30d'] * (1 / (df['volatility_ratio'] + 1))

# Select features
X = df.drop(columns=['Worker ID', 'default_prob', 'default', 'credit_score', 'risk_segment'])
y = df['default']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train_scaled, X_test_scaled, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

def get_optimal_threshold(y_true, y_prob, target_recall=0.80):
    thresholds = np.arange(0.01, 0.99, 0.01)
    for thresh in sorted(thresholds, reverse=True):
        preds = (y_prob >= thresh).astype(int)
        rec = recall_score(y_true, preds)
        if rec >= target_recall:
            return float(thresh)
    return 0.1

def evaluate_model(name, model, X_test, y_test, X_train=None, y_train=None):
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    
    best_thresh = 0.5
    if X_train is not None and y_train is not None and hasattr(model, "predict_proba"):
        y_train_prob = model.predict_proba(X_train)[:, 1]
        best_thresh = get_optimal_threshold(y_train, y_train_prob)
        print(f"Optimal Threshold for {name}: {best_thresh:.2f}")
    
    y_pred = (y_prob >= best_thresh).astype(int) if hasattr(model, "predict_proba") else model.predict(X_test)
    
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision (Class 1):", precision_score(y_test, y_pred))
    print("Recall (Class 1):", recall_score(y_test, y_pred))
    print("F1 (Class 1):", f1_score(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print(classification_report(y_test, y_pred))
    print()

print("Training Improved Logistic Regression (class_weight='balanced')...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
evaluate_model("Improved Logistic Regression", lr_model, X_test_scaled, y_test, X_train_scaled, y_train)

print("Training Improved Random Forest (class_weight='balanced')...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)
evaluate_model("Improved Random Forest", rf_model, X_test, y_test, X_train, y_train)

print("Training Improved XGBoost (scale_pos_weight)...")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42, max_depth=5, learning_rate=0.05, n_estimators=200)
xgb_model.fit(X_train, y_train)
evaluate_model("Improved XGBoost", xgb_model, X_test, y_test, X_train, y_train)

print("--- Generating evaluation graphs ---")
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Get optimal threshold for Random Forest graph
y_train_prob_rf = rf_model.predict_proba(X_train)[:, 1]
best_thresh_rf = get_optimal_threshold(y_train, y_train_prob_rf)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_pred_rf = (y_prob_rf >= best_thresh_rf).astype(int)

print("Generating ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#2563eb', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.savefig('graph1_roc_curve.png', dpi=300)
plt.close()

print("Generating Precision-Recall Curve...")
precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)
pr_auc = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='#16a34a', lw=2, label=f'Random Forest (PR-AUC = {pr_auc:.3f})')
plt.xlabel('Recall (True Positive Rate)', fontsize=12)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, pad=15)
plt.legend(loc="lower left", fontsize=11)
plt.tight_layout()
plt.savefig('graph2_pr_curve.png', dpi=300)
plt.close()

print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7, 6))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                 xticklabels=['Non-Defaulter', 'Defaulter'], 
                 yticklabels=['Non-Defaulter', 'Defaulter'],
                 annot_kws={"size": 14})
plt.xlabel('Predicted Class', fontsize=12, labelpad=10)
plt.ylabel('Actual Class', fontsize=12, labelpad=10)
plt.title(f'Confusion Matrix (Random Forest, Thresh={best_thresh_rf:.2f})', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('graph3_confusion_matrix.png', dpi=300)
plt.close()

print("Generating Feature Importance Bar Chart...")
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette='viridis', orient='h')
plt.xlabel('Relative Importance (Gini)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Feature Importances in Risk Prediction', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('graph4_feature_importance.png', dpi=300)
plt.close()

print("Graphs successfully generated and saved in the dataset directory.")
