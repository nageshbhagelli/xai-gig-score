import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
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
