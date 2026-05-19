import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score
import joblib
import json
import os

print("Loading dataset...")
df = pd.read_csv('realistic_gig_worker_dataset.csv')

print("Performing feature engineering...")
# Replicating notebook logic
df['risk_segment'] = 'Medium'
df.loc[(df['volatility_ratio'] > 1.5) | (df['complaints_30d'] > 3), 'risk_segment'] = 'High Risk'
df.loc[(df['reliability_score'] > 0.8) & (df['active_days_30d'] > 25), 'risk_segment'] = 'Low Risk'

df['financial_stress_ratio'] = df['volatility_ratio'] / (df['net_payout_30d'] / 1000 + 1)
df['earnings_consistency'] = df['active_days_30d'] * (1 / (df['volatility_ratio'] + 1))

# Select features
X = df.drop(columns=['Worker ID', 'default_prob', 'default', 'credit_score', 'risk_segment'])
y = df['default']

print(f"Feature matrix shape: {X.shape}")

# Scaling (matching notebook's fit on entire X before split)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_scaled, X_test_scaled, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print("Training Logistic Regression (class_weight='balanced')...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)

print("Training Random Forest (class_weight='balanced')...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

print("Training XGBoost...")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42, max_depth=5, learning_rate=0.05, n_estimators=200)
xgb_model.fit(X_train, y_train)

def get_optimal_threshold(y_true, y_prob, target_recall=0.80):
    thresholds = np.arange(0.01, 0.99, 0.01)
    for thresh in sorted(thresholds, reverse=True):
        preds = (y_prob >= thresh).astype(int)
        rec = recall_score(y_true, preds)
        if rec >= target_recall:
            return float(thresh)
    return 0.1

print("Calculating optimal thresholds...")
thresholds = {
    'logistic_regression': get_optimal_threshold(y_train, lr_model.predict_proba(X_train_scaled)[:, 1]),
    'random_forest': get_optimal_threshold(y_train, rf_model.predict_proba(X_train)[:, 1]),
    'xgboost': get_optimal_threshold(y_train, xgb_model.predict_proba(X_train)[:, 1])
}

print("Saving models, scaler, and thresholds...")
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

with open('thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=4)

print("All models exported successfully:")
print("- logistic_regression_model.pkl")
print("- random_forest_model.pkl")
print("- xgboost_model.pkl")
print("- scaler.pkl")
print("- thresholds.json")
