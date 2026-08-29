"""
Model Training & Explainability Engine
Trains an XGBoost Churn Classifier and generates per-account SHAP drivers.
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

FEATURE_COLS = [
    "Tenure_Months",
    "Monthly_Charges",
    "Total_Charges",
    "Total_Cases",
    "Escalated_Cases",
    "Avg_Days_To_Resolve",
    "Critical_Cases",
    "High_Cases",
    "Escalation_Rate",
    "Avg_Monthly_Case_Load",
    "Est_Annual_Value",
]

CATEGORICAL_COLS = ["Industry", "Contract_Type"]


def extract_top_driver(shap_row: np.ndarray, feature_names: list) -> str:
    """Extracts the strongest positive SHAP contributor pushing the record toward churn."""
    max_feat_idx = int(np.argmax(shap_row))
    max_val = shap_row[max_feat_idx]
    
    if max_val <= 0:
        return "Stable Account / High Engagement"
        
    feat_name = feature_names[max_feat_idx]
    clean_name = feat_name.replace("_", " ").title()
    return f"Elevated {clean_name}"


def train_churn_model():
    data_path = "data/features_engineered.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found at {data_path}. Run src/etl.py first.")
        
    df = pd.read_csv(data_path)
    print(f"--- Training XGBoost Churn Model on {len(df)} Records ---")
    
    # Preprocessing & One-Hot Encoding
    X_encoded = pd.get_dummies(df[FEATURE_COLS + CATEGORICAL_COLS], drop_first=True)
    y = df["True_Churn_Label"]
    
    feature_names = list(X_encoded.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost Classifier
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on Holdout Test Set
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, probs)
    print(f"\nModel Performance on Test Set:")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    # SHAP Explainability Engine
    print("\nComputing TreeSHAP values for inference explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_encoded)
    
    # Generate inference predictions across all accounts
    full_probs = model.predict_proba(X_encoded)[:, 1]
    
    top_drivers = []
    for i in range(len(df)):
        driver = extract_top_driver(shap_values[i], feature_names)
        top_drivers.append(driver)
        
    df["Predicted_Churn_Prob"] = np.round(full_probs, 2)
    df["Predicted_Risk_Level"] = pd.cut(
        df["Predicted_Churn_Prob"],
        bins=[-0.01, 0.35, 0.70, 1.0],
        labels=["Low", "Medium", "High"]
    )
    df["Top_Churn_Driver"] = top_drivers
    
    # Save Model Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, "models/xgb_churn_model.joblib")
    
    # Save Scored Predictions
    df.to_csv("data/scored_accounts.csv", index=False)
    print("Scored dataset saved to data/scored_accounts.csv")
    print("Model serialized to models/xgb_churn_model.joblib")
    
    print("\nSample Scored Output (Top 5 Accounts):")
    cols = ["Name", "Predicted_Churn_Prob", "Predicted_Risk_Level", "Top_Churn_Driver"]
    print(df[cols].head())


if __name__ == "__main__":
    train_churn_model()