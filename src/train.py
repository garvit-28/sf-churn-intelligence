"""
Model Training & Explainability Engine

Trains the customer churn XGBoost model using the engineered dataset,
saves the preprocessing + model together, evaluates the model,
and generates SHAP-based churn drivers.
"""

import os
import joblib
import pandas as pd
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ============================================================
# 1. FEATURES USED BY THE MODEL
# ============================================================

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "Industry",
    "Contract_Type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 2. RISK CLASSIFICATION
# ============================================================

def assign_risk_level(probability):
    """
    Converts churn probability into a business-friendly risk level.
    """

    if probability >= 0.70:
        return "High"

    elif probability >= 0.35:
        return "Medium"

    else:
        return "Low"


# ============================================================
# 3. TRAINING FUNCTION
# ============================================================

def train_churn_model():

    data_path = "data/features_engineered.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Feature dataset not found at {data_path}. "
            "Run src/etl.py first."
        )

    # --------------------------------------------------------
    # Load engineered data
    # --------------------------------------------------------

    df = pd.read_csv(data_path)

    print(
        f"--- Training XGBoost Churn Model "
        f"on {len(df)} Records ---"
    )

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    X = df[ALL_FEATURES].copy()

    y = df["True_Churn_Label"]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES
            ),

            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    # --------------------------------------------------------
    # XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    # --------------------------------------------------------
    # Combine preprocessing + model
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining model...")

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nModel Performance on Test Set:")
    print(
        f"ROC-AUC Score: {roc_auc:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # ========================================================
    # 4. GET TRANSFORMED FEATURE NAMES
    # ========================================================

    fitted_preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    feature_names = (
        fitted_preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        f"\nTotal model features after encoding: "
        f"{len(feature_names)}"
    )

    # ========================================================
    # 5. SHAP EXPLAINABILITY
    # ========================================================

    print("\nGenerating SHAP explanations...")

    # Transform the complete dataset using
    # the SAME preprocessing used during training.

    X_encoded = fitted_preprocessor.transform(X)

    trained_model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(
        trained_model
    )

    shap_values = explainer.shap_values(
        X_encoded
    )

    # ========================================================
    # 6. FIND TOP CHURN DRIVER
    # ========================================================

    top_drivers = []

    for row in shap_values:

        # Find the feature with the largest
        # positive contribution toward churn.

        max_index = int(
            np.argmax(row)
        )

        max_value = row[max_index]

        if max_value <= 0:

            driver = "No Strong Positive Churn Driver"

        else:

            driver_name = feature_names[
                max_index
            ]

            # Remove preprocessing prefixes
            # such as numeric__ or categorical__.

            driver_name = driver_name.replace(
                "numeric__",
                ""
            )

            driver_name = driver_name.replace(
                "categorical__",
                ""
            )

            driver_name = driver_name.replace(
                "_",
                " "
            ).title()

            driver = (
                f"Elevated {driver_name}"
            )

        top_drivers.append(
            driver
        )

    # ========================================================
    # 7. SCORE ALL ACCOUNTS
    # ========================================================

    full_probabilities = (
        pipeline.predict_proba(X)[:, 1]
    )

    df["Predicted_Churn_Prob"] = (
        np.round(
            full_probabilities,
            4
        )
    )

    df["Predicted_Risk_Level"] = (
        df["Predicted_Churn_Prob"]
        .apply(assign_risk_level)
    )

    df["Top_Churn_Driver"] = top_drivers

    # ========================================================
    # 8. SAVE MODEL ARTIFACT
    # ========================================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_artifact = {
        "pipeline": pipeline,
        "model": trained_model,
        "explainer": explainer,
        "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "risk_thresholds": {
            "low": 0.35,
            "high": 0.70
        }
    }

    model_path = (
        "models/churn_model_artifact.joblib"
    )

    joblib.dump(
        model_artifact,
        model_path
    )

    # ========================================================
    # 9. SAVE SCORED DATA
    # ========================================================

    scored_path = (
        "data/scored_accounts.csv"
    )

    df.to_csv(
        scored_path,
        index=False
    )

    # ========================================================
    # 10. OUTPUT
    # ========================================================

    print(
        f"\nScored dataset saved to "
        f"{scored_path}"
    )

    print(
        f"Model artifact saved to "
        f"{model_path}"
    )

    print(
        "\nSample Scored Output:"
    )

    print(
        df[
            [
                "Name",
                "Predicted_Churn_Prob",
                "Predicted_Risk_Level",
                "Top_Churn_Driver"
            ]
        ].head()
    )

    return model_artifact


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_churn_model()