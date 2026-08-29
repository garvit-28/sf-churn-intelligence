"""
ETL & Feature Engineering Pipeline
Extracts raw Accounts and Cases, performs relational aggregation,
and builds an ML-ready feature matrix.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

def extract_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts raw account and case data from the data lake/directory."""
    accounts_path = "data/accounts_raw.csv"
    cases_path = "data/cases_raw.csv"
    
    if not os.path.exists(accounts_path) or not os.path.exists(cases_path):
        raise FileNotFoundError("Raw data files missing in data/. Run src/data_gen.py first.")
        
    df_accounts = pd.read_csv(accounts_path)
    df_cases = pd.read_csv(cases_path)
    return df_accounts, df_cases


def transform_features(df_accounts: pd.DataFrame, df_cases: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering and multi-touch aggregation across Accounts and Cases.
    """
    # 1. Aggregate Support Case Metrics per Account
    case_agg = df_cases.groupby("Account_Ref").agg(
        Total_Cases=("Case_Type", "count"),
        Escalated_Cases=("Is_Escalated", "sum"),
        Avg_Days_To_Resolve=("Days_To_Resolve", "mean"),
        Critical_Cases=("Priority", lambda x: (x == "Critical").sum()),
        High_Cases=("Priority", lambda x: (x == "High").sum())
    ).reset_index()

    # Calculate Escalated Case Ratio
    case_agg["Escalation_Rate"] = (
        case_agg["Escalated_Cases"] / case_agg["Total_Cases"]
    ).fillna(0.0).round(3)

    # 2. Relational Join: Merge Account Features with Case Aggregations
    df_merged = pd.merge(df_accounts, case_agg, on="Account_Ref", how="left")

    # Fill missing values for accounts with zero logged support cases
    df_merged["Total_Cases"] = df_merged["Total_Cases"].fillna(0).astype(int)
    df_merged["Escalated_Cases"] = df_merged["Escalated_Cases"].fillna(0).astype(int)
    df_merged["Avg_Days_To_Resolve"] = df_merged["Avg_Days_To_Resolve"].fillna(0.0).round(2)
    df_merged["Critical_Cases"] = df_merged["Critical_Cases"].fillna(0).astype(int)
    df_merged["High_Cases"] = df_merged["High_Cases"].fillna(0).astype(int)
    df_merged["Escalation_Rate"] = df_merged["Escalation_Rate"].fillna(0.0)

    # 3. Derived Business Features
    df_merged["Avg_Monthly_Case_Load"] = (
        df_merged["Total_Cases"] / df_merged["Tenure_Months"]
    ).round(3)

    df_merged["Est_Annual_Value"] = (
        df_merged["Monthly_Charges"] * 12
    ).round(2)

    return df_merged


def run_pipeline() -> pd.DataFrame:
    """Orchestrates extraction, transformation, and storage of the feature store."""
    print("--- Running ETL & Feature Engineering Pipeline ---")
    df_acc, df_cases = extract_raw_data()
    print(f"Extracted {len(df_acc)} Accounts and {len(df_cases)} Support Cases.")

    df_features = transform_features(df_acc, df_cases)
    
    os.makedirs("data", exist_ok=True)
    output_path = "data/features_engineered.csv"
    df_features.to_csv(output_path, index=False)
    
    print(f"Feature engineering complete. Saved {len(df_features)} records to {output_path}.\n")
    print("Sample Engineered Features (Top 5 Rows):")
    sample_cols = [
        "Account_Ref", "Name", "Tenure_Months", "Monthly_Charges",
        "Total_Cases", "Escalation_Rate", "Avg_Days_To_Resolve", "True_Churn_Label"
    ]
    print(df_features[sample_cols].head())
    
    return df_features


if __name__ == "__main__":
    run_pipeline()