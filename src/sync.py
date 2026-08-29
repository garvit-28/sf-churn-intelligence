"""
Reverse-ETL & Inference Sync Module
Pushes ML churn predictions and SHAP explainability drivers into live Salesforce Account records.
"""

import os
import pandas as pd
from sf_client import SalesforceClient

def sync_predictions_to_salesforce():
    scored_path = "data/scored_accounts.csv"
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"Scored dataset missing at {scored_path}. Run src/train.py first.")
        
    df_scored = pd.read_csv(scored_path)
    sf = SalesforceClient()
    
    print("--- Fetching Live Account Records from Salesforce ---")
    query = "SELECT Id, Name, Churn_Risk_Score__c, Risk_Level__c, Top_Churn_Driver__c FROM Account"
    live_accounts = sf.query(query)
    
    if live_accounts.empty:
        print("No live Account records found in Salesforce org.")
        return
        
    print(f"Found {len(live_accounts)} live Account records in org.")
    
    # Merge live SF records with model predictions based on Account Name
    sync_df = pd.merge(
        live_accounts[["Id", "Name"]],
        df_scored[["Name", "Predicted_Churn_Prob", "Predicted_Risk_Level", "Top_Churn_Driver"]],
        on="Name",
        how="inner"
    )
    
    if sync_df.empty:
        print("No matching account names between local predictions and live Salesforce records.")
        return
        
    print(f"\n--- Syncing {len(sync_df)} Account Predictions Back to Salesforce CRM ---")
    
    success_count = 0
    for _, row in sync_df.iterrows():
        record_id = row["Id"]
        acc_name = row["Name"]
        
        update_payload = {
            "Churn_Risk_Score__c": float(row["Predicted_Churn_Prob"]),
            "Risk_Level__c": str(row["Predicted_Risk_Level"]),
            "Top_Churn_Driver__c": str(row["Top_Churn_Driver"])
        }
        
        try:
            sf.update_record("Account", record_id, update_payload)
            print(f"Synced {acc_name} ({record_id}) -> Risk: {row['Predicted_Risk_Level']} | Score: {row['Predicted_Churn_Prob']} | Driver: {row['Top_Churn_Driver']}")
            success_count += 1
        except Exception as e:
            print(f"Failed to sync {acc_name}: {e}")
            
    print(f"\nReverse-ETL completed successfully: {success_count}/{len(sync_df)} accounts updated in Salesforce.")


if __name__ == "__main__":
    sync_predictions_to_salesforce()