"""
Data Generation & Salesforce Seeding Module
Generates synthetic multi-touch B2B SaaS customer data and seeds Account & Case records into Salesforce.
"""

import os
import random
import pandas as pd
import numpy as np
from sf_client import SalesforceClient

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_saas_dataset(n_accounts: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic B2B SaaS customer profiles and historical support cases.
    """
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Education"]
    company_prefixes = ["Apex", "Vertex", "Nova", "Pulse", "Quantum", "Synergy", "Cloud", "Nexus", "Stratum", "Beacon"]
    company_suffixes = ["Analytics", "Systems", "Health", "Logistics", "Software", "Tech", "Ventures", "Solutions", "Media", "Labs"]
    contract_types = ["Month-to-Month", "One Year", "Two Year"]
    
    accounts = []
    cases = []
    
    for i in range(1, n_accounts + 1):
        account_id = f"ACC-{i:04d}"
        company_name = f"{random.choice(company_prefixes)} {random.choice(company_suffixes)} {i}"
        industry = random.choice(industries)
        tenure_months = int(np.random.gamma(shape=2.5, scale=8.0)) + 1
        monthly_charges = round(float(np.random.normal(loc=2500, scale=800)), 2)
        monthly_charges = max(499.0, monthly_charges)
        total_charges = round(monthly_charges * tenure_months * random.uniform(0.9, 1.05), 2)
        contract = random.choices(contract_types, weights=[0.45, 0.35, 0.20])[0]
        
        # Calculate latent churn probability based on SaaS business logic
        churn_logits = (
            (1.5 if contract == "Month-to-Month" else -0.8) +
            (0.0006 * monthly_charges) -
            (0.06 * tenure_months) +
            (0.5 if industry in ["Retail", "Education"] else -0.2)
        )
        churn_prob = 1 / (1 + np.exp(-churn_logits))
        churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1), 0.02, 0.98)
        
        # Generate correlated support ticket frequency
        case_lambda = 4.5 if churn_prob > 0.6 else (2.0 if churn_prob > 0.3 else 0.8)
        num_cases = np.random.poisson(lam=case_lambda)
        
        # Adjust churn probability with case interaction
        if num_cases >= 4:
            churn_prob = min(0.99, churn_prob + 0.15)
            
        is_churned = int(np.random.binomial(1, churn_prob))
        
        accounts.append({
            "Account_Ref": account_id,
            "Name": company_name,
            "Industry": industry,
            "Tenure_Months": tenure_months,
            "Monthly_Charges": monthly_charges,
            "Total_Charges": total_charges,
            "Contract_Type": contract,
            "True_Churn_Label": is_churned,
            "Num_Cases": num_cases
        })
        
        # Generate individual Case records
        for _ in range(num_cases):
            case_type = random.choices(
                ["Billing", "Technical / Bug", "Feature Request", "Performance / Latency", "Onboarding"],
                weights=[0.35, 0.30, 0.15, 0.15, 0.05] if is_churned else [0.15, 0.35, 0.30, 0.10, 0.10]
            )[0]
            
            is_escalated = random.random() < (0.45 if is_churned else 0.10)
            priority = random.choices(["Low", "Medium", "High", "Critical"], weights=[0.1, 0.3, 0.4, 0.2] if is_escalated else [0.3, 0.5, 0.18, 0.02])[0]
            days_open = int(np.random.exponential(scale=12.0)) if is_escalated else int(np.random.exponential(scale=3.0))
            
            cases.append({
                "Account_Ref": account_id,
                "Case_Type": case_type,
                "Priority": priority,
                "Is_Escalated": int(is_escalated),
                "Days_To_Resolve": days_open,
                "Status": "Escalated" if is_escalated else random.choices(["Closed", "Working", "New"], weights=[0.75, 0.18, 0.07])[0]
            })

    df_accounts = pd.DataFrame(accounts)
    df_cases = pd.DataFrame(cases)
    
    return df_accounts, df_cases


def seed_salesforce_org(df_accounts: pd.DataFrame, df_cases: pd.DataFrame, seed_count: int = 10):
    """
    Seeds a sample of accounts and related cases into the live Salesforce org.
    """
    sf = SalesforceClient()
    os.makedirs("data", exist_ok=True)
    
    print(f"\n--- Seeding {seed_count} Accounts & Related Cases to Salesforce Org ---")
    seeded_accounts = df_accounts.head(seed_count).copy()
    
    id_map = {}
    
    for _, row in seeded_accounts.iterrows():
        acc_values = {
            "Name": row["Name"],
            "Industry": row["Industry"],
            "Type": "Customer - Direct",
            "AnnualRevenue": int(row["Monthly_Charges"] * 12)
        }
        
        try:
            sf_id = sf.create_record("Account", acc_values)
            id_map[row["Account_Ref"]] = sf_id
            print(f"Created Account: {row['Name']} -> SF ID: {sf_id}")
        except Exception as e:
            print(f"Failed to create account {row['Name']}: {e}")

    # Seed related cases with mandatory Origin and boolean IsEscalated
    for _, row in df_cases[df_cases["Account_Ref"].isin(id_map.keys())].iterrows():
        sf_acc_id = id_map.get(row["Account_Ref"])
        if not sf_acc_id:
            continue
            
        case_values = {
            "AccountId": sf_acc_id,
            "Subject": f"{row['Case_Type']} Issue - Priority {row['Priority']}",
            "Priority": row["Priority"],
            "Status": row["Status"],
            "Type": row["Case_Type"],
            "Origin": "Web",  # Standard mandatory field in Salesforce
            "IsEscalated": "true" if bool(row["Is_Escalated"]) else "false"
        }
        
        try:
            sf.create_record("Case", case_values)
        except Exception as e:
            print(f"Warning: Could not create case for {sf_acc_id}: {e}")

    print("Seeding completed successfully.")


if __name__ == "__main__":
    print("Generating B2B SaaS dataset...")
    df_acc, df_cs = generate_saas_dataset(n_accounts=300)
    
    os.makedirs("data", exist_ok=True)
    df_acc.to_csv("data/accounts_raw.csv", index=False)
    df_cs.to_csv("data/cases_raw.csv", index=False)
    
    print(f"Generated {len(df_acc)} Accounts and {len(df_cs)} Support Cases.")
    print(f"Account Churn Distribution:\n{df_acc['True_Churn_Label'].value_counts(normalize=True).round(2)}")
    
    seed_salesforce_org(df_acc, df_cs, seed_count=10)