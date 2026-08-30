"""
Salesforce Data Engine Client
Provides a multi-tier abstraction layer for SOQL queries and mutations:
1. Streamlit Secrets / Environment Variables (Cloud Native via simple_salesforce)
2. Local SF CLI (Local Development)
3. CSV Offline Fallback (Zero-Config Demo Mode)
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SalesforceClient:
    def __init__(self, target_org: str = "my-dev-org"):
        self.target_org = target_org
        self.sf_api = None
        self._init_cloud_api()

    def _init_cloud_api(self):
        """Attempts to initialize simple-salesforce from Streamlit Secrets or Environment Variables."""
        credentials = {}

        # 1. Try reading from Streamlit Secrets (Cloud runtime)
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                credentials = {
                    "username": st.secrets.get("SF_USERNAME"),
                    "password": st.secrets.get("SF_PASSWORD"),
                    "security_token": st.secrets.get("SF_SECURITY_TOKEN"),
                    "domain": st.secrets.get("SF_DOMAIN", "login")
                }
        except Exception:
            pass

        # 2. Try standard environment variables (Local / Docker)
        if not credentials.get("username"):
            credentials = {
                "username": os.getenv("SF_USERNAME"),
                "password": os.getenv("SF_PASSWORD"),
                "security_token": os.getenv("SF_SECURITY_TOKEN"),
                "domain": os.getenv("SF_DOMAIN", "login")
            }

        # If credentials exist, instantiate simple_salesforce
        if credentials.get("username") and credentials.get("password"):
            try:
                from simple_salesforce import Salesforce
                self.sf_api = Salesforce(
                    username=credentials["username"],
                    password=credentials["password"],
                    security_token=credentials.get("security_token") or "",
                    domain=credentials.get("domain", "login")
                )
                logging.info("Initialized Salesforce connection via simple-salesforce REST API.")
            except Exception as e:
                logging.warning(f"Failed to authenticate via simple-salesforce: {e}")
                self.sf_api = None

    def query(self, soql: str) -> pd.DataFrame:
        """Executes a SOQL query using the best available channel."""
        clean_soql = " ".join(soql.split())
        logging.info(f"Executing SOQL query: {clean_soql}")

        # Tier 1: Cloud REST API
        if self.sf_api:
            try:
                res = self.sf_api.query_all(clean_soql)
                records = res.get("records", [])
                for r in records:
                    r.pop("attributes", None)
                logging.info(f"REST API query returned {len(records)} record(s).")
                return pd.DataFrame(records)
            except Exception as e:
                logging.warning(f"REST API query failed: {e}. Falling back...")

        # Tier 2: Local Salesforce CLI
        try:
            cmd = f'sf data query --target-org {self.target_org} --query "{clean_soql}" --json'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                records = data.get("result", {}).get("records", [])
                for r in records:
                    r.pop("attributes", None)
                logging.info(f"SF CLI query returned {len(records)} record(s).")
                return pd.DataFrame(records)
        except Exception:
            pass

        # Tier 3: Demo Mode / Offline CSV Fallback
        logging.info("SF connection unavailable. Loading fallback snapshot CSV.")
        return self._load_csv_fallback(clean_soql)

    def _load_csv_fallback(self, soql: str) -> pd.DataFrame:
        """Loads static CSV data if cloud and CLI connections are unreachable."""
        soql_lower = soql.lower()
        
        # Determine whether query is asking for Account or Case
        if "from account" in soql_lower:
            candidates = ["data/raw_accounts.csv", "data/accounts.csv", "accounts.csv", "data/salesforce_data.csv"]
        elif "from case" in soql_lower:
            candidates = ["data/raw_cases.csv", "data/cases.csv", "cases.csv", "data/case_data.csv"]
        else:
            candidates = ["data/salesforce_data.csv", "data/dataset.csv"]

        for path in candidates:
            if os.path.exists(path):
                logging.info(f"Loaded fallback data from: {path}")
                df = pd.read_csv(path)
                return df

        # If no CSV files are found, return an empty DataFrame
        logging.warning("No fallback CSV files found.")
        return pd.DataFrame()

    def _format_values(self, values: Dict[str, Any]) -> str:
        """Formats dictionary values safely for SF CLI --values parameter."""
        formatted_pairs = []
        for k, v in values.items():
            if isinstance(v, str):
                safe_val = v.replace("'", "\\'")
                formatted_pairs.append(f"{k}='{safe_val}'")
            elif v is None:
                continue
            else:
                formatted_pairs.append(f"{k}={v}")
        return " ".join(formatted_pairs)

    def create_record(self, sobject: str, values: Dict[str, Any]) -> Optional[str]:
        """Inserts a new sObject record and returns the generated Salesforce Id."""
        if self.sf_api:
            try:
                obj = getattr(self.sf_api, sobject)
                res = obj.create(values)
                return res.get("id")
            except Exception as e:
                logging.warning(f"REST API create failed: {e}")

        # Local CLI Fallback
        try:
            val_str = self._format_values(values)
            cmd = f'sf data create record --target-org {self.target_org} --sobject {sobject} --values "{val_str}" --json'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("result", {}).get("id")
        except Exception:
            pass

        logging.info("Demo Mode: Record creation simulated successfully.")
        return "DEMO_RECORD_ID_001"

    def update_record(self, sobject: str, record_id: str, values: Dict[str, Any]) -> bool:
        """Updates fields on an existing sObject record by Id."""
        if self.sf_api:
            try:
                obj = getattr(self.sf_api, sobject)
                obj.update(record_id, values)
                return True
            except Exception as e:
                logging.warning(f"REST API update failed: {e}")

        # Local CLI Fallback
        try:
            val_str = self._format_values(values)
            cmd = f'sf data update record --target-org {self.target_org} --sobject {sobject} --record-id {record_id} --values "{val_str}" --json'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except Exception:
            pass

        logging.info(f"Demo Mode: Record {record_id} update simulated successfully.")
        return True


if __name__ == "__main__":
    sf = SalesforceClient()
    df = sf.query("SELECT Id, Name FROM Account LIMIT 2")
    print("\nsf_client.py test output:")
    print(df.head())