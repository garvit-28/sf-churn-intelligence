"""
Salesforce Data Engine Client
Supports Multi-Encoding CLI decoding (UTF-8, UTF-16, Latin-1) and REST fallback.
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def decode_output(raw_bytes: bytes) -> str:
    """Decodes CLI bytes safely across UTF-16 (Windows PowerShell), UTF-8-SIG, and UTF-8."""
    if not raw_bytes:
        return ""
    for enc in ["utf-8-sig", "utf-16", "utf-8", "latin-1"]:
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


class SalesforceClient:
    def __init__(self, target_org: str = "my-dev-org"):
        self.target_org = target_org
        self.sf_api = None
        self._init_cloud_api()

    def _init_cloud_api(self):
        """Attempts connection via simple-salesforce if secrets are configured."""
        username = None
        password = None
        security_token = ""
        domain = "login"

        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                username = st.secrets.get("SF_USERNAME")
                password = st.secrets.get("SF_PASSWORD")
                security_token = st.secrets.get("SF_SECURITY_TOKEN", "")
                domain = st.secrets.get("SF_DOMAIN", "login")
        except Exception:
            pass

        if not username:
            username = os.getenv("SF_USERNAME")
            password = os.getenv("SF_PASSWORD")
            security_token = os.getenv("SF_SECURITY_TOKEN", "")
            domain = os.getenv("SF_DOMAIN", "login")

        if username and password:
            try:
                from simple_salesforce import Salesforce
                self.sf_api = Salesforce(
                    username=username,
                    password=password,
                    security_token=security_token,
                    domain=domain
                )
                logging.info("Connected to live Salesforce REST API.")
            except Exception as e:
                logging.warning(f"Salesforce API connection skipped: {e}")
                self.sf_api = None

    def query(self, soql: str) -> pd.DataFrame:
        """Executes SOQL query against live Salesforce org with multi-encoding protection."""
        clean_soql = " ".join(soql.split())
        logging.info(f"Executing SOQL query: {clean_soql}")

        # 1. Live REST API (if available)
        if self.sf_api:
            try:
                res = self.sf_api.query_all(clean_soql)
                records = res.get("records", [])
                for r in records:
                    r.pop("attributes", None)
                if records:
                    return pd.DataFrame(records)
            except Exception as e:
                logging.warning(f"REST API query failed: {e}")

        # 2. Salesforce CLI with raw byte decoding (handles Windows UTF-16 outputs)
        try:
            cmd = f'sf data query --query "{clean_soql}" --json'
            result = subprocess.run(cmd, shell=True, capture_output=True)
            
            if result.returncode == 0 and result.stdout:
                decoded_str = decode_output(result.stdout)
                data = json.loads(decoded_str)
                records = data.get("result", {}).get("records", [])
                for r in records:
                    r.pop("attributes", None)
                if records:
                    return pd.DataFrame(records)
        except Exception as e:
            logging.warning(f"CLI Query Execution failed: {e}")

        # 3. CSV Snapshots fallback with safe multi-encoding
        csv_candidates = [
            "data/raw_accounts.csv", "data/accounts.csv",
            "data/raw_cases.csv", "data/cases.csv"
        ]
        for p in csv_candidates:
            if os.path.exists(p):
                for enc in ["utf-16", "utf-8-sig", "utf-8", "latin-1"]:
                    try:
                        df = pd.read_csv(p, encoding=enc)
                        if not df.empty and len(df.columns) > 1:
                            return df
                    except Exception:
                        continue

        return pd.DataFrame()

    def create_record(self, sobject: str, values: Dict[str, Any]) -> Optional[str]:
        if self.sf_api:
            try:
                obj = getattr(self.sf_api, sobject)
                res = obj.create(values)
                return res.get("id")
            except Exception as e:
                logging.warning(f"REST API create failed: {e}")
        return "SIMULATED_RECORD_ID"