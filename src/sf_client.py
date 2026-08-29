"""
Salesforce Data Engine Client
Provides an abstraction layer for SOQL queries and record mutations via SF CLI.
"""

import subprocess
import json
import logging
from typing import Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SalesforceClient:
    def __init__(self, target_org: str = "my-dev-org"):
        self.target_org = target_org

    def query(self, soql: str) -> pd.DataFrame:
        """Executes a SOQL query and returns records as a pandas DataFrame."""
        logging.info(f"Executing SOQL query: {soql}")
        clean_soql = " ".join(soql.split())
        cmd = f'sf data query --target-org {self.target_org} --query "{clean_soql}" --json'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Salesforce query error: {result.stderr or result.stdout}")
            
        data = json.loads(result.stdout)
        records = data.get("result", {}).get("records", [])
        
        for r in records:
            r.pop("attributes", None)
            
        logging.info(f"Query returned {len(records)} record(s).")
        return pd.DataFrame(records)

    def _format_values(self, values: Dict[str, Any]) -> str:
        """Formats dictionary values safely for SF CLI --values parameter."""
        formatted_pairs = []
        for k, v in values.items():
            if isinstance(v, str):
                # Escape internal single quotes and enclose in single quotes
                safe_val = v.replace("'", "\\'")
                formatted_pairs.append(f"{k}='{safe_val}'")
            elif v is None:
                continue
            else:
                formatted_pairs.append(f"{k}={v}")
        return " ".join(formatted_pairs)

    def create_record(self, sobject: str, values: Dict[str, Any]) -> str:
        """Inserts a new sObject record and returns the generated Salesforce Id."""
        val_str = self._format_values(values)
        cmd = f'sf data create record --target-org {self.target_org} --sobject {sobject} --values "{val_str}" --json'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Salesforce insert error on {sobject}: {result.stderr or result.stdout}")
            
        data = json.loads(result.stdout)
        record_id = data.get("result", {}).get("id")
        return record_id

    def update_record(self, sobject: str, record_id: str, values: Dict[str, Any]) -> bool:
        """Updates fields on an existing sObject record by Id."""
        val_str = self._format_values(values)
        cmd = f'sf data update record --target-org {self.target_org} --sobject {sobject} --record-id {record_id} --values "{val_str}" --json'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Salesforce update error on {sobject} (ID: {record_id}): {result.stderr or result.stdout}")
            
        return True


if __name__ == "__main__":
    sf = SalesforceClient()
    df = sf.query("SELECT Id, Name FROM Account LIMIT 2")
    print("\nsf_client.py self-test passed:")
    print(df)