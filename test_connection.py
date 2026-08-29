import subprocess
import json
import pandas as pd

class SalesforceClient:
    def __init__(self, target_org: str = "my-dev-org"):
        self.target_org = target_org

    def query(self, soql: str) -> pd.DataFrame:
        """Executes SOQL queries via Salesforce CLI and returns a pandas DataFrame."""
        cmd = f'sf data query --target-org {self.target_org} --query "{soql}" --json'
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Salesforce query error: {result.stderr or result.stdout}")
            
        data = json.loads(result.stdout)
        records = data.get("result", {}).get("records", [])
        
        # Clean metadata attributes
        for r in records:
            r.pop("attributes", None)
            
        return pd.DataFrame(records)

    def update_record(self, sobject: str, record_id: str, values: dict) -> dict:
        """Updates fields on an sObject record."""
        # Formats key=value pairs for the CLI command
        val_str = " ".join([f'{k}="{v}"' for k, v in values.items()])
        cmd = f'sf data update record --target-org {self.target_org} --sobject {sobject} --record-id {record_id} --values {val_str} --json'
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Salesforce update error: {result.stderr or result.stdout}")
            
        return json.loads(result.stdout)


if __name__ == "__main__":
    try:
        print("Connecting to Salesforce via Native SF Data Engine...")
        sf = SalesforceClient(target_org="my-dev-org")

        print("Executing test SOQL query...")
        query = "SELECT Id, Name, Industry, Type FROM Account LIMIT 5"
        df = sf.query(query)

        print("\nRetrieved Records Successfully:")
        print(df)

    except Exception as e:
        print(f"\nExecution failed: {e}")