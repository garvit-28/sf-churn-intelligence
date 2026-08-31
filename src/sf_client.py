"""
Salesforce REST Client (Direct Session Authentication)
Safe, lightweight, and bypasses all SOAP / Connected App restrictions.
"""

import os
import logging
import pandas as pd
from simple_salesforce import Salesforce

logger = logging.getLogger(__name__)


class SalesforceClient:
    def __init__(self):
        self.instance_url = os.getenv("SF_INSTANCE_URL")
        self.session_id = os.getenv("SF_SESSION_ID")

        if not self.instance_url or not self.session_id:
            raise ValueError("Missing credentials. Please set SF_INSTANCE_URL and SF_SESSION_ID.")

        try:
            self.sf = Salesforce(
                instance_url=self.instance_url.rstrip("/"),
                session_id=self.session_id
            )
            logger.info("Connected to Salesforce REST API successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Salesforce: {e}")
            raise

    def query(self, soql: str) -> pd.DataFrame:
        """Executes SOQL and returns results as a DataFrame."""
        try:
            result = self.sf.query_all(soql)
            records = result.get("records", [])
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.drop(columns=["attributes"], errors="ignore", inplace=True)
            return df
        except Exception as e:
            logger.error(f"SOQL execution failed: {e}")
            raise

    def create_record(self, object_name: str, payload: dict) -> str:
        """Inserts a record into Salesforce and returns the ID."""
        try:
            res = getattr(self.sf, object_name).create(payload)
            if res.get("success"):
                return res.get("id")
            raise Exception(f"Failed to create {object_name}: {res.get('errors')}")
        except Exception as e:
            logger.error(f"Error creating {object_name}: {e}")
            raise

    def update_record(self, object_name: str, record_id: str, payload: dict) -> bool:
        """Updates an existing record in Salesforce by Record ID."""
        try:
            getattr(self.sf, object_name).update(record_id, payload)
            logger.info(f"Updated {object_name} {record_id} successfully.")
            return True
        except Exception as e:
            logger.error(f"Error updating {object_name} {record_id}: {e}")
            raise