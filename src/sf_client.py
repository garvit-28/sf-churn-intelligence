"""
Salesforce REST & Bulk API Client Wrapper
Handles authentication, dynamic SOQL queries, and DML operations (Create, Update).
"""

import os
import logging
import pandas as pd
from simple_salesforce import Salesforce, SalesforceAuthenticationFailed

logger = logging.getLogger(__name__)


class SalesforceClient:
    def __init__(self):
        self.username = os.getenv("SF_USERNAME")
        self.password = os.getenv("SF_PASSWORD")
        self.security_token = os.getenv("SF_SECURITY_TOKEN")
        self.domain = os.getenv("SF_DOMAIN", "login")  # 'login' for prod/developer orgs, 'test' for sandboxes

        if not all([self.username, self.password, self.security_token]):
            raise ValueError(
                "Salesforce credentials missing. Set SF_USERNAME, SF_PASSWORD, and SF_SECURITY_TOKEN in environment variables."
            )

        try:
            self.sf = Salesforce(
                username=self.username,
                password=self.password,
                security_token=self.security_token,
                domain=self.domain
            )
            logger.info("Connected to Salesforce successfully.")
        except SalesforceAuthenticationFailed as e:
            logger.error(f"Salesforce authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Salesforce client: {e}")
            raise

    def query(self, soql: str) -> pd.DataFrame:
        """
        Executes SOQL and returns results as a clean Pandas DataFrame.
        Automatically strips Salesforce record metadata attributes.
        """
        try:
            result = self.sf.query_all(soql)
            records = result.get("records", [])
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.drop(columns=["attributes"], errors="ignore", inplace=True)
            return df
        except Exception as e:
            logger.error(f"SOQL execution failed for query '{soql}': {e}")
            raise

    def create_record(self, object_name: str, payload: dict) -> str:
        """
        Inserts a record into Salesforce and returns the new Record ID.
        """
        try:
            res = getattr(self.sf, object_name).create(payload)
            if res.get("success"):
                record_id = res.get("id")
                logger.info(f"Created {object_name} with ID: {record_id}")
                return record_id
            raise Exception(f"Failed to create {object_name}: {res.get('errors')}")
        except Exception as e:
            logger.error(f"Error creating {object_name}: {e}")
            raise

    def update_record(self, object_name: str, record_id: str, payload: dict) -> bool:
        """
        Updates an existing record in Salesforce by Record ID.
        """
        try:
            getattr(self.sf, object_name).update(record_id, payload)
            logger.info(f"Updated {object_name} {record_id} successfully.")
            return True
        except Exception as e:
            logger.error(f"Error updating {object_name} {record_id}: {e}")
            raise