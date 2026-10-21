
"""
Salesforce REST Client (OAuth Refresh Token Authentication)
Automatically obtains a fresh access token when the client is initialized.
"""

import os
import logging

import pandas as pd
import requests
from simple_salesforce import Salesforce
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SalesforceClient:
    def __init__(self):
        self.client_id = os.getenv("SF_CLIENT_ID")
        self.client_secret = os.getenv("SF_CLIENT_SECRET")
        self.refresh_token = os.getenv("SF_REFRESH_TOKEN")
        self.instance_url = os.getenv("SF_INSTANCE_URL", "").rstrip("/")
        self.login_url = os.getenv(
            "SF_LOGIN_URL", "https://login.salesforce.com"
        ).rstrip("/")

        if not all([
            self.client_id,
            self.client_secret,
            self.refresh_token,
        ]):
            raise ValueError(
                "Missing Salesforce OAuth credentials. Please set "
                "SF_CLIENT_ID, SF_CLIENT_SECRET, and SF_REFRESH_TOKEN."
            )

        self._connect()

    def _connect(self):
        """Exchange the refresh token for a fresh access token."""
        token_url = f"{self.login_url}/services/oauth2/token"

        try:
            response = requests.post(
                token_url,
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                },
                timeout=30,
            )
            response.raise_for_status()
            token_data = response.json()

            access_token = token_data["access_token"]
            self.instance_url = (
                token_data.get("instance_url") or self.instance_url
            ).rstrip("/")

            if not self.instance_url:
                raise ValueError(
                    "Salesforce did not return an instance URL, and "
                    "SF_INSTANCE_URL is not configured."
                )

            self.sf = Salesforce(
                instance_url=self.instance_url,
                session_id=access_token,
            )

            logger.info("Connected to Salesforce using OAuth successfully.")

        except requests.RequestException as e:
            logger.error("Salesforce OAuth token refresh failed.")
            if e.response is not None:
                logger.error(
                    "Salesforce response: %s",
                    e.response.text[:500],
                )
            raise RuntimeError(
                "Salesforce authentication failed. Check the OAuth "
                "credentials and refresh-token configuration."
            ) from e
        except (KeyError, ValueError) as e:
            logger.error("Invalid Salesforce OAuth response or configuration.")
            raise RuntimeError(
                "Could not establish the Salesforce connection."
            ) from e

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
            logger.error("SOQL execution failed: %s", e)
            raise

    def create_record(self, object_name: str, payload: dict) -> str:
        """Inserts a record into Salesforce and returns the ID."""
        try:
            res = getattr(self.sf, object_name).create(payload)

            if res.get("success"):
                return res.get("id")

            raise RuntimeError(
                f"Failed to create {object_name}: {res.get('errors')}"
            )

        except Exception as e:
            logger.error("Error creating %s: %s", object_name, e)
            raise

    def update_record(
        self, object_name: str, record_id: str, payload: dict
    ) -> bool:
        """Updates an existing Salesforce record."""
        try:
            getattr(self.sf, object_name).update(record_id, payload)
            logger.info(
                "Updated %s %s successfully.", object_name, record_id
            )
            return True

        except Exception as e:
            logger.error(
                "Error updating %s %s: %s", object_name, record_id, e
            )
            raise
