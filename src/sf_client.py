
"""
Salesforce REST Client
OAuth Refresh Token Authentication with detailed error diagnostics.
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

        self.instance_url = os.getenv(
            "SF_INSTANCE_URL", ""
        ).rstrip("/")

        self.login_url = os.getenv(
            "SF_LOGIN_URL",
            "https://login.salesforce.com",
        ).rstrip("/")

        if not all([
            self.client_id,
            self.client_secret,
            self.refresh_token,
        ]):
            raise ValueError(
                "Missing Salesforce OAuth credentials. Check that "
                "SF_CLIENT_ID, SF_CLIENT_SECRET, and SF_REFRESH_TOKEN "
                "are configured in Streamlit Secrets."
            )

        self.sf = None
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

            if not response.ok:
                # Salesforce normally returns a JSON error response.
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict):
                        error_code = error_data.get(
                            "error", "unknown_error"
                        )
                        error_description = error_data.get(
                            "error_description",
                            "No error description provided.",
                        )
                        details = (
                            f"{error_code}: {error_description}"
                        )
                    else:
                        details = str(error_data)
                except ValueError:
                    details = response.text[:500]

                # Never log client secrets, refresh tokens, or access tokens.
                logger.error(
                    "Salesforce OAuth token refresh failed "
                    "(HTTP %s): %s",
                    response.status_code,
                    details,
                )

                raise RuntimeError(
                    "Salesforce OAuth token refresh failed "
                    f"(HTTP {response.status_code}): {details}"
                )

            token_data = response.json()

            access_token = token_data.get("access_token")
            returned_instance_url = token_data.get("instance_url")

            if not access_token:
                raise RuntimeError(
                    "Salesforce OAuth response did not contain "
                    "an access token."
                )

            self.instance_url = (
                returned_instance_url or self.instance_url
            ).rstrip("/")

            if not self.instance_url:
                raise RuntimeError(
                    "No Salesforce instance URL was returned. "
                    "Check SF_INSTANCE_URL in Streamlit Secrets."
                )

            self.sf = Salesforce(
                instance_url=self.instance_url,
                session_id=access_token,
            )

            logger.info(
                "Salesforce OAuth authentication completed successfully."
            )

        except requests.RequestException as e:
            logger.error(
                "Unable to contact Salesforce OAuth endpoint: %s",
                type(e).__name__,
            )
            raise RuntimeError(
                "Could not contact Salesforce for authentication. "
                "Check the login URL and network connection."
            ) from e

    def query(self, soql: str) -> pd.DataFrame:
        """Executes SOQL and returns results as a DataFrame."""

        try:
            result = self.sf.query_all(soql)
            records = result.get("records", [])

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df.drop(
                columns=["attributes"],
                errors="ignore",
                inplace=True,
            )
            return df

        except Exception as e:
            logger.error("SOQL execution failed: %s", e)
            raise

    def create_record(self, object_name: str, payload: dict) -> str:
        """Inserts a record into Salesforce and returns its ID."""

        try:
            result = getattr(self.sf, object_name).create(payload)

            if result.get("success"):
                return result.get("id")

            raise RuntimeError(
                f"Failed to create {object_name}: "
                f"{result.get('errors')}"
            )

        except Exception as e:
            logger.error(
                "Error creating %s: %s", object_name, e
            )
            raise

    def update_record(
        self,
        object_name: str,
        record_id: str,
        payload: dict,
    ) -> bool:
        """Updates an existing Salesforce record."""

        try:
            getattr(self.sf, object_name).update(
                record_id, payload
            )

            logger.info(
                "Updated %s %s successfully.",
                object_name,
                record_id,
            )
            return True

        except Exception as e:
            logger.error(
                "Error updating %s %s: %s",
                object_name,
                record_id,
                e,
            )
            raise
