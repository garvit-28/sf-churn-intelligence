"""
Customer 360 Pulse
Salesforce Customer Churn Intelligence Engine

Architecture:

Salesforce / Local Data
        ↓
Feature Engineering
        ↓
Saved XGBoost Pipeline
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
SHAP Explainability
        ↓
Streamlit Dashboard
        ↓
Optional Salesforce Write-back
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from src.sf_client import SalesforceClient
except ImportError:
    SalesforceClient = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer 360 Pulse",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .slds-card {
        background: #ffffff;
        border: 1px solid #dddbda;
        border-radius: 0.25rem;
        box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.08);
        padding: 1.25rem;
        margin-bottom: 1rem;
        color: #181818;
    }

    .slds-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0176d3;
        margin-bottom: 0.75rem;
    }

    .risk-high {
        color: #ba0517;
        font-weight: 700;
    }

    .risk-medium {
        color: #8a6d1d;
        font-weight: 700;
    }

    .risk-low {
        color: #2e844a;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #706e6b;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #181818;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONSTANTS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "churn_model_artifact.joblib"
)

LOCAL_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "features_engineered.csv"
)

LOCAL_SCORED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "scored_accounts.csv"
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_trained_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model artifact not found.\n\n"
            "Please run:\n"
            "python src/etl.py\n"
            "python src/train.py"
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = [
        "pipeline",
        "model",
        "explainer",
        "feature_names",
        "numeric_features",
        "categorical_features",
        "risk_thresholds"
    ]

    missing = [
        key for key in required_keys
        if key not in artifact
    ]

    if missing:
        raise ValueError(
            f"Model artifact is missing keys: {missing}"
        )

    return artifact


# ============================================================
# LOAD MODEL
# ============================================================

try:

    artifact = load_trained_model()

    pipeline = artifact["pipeline"]
    model = artifact["model"]
    explainer = artifact["explainer"]

    NUMERIC_FEATURES = artifact["numeric_features"]
    CATEGORICAL_FEATURES = artifact["categorical_features"]

    FEATURE_COLS = (
        NUMERIC_FEATURES +
        CATEGORICAL_FEATURES
    )

    MODEL_FEATURE_NAMES = artifact["feature_names"]

    RISK_THRESHOLDS = artifact["risk_thresholds"]

    LOW_THRESHOLD = float(
        RISK_THRESHOLDS["low"]
    )

    HIGH_THRESHOLD = float(
        RISK_THRESHOLDS["high"]
    )

except Exception as e:

    st.error(
        f"Unable to load trained model: {e}"
    )

    st.stop()


# ============================================================
# SAFE CSV READER
# ============================================================

def safe_read_csv(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        return pd.DataFrame()

    encodings = [
        "utf-8",
        "utf-8-sig",
        "utf-16",
        "latin-1"
    ]

    for encoding in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=encoding
            )

            if not df.empty:
                return df

        except Exception:
            continue

    return pd.DataFrame()


# ============================================================
# DEMO DATA
# ============================================================

def generate_demo_data():

    """
    Creates a small demo Salesforce-like dataset.

    This is only a fallback for demonstration when
    Salesforce and local datasets are unavailable.
    """

    np.random.seed(42)

    n = 30

    industries = [
        "Technology",
        "Healthcare",
        "Finance",
        "Manufacturing",
        "Retail"
    ]

    contracts = [
        "Month-to-Month",
        "One Year",
        "Two Year"
    ]

    accounts = []

    for i in range(1, n + 1):

        tenure = int(
            np.random.randint(
                3,
                60
            )
        )

        monthly_charge = float(
            np.random.choice(
                [
                    500,
                    1000,
                    1500,
                    2500,
                    4000,
                    6000
                ]
            )
        )

        accounts.append(
            {
                "Id": f"DEMO-{i:04d}",
                "Name": f"Demo Account {i}",
                "Industry": np.random.choice(
                    industries
                ),
                "Contract_Type": np.random.choice(
                    contracts
                ),
                "Tenure_Months": tenure,
                "Monthly_Charges": monthly_charge,
                "Total_Charges": (
                    tenure *
                    monthly_charge
                ),
                "Total_Cases": int(
                    np.random.randint(
                        0,
                        15
                    )
                ),
                "Escalated_Cases": int(
                    np.random.randint(
                        0,
                        5
                    )
                ),
                "Critical_Cases": int(
                    np.random.randint(
                        0,
                        3
                    )
                ),
                "High_Cases": int(
                    np.random.randint(
                        0,
                        5
                    )
                ),
                "Avg_Days_To_Resolve": float(
                    np.random.uniform(
                        1,
                        10
                    )
                )
            }
        )

    return pd.DataFrame(accounts)


# ============================================================
# SALESFORCE DATA FETCH
# ============================================================

@st.cache_data(
    ttl=60,
    show_spinner=False
)
def fetch_salesforce_data():

    accounts = pd.DataFrame()
    cases = pd.DataFrame()

    if SalesforceClient is not None:

        try:

            sf = SalesforceClient()

            # ------------------------------------------------
            # Account query
            # ------------------------------------------------

            account_query = """
                SELECT
                    Id,
                    Name,
                    AnnualRevenue,
                    NumberOfEmployees,
                    Industry,
                    Tenure_Months__c,
                    Monthly_Charges__c,
                    Total_Charges__c,
                    Contract_Type__c,
                    Churn_Risk_Score__c,
                    Risk_Level__c,
                    Top_Churn_Driver__c
                FROM Account
            """

            try:

                accounts = sf.query(
                    account_query
                )

            except Exception as e:

                logging.warning(
                    "Extended Account query failed: %s",
                    e
                )

                # --------------------------------------------
                # Fallback basic Account query
                # --------------------------------------------

                try:

                    accounts = sf.query(
                        """
                        SELECT
                            Id,
                            Name,
                            AnnualRevenue,
                            NumberOfEmployees,
                            Industry
                        FROM Account
                        """
                    )

                except Exception as basic_error:

                    logging.warning(
                        "Basic Account query failed: %s",
                        basic_error
                    )

            # ------------------------------------------------
            # Case query
            # ------------------------------------------------

            try:

                cases = sf.query(
                    """
                    SELECT
                        Id,
                        AccountId,
                        Status,
                        Priority,
                        Origin
                    FROM Case
                    """
                )

            except Exception as e:

                logging.warning(
                    "Case query failed: %s",
                    e
                )

        except Exception as e:

            logging.warning(
                "Salesforce connection failed: %s",
                e
            )

    # ========================================================
    # LOCAL FALLBACK
    # ========================================================

    if accounts.empty:

        local_features = safe_read_csv(
            LOCAL_FEATURE_PATH
        )

        if not local_features.empty:

            accounts = local_features.copy()

            # Local engineered data already contains
            # many of the required ML features.

            return accounts, cases, "Local engineered dataset"

    # ========================================================
    # DEMO FALLBACK
    # ========================================================

    if accounts.empty:

        accounts = generate_demo_data()

        return accounts, cases, "Demo data"

    return accounts, cases, "Salesforce"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_matrix(
    accounts: pd.DataFrame,
    cases: pd.DataFrame
) -> pd.DataFrame:

    features = accounts.copy()

    # ========================================================
    # BASIC ACCOUNT FIELDS
    # ========================================================

    if "Name" not in features.columns:

        features["Name"] = [
            f"Account {i + 1}"
            for i in range(len(features))
        ]

    if "Id" not in features.columns:

        features["Id"] = [
            f"LOCAL-{i + 1}"
            for i in range(len(features))
        ]

    # ========================================================
    # INDUSTRY
    # ========================================================

    if "Industry" not in features.columns:

        features["Industry"] = "Other"

    features["Industry"] = (
        features["Industry"]
        .fillna("Other")
        .astype(str)
    )

    # ========================================================
    # TENURE
    # ========================================================

    if "Tenure_Months" not in features.columns:

        if "Tenure_Months__c" in features.columns:

            features["Tenure_Months"] = pd.to_numeric(
                features["Tenure_Months__c"],
                errors="coerce"
            )

        else:

            features["Tenure_Months"] = 12

    features["Tenure_Months"] = (
        pd.to_numeric(
            features["Tenure_Months"],
            errors="coerce"
        )
        .fillna(12)
        .clip(lower=1)
    )

    # ========================================================
    # MONTHLY CHARGES
    # ========================================================

    if "Monthly_Charges" not in features.columns:

        if "Monthly_Charges__c" in features.columns:

            features["Monthly_Charges"] = pd.to_numeric(
                features["Monthly_Charges__c"],
                errors="coerce"
            )

        elif "AnnualRevenue" in features.columns:

            features["Monthly_Charges"] = (
                pd.to_numeric(
                    features["AnnualRevenue"],
                    errors="coerce"
                )
                .fillna(6000)
                / 12
            )

        else:

            features["Monthly_Charges"] = 500.0

    features["Monthly_Charges"] = (
        pd.to_numeric(
            features["Monthly_Charges"],
            errors="coerce"
        )
        .fillna(500.0)
        .clip(lower=0)
    )

    # ========================================================
    # TOTAL CHARGES
    # ========================================================

    if "Total_Charges" not in features.columns:

        if "Total_Charges__c" in features.columns:

            features["Total_Charges"] = pd.to_numeric(
                features["Total_Charges__c"],
                errors="coerce"
            )

        else:

            features["Total_Charges"] = (
                features["Monthly_Charges"]
                *
                features["Tenure_Months"]
            )

    features["Total_Charges"] = (
        pd.to_numeric(
            features["Total_Charges"],
            errors="coerce"
        )
        .fillna(
            features["Monthly_Charges"]
            *
            features["Tenure_Months"]
        )
        .clip(lower=0)
    )

    # ========================================================
    # CONTRACT TYPE
    # ========================================================

    if "Contract_Type" not in features.columns:

        if "Contract_Type__c" in features.columns:

            features["Contract_Type"] = (
                features["Contract_Type__c"]
            )

        else:

            features["Contract_Type"] = (
                "Month-to-Month"
            )

    features["Contract_Type"] = (
        features["Contract_Type"]
        .fillna("Month-to-Month")
        .astype(str)
    )

    # ========================================================
    # CASE DATA
    # ========================================================

    required_case_features = [
        "Total_Cases",
        "Escalated_Cases",
        "Critical_Cases",
        "High_Cases",
        "Avg_Days_To_Resolve"
    ]

    # --------------------------------------------------------
    # If local engineered dataset already contains these,
    # preserve them.
    # --------------------------------------------------------

    has_existing_case_features = all(
        col in features.columns
        for col in required_case_features
    )

    if not has_existing_case_features:

        if (
            not cases.empty
            and
            "AccountId" in cases.columns
        ):

            cases_clean = cases.copy()

            cases_clean["AccountId"] = (
                cases_clean["AccountId"]
                .astype(str)
                .str.strip()
            )

            # ----------------------------------------------
            # Status normalization
            # ----------------------------------------------

            if "Status" not in cases_clean.columns:

                cases_clean["Status"] = ""

            status = (
                cases_clean["Status"]
                .astype(str)
                .str.lower()
            )

            cases_clean["Is_Escalated"] = (
                status.str.contains(
                    "escalat",
                    na=False
                )
            )

            # ----------------------------------------------
            # Priority
            # ----------------------------------------------

            if "Priority" not in cases_clean.columns:

                cases_clean["Priority"] = ""

            priority = (
                cases_clean["Priority"]
                .astype(str)
                .str.lower()
            )

            cases_clean["Is_Critical"] = (
                priority == "critical"
            )

            cases_clean["Is_High"] = (
                priority == "high"
            )

            # ----------------------------------------------
            # Resolution time
            # ----------------------------------------------

            if "Days_To_Resolve" in cases_clean.columns:

                cases_clean["Days_To_Resolve"] = (
                    pd.to_numeric(
                        cases_clean[
                            "Days_To_Resolve"
                        ],
                        errors="coerce"
                    )
                )

            else:

                cases_clean["Days_To_Resolve"] = np.nan

            case_agg = (
                cases_clean
                .groupby("AccountId")
                .agg(
                    Total_Cases=(
                        "AccountId",
                        "count"
                    ),
                    Escalated_Cases=(
                        "Is_Escalated",
                        "sum"
                    ),
                    Critical_Cases=(
                        "Is_Critical",
                        "sum"
                    ),
                    High_Cases=(
                        "Is_High",
                        "sum"
                    ),
                    Avg_Days_To_Resolve=(
                        "Days_To_Resolve",
                        "mean"
                    )
                )
                .reset_index()
            )

            case_agg.rename(
                columns={
                    "AccountId": "Salesforce_Account_Id"
                },
                inplace=True
            )

            features["Salesforce_Account_Id"] = (
                features["Id"]
                .astype(str)
                .str.strip()
            )

            features = features.merge(
                case_agg,
                left_on="Salesforce_Account_Id",
                right_on="Salesforce_Account_Id",
                how="left"
            )

            features.drop(
                columns=[
                    "Salesforce_Account_Id"
                ],
                errors="ignore",
                inplace=True
            )

        else:

            for col in required_case_features:

                features[col] = 0

    # ========================================================
    # NUMERIC CLEANUP
    # ========================================================

    for col in [
        "Total_Cases",
        "Escalated_Cases",
        "Critical_Cases",
        "High_Cases"
    ]:

        if col not in features.columns:

            features[col] = 0

        features[col] = (
            pd.to_numeric(
                features[col],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

    # ========================================================
    # AVG DAYS TO RESOLVE
    # ========================================================

    if "Avg_Days_To_Resolve" not in features.columns:

        features["Avg_Days_To_Resolve"] = 0.0

    features["Avg_Days_To_Resolve"] = (
        pd.to_numeric(
            features["Avg_Days_To_Resolve"],
            errors="coerce"
        )
        .fillna(0.0)
    )

    # ========================================================
    # DERIVED FEATURES
    # ========================================================

    features["Escalation_Rate"] = np.where(
        features["Total_Cases"] > 0,
        (
            features["Escalated_Cases"]
            /
            features["Total_Cases"]
        ),
        0.0
    )

    features["Escalation_Rate"] = (
        features["Escalation_Rate"]
        .clip(0, 1)
        .round(3)
    )

    features["Avg_Monthly_Case_Load"] = (
        features["Total_Cases"]
        /
        features["Tenure_Months"].replace(
            0,
            1
        )
    ).round(3)

    features["Est_Annual_Value"] = (
        features["Monthly_Charges"]
        *
        12
    ).round(2)

    return features


# ============================================================
# MODEL INFERENCE
# ============================================================

def score_accounts(
    df_features: pd.DataFrame
):

    X = df_features[
        FEATURE_COLS
    ].copy()

    # --------------------------------------------------------
    # Ensure numeric columns are numeric
    # --------------------------------------------------------

    for col in NUMERIC_FEATURES:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------------
    # Ensure categorical columns are strings
    # --------------------------------------------------------

    for col in CATEGORICAL_FEATURES:

        X[col] = (
            X[col]
            .fillna("Other")
            .astype(str)
        )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probabilities = (
        pipeline
        .predict_proba(X)[:, 1]
    )

    df_scored = df_features.copy()

    df_scored["Churn Risk Score"] = (
        np.round(
            probabilities,
            4
        )
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    df_scored["Risk Level"] = np.select(
        [
            df_scored["Churn Risk Score"]
            >= HIGH_THRESHOLD,

            df_scored["Churn Risk Score"]
            >= LOW_THRESHOLD
        ],
        [
            "High",
            "Medium"
        ],
        default="Low"
    )

    # --------------------------------------------------------
    # Estimated Revenue At Risk
    # --------------------------------------------------------

    if "AnnualRevenue" in df_scored.columns:

        annual_revenue = pd.to_numeric(
            df_scored["AnnualRevenue"],
            errors="coerce"
        ).fillna(
            df_scored["Est_Annual_Value"]
        )

    else:

        annual_revenue = (
            df_scored["Est_Annual_Value"]
        )

    df_scored["Estimated Revenue At Risk"] = (
        annual_revenue
        *
        df_scored["Churn Risk Score"]
    ).round(2)

    return df_scored, X


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_values(X):

    preprocessor = (
        pipeline
        .named_steps["preprocessor"]
    )

    X_encoded = (
        preprocessor.transform(X)
    )

    shap_raw = (
        explainer
        .shap_values(X_encoded)
    )

    if isinstance(
        shap_raw,
        list
    ):

        shap_matrix = shap_raw[1]

    else:

        shap_matrix = shap_raw

    return (
        shap_matrix,
        X_encoded
    )


# ============================================================
# FRIENDLY SHAP NAME
# ============================================================

def clean_feature_name(
    feature_name: str
):

    name = feature_name

    name = name.replace(
        "numeric__",
        ""
    )

    name = name.replace(
        "categorical__",
        ""
    )

    name = name.replace(
        "_",
        " "
    )

    return name.title()


# ============================================================
# TOP DRIVER
# ============================================================

def get_top_driver(
    shap_row,
    probability
):

    if probability < LOW_THRESHOLD:

        return "No Strong Positive Churn Driver"

    positive_values = np.where(
        shap_row > 0,
        shap_row,
        -np.inf
    )

    max_index = int(
        np.argmax(
            positive_values
        )
    )

    max_value = shap_row[
        max_index
    ]

    if max_value <= 0:

        return "No Strong Positive Churn Driver"

    return (
        f"Elevated "
        f"{clean_feature_name(MODEL_FEATURE_NAMES[max_index])}"
    )


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "Loading customer data..."
):

    accounts_df, cases_df, DATA_SOURCE = (
        fetch_salesforce_data()
    )

    df_features = build_feature_matrix(
        accounts_df,
        cases_df
    )

    try:

        df_scored, X_input = score_accounts(
            df_features
        )

        shap_matrix, X_encoded = (
            generate_shap_values(
                X_input
            )
        )

    except Exception as e:

        st.error(
            f"Model scoring failed: {e}"
        )

        st.stop()


# ============================================================
# ADD TOP DRIVERS
# ============================================================

top_drivers = []

for i in range(
    len(df_scored)
):

    driver = get_top_driver(
        shap_matrix[i],
        float(
            df_scored[
                "Churn Risk Score"
            ].iloc[i]
        )
    )

    top_drivers.append(
        driver
    )

df_scored[
    "Top Churn Driver"
] = top_drivers


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚡ Customer 360 Pulse"
)

st.caption(
    "Salesforce Customer Churn Intelligence Engine"
)

st.info(
    f"Data source: **{DATA_SOURCE}**  |  "
    f"Model: **XGBoost**  |  "
    f"ROC-AUC during training: **0.8583**  |  "
    f"Risk thresholds: Low < {LOW_THRESHOLD:.2f}, "
    f"Medium ≥ {LOW_THRESHOLD:.2f}, "
    f"High ≥ {HIGH_THRESHOLD:.2f}"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Customer 360 Pulse"
    )

    st.divider()

    st.subheader(
        "Dashboard Controls"
    )

    if st.button(
        "🔄 Refresh Data",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # Risk filter
    # --------------------------------------------------------

    selected_risk = st.multiselect(
        "Risk Level",
        options=[
            "High",
            "Medium",
            "Low"
        ],
        default=[
            "High",
            "Medium",
            "Low"
        ]
    )

    # --------------------------------------------------------
    # Industry filter
    # --------------------------------------------------------

    industries = sorted(
        df_scored[
            "Industry"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_industries = st.multiselect(
        "Industry",
        options=industries,
        default=industries
    )

    st.divider()

    st.subheader(
        "Salesforce Sync"
    )

    auto_sync = st.checkbox(
        "Enable automatic sync",
        value=False
    )

    manual_sync = st.button(
        "☁️ Sync Predictions to Salesforce",
        use_container_width=True
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df_scored[
    df_scored[
        "Risk Level"
    ].isin(selected_risk)
    &
    df_scored[
        "Industry"
    ].isin(selected_industries)
].copy()


# ============================================================
# SALESFORCE SYNC
# ============================================================

def execute_salesforce_sync(
    data_frame: pd.DataFrame
):

    if SalesforceClient is None:

        return (
            0,
            [
                "SalesforceClient could not be imported."
            ]
        )

    if data_frame.empty:

        return (
            0,
            [
                "No accounts available for synchronization."
            ]
        )

    try:

        sf = SalesforceClient()

    except Exception as e:

        return (
            0,
            [
                f"Salesforce connection failed: {e}"
            ]
        )

    success_count = 0
    errors = []

    for _, row in data_frame.iterrows():

        record_id = str(
            row.get(
                "Id",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # Never attempt to update demo/local IDs
        # ----------------------------------------------------

        if (
            not record_id
            or
            record_id.startswith("DEMO-")
            or
            record_id.startswith("LOCAL-")
        ):

            continue

        payload = {
            "Churn_Risk_Score__c": float(
                row["Churn Risk Score"]
            ),
            "Risk_Level__c": str(
                row["Risk Level"]
            ),
            "Top_Churn_Driver__c": str(
                row["Top Churn Driver"]
            )
        }

        try:

            sf.update_record(
                "Account",
                record_id,
                payload
            )

            success_count += 1

        except Exception as e:

            errors.append(
                f"{row.get('Name', record_id)}: {e}"
            )

    return (
        success_count,
        errors
    )


if manual_sync:

    with st.spinner(
        "Syncing predictions to Salesforce..."
    ):

        synced_count, sync_errors = (
            execute_salesforce_sync(
                filtered_df
            )
        )

    if synced_count > 0:

        st.success(
            f"{synced_count} account(s) updated in Salesforce."
        )

    if sync_errors:

        st.warning(
            f"{len(sync_errors)} account(s) were not updated."
        )

        with st.expander(
            "View sync errors"
        ):

            for error in sync_errors:

                st.write(
                    f"- {error}"
                )


# Automatic sync is intentionally explicit.
# It does NOT run automatically merely because the
# dashboard refreshed.


# ============================================================
# TOP METRICS
# ============================================================

total_accounts = len(
    filtered_df
)

high_risk = int(
    (
        filtered_df["Risk Level"]
        == "High"
    ).sum()
)

medium_risk = int(
    (
        filtered_df["Risk Level"]
        == "Medium"
    ).sum()
)

low_risk = int(
    (
        filtered_df["Risk Level"]
        == "Low"
    ).sum()
)

avg_risk = (
    filtered_df[
        "Churn Risk Score"
    ].mean()
    if not filtered_df.empty
    else 0
)

revenue_at_risk = (
    filtered_df[
        "Estimated Revenue At Risk"
    ].sum()
    if not filtered_df.empty
    else 0
)


col1, col2, col3, col4, col5 = (
    st.columns(5)
)

col1.metric(
    "Accounts",
    total_accounts
)

col2.metric(
    "High Risk",
    high_risk
)

col3.metric(
    "Medium Risk",
    medium_risk
)

col4.metric(
    "Average Risk",
    f"{avg_risk:.1%}"
)

col5.metric(
    "Estimated Revenue At Risk",
    f"₹{revenue_at_risk:,.0f}"
)


st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Executive Overview",
        "🔍 Customer 360 & SHAP",
        "🎛️ What-If Simulator",
        "⚡ Action Studio",
        "🧪 SOQL Studio"
    ]
)


# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tab1:

    st.subheader(
        "Executive Overview"
    )

    if filtered_df.empty:

        st.warning(
            "No accounts match the selected filters."
        )

    else:

        left, right = st.columns(2)

        # ----------------------------------------------------
        # Risk distribution
        # ----------------------------------------------------

        with left:

            risk_counts = (
                filtered_df[
                    "Risk Level"
                ]
                .value_counts()
                .reindex(
                    [
                        "High",
                        "Medium",
                        "Low"
                    ],
                    fill_value=0
                )
                .reset_index()
            )

            risk_counts.columns = [
                "Risk Level",
                "Accounts"
            ]

            fig_risk = px.pie(
                risk_counts,
                names="Risk Level",
                values="Accounts",
                title="Customer Risk Distribution",
                hole=0.45
            )

            st.plotly_chart(
                fig_risk,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Risk score distribution
        # ----------------------------------------------------

        with right:

            fig_hist = px.histogram(
                filtered_df,
                x="Churn Risk Score",
                nbins=20,
                title="Churn Probability Distribution"
            )

            fig_hist.add_vline(
                x=LOW_THRESHOLD,
                line_dash="dash",
                annotation_text="Medium threshold"
            )

            fig_hist.add_vline(
                x=HIGH_THRESHOLD,
                line_dash="dash",
                annotation_text="High threshold"
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Revenue at risk
        # ----------------------------------------------------

        if not filtered_df.empty:

            revenue_df = (
                filtered_df
                .groupby(
                    "Risk Level"
                )[
                    "Estimated Revenue At Risk"
                ]
                .sum()
                .reindex(
                    [
                        "High",
                        "Medium",
                        "Low"
                    ],
                    fill_value=0
                )
                .reset_index()
            )

            fig_revenue = px.bar(
                revenue_df,
                x="Risk Level",
                y="Estimated Revenue At Risk",
                title="Estimated Revenue At Risk by Risk Level"
            )

            st.plotly_chart(
                fig_revenue,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Account registry
        # ----------------------------------------------------

        st.subheader(
            "Account Registry"
        )

        display_columns = [
            "Name",
            "Industry",
            "Contract_Type",
            "Tenure_Months",
            "Monthly_Charges",
            "Total_Cases",
            "Escalated_Cases",
            "Critical_Cases",
            "Escalation_Rate",
            "Churn Risk Score",
            "Risk Level",
            "Top Churn Driver"
        ]

        available_columns = [
            col
            for col in display_columns
            if col in filtered_df.columns
        ]

        registry = (
            filtered_df[
                available_columns
            ]
            .sort_values(
                "Churn Risk Score",
                ascending=False
            )
        )

        st.dataframe(
            registry,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2 — CUSTOMER 360 & SHAP
# ============================================================

with tab2:

    st.subheader(
        "Customer 360 & Explainability"
    )

    if filtered_df.empty:

        st.warning(
            "No accounts available."
        )

    else:

        account_names = (
            filtered_df[
                "Name"
            ]
            .astype(str)
            .tolist()
        )

        selected_account = st.selectbox(
            "Select Customer",
            account_names
        )

        account_mask = (
            df_scored["Name"]
            .astype(str)
            ==
            str(selected_account)
        )

        matching_indices = (
            df_scored.index[
                account_mask
            ]
            .tolist()
        )

        if not matching_indices:

            st.error(
                "Selected account could not be found."
            )

        else:

            acc_idx = matching_indices[0]

            account = (
                df_scored
                .loc[acc_idx]
            )

            # ------------------------------------------------
            # Customer metrics
            # ------------------------------------------------

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Churn Probability",
                f"{account['Churn Risk Score']:.1%}"
            )

            m2.metric(
                "Risk Level",
                account["Risk Level"]
            )

            m3.metric(
                "Support Cases",
                int(
                    account[
                        "Total_Cases"
                    ]
                )
            )

            m4.metric(
                "Escalation Rate",
                f"{account['Escalation_Rate']:.1%}"
            )

            st.divider()

            left, right = st.columns(
                [1, 1]
            )

            # ------------------------------------------------
            # Customer profile
            # ------------------------------------------------

            with left:

                st.markdown(
                    "### Customer Profile"
                )

                profile = {
                    "Account": account["Name"],
                    "Industry": account["Industry"],
                    "Contract": account["Contract_Type"],
                    "Tenure": f"{account['Tenure_Months']:.0f} months",
                    "Monthly Charges": f"₹{account['Monthly_Charges']:,.2f}",
                    "Total Charges": f"₹{account['Total_Charges']:,.2f}",
                    "Support Cases": int(account["Total_Cases"]),
                    "Escalated Cases": int(account["Escalated_Cases"]),
                    "Critical Cases": int(account["Critical_Cases"]),
                    "High Priority Cases": int(account["High_Cases"]),
                    "Avg Resolution Days": f"{account['Avg_Days_To_Resolve']:.2f}"
                }

                profile_df = pd.DataFrame(
                    profile.items(),
                    columns=[
                        "Metric",
                        "Value"
                    ]
                )

                st.dataframe(
                    profile_df,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # SHAP chart
            # ------------------------------------------------

            with right:

                st.markdown(
                    "### Churn Drivers"
                )

                acc_shap = (
                    shap_matrix[
                        acc_idx
                    ]
                )

                shap_df = pd.DataFrame(
                    {
                        "Feature":
                            [
                                clean_feature_name(
                                    name
                                )
                                for name
                                in MODEL_FEATURE_NAMES
                            ],
                        "SHAP":
                            acc_shap
                    }
                )

                # Show strongest absolute contributions
                shap_df["Absolute"] = (
                    shap_df["SHAP"]
                    .abs()
                )

                shap_df = (
                    shap_df
                    .sort_values(
                        "Absolute",
                        ascending=False
                    )
                    .head(10)
                    .sort_values(
                        "SHAP"
                    )
                )

                fig_shap = px.bar(
                    shap_df,
                    x="SHAP",
                    y="Feature",
                    orientation="h",
                    title="Top SHAP Contributions"
                )

                fig_shap.add_vline(
                    x=0,
                    line_dash="solid"
                )

                st.plotly_chart(
                    fig_shap,
                    use_container_width=True
                )

            # ------------------------------------------------
            # Top driver
            # ------------------------------------------------

            st.info(
                f"**Primary Churn Driver:** "
                f"{account['Top Churn Driver']}"
            )

            # ------------------------------------------------
            # Recommended action
            # ------------------------------------------------

            st.markdown(
                "### Suggested Retention Action"
            )

            if account["Risk Level"] == "High":

                st.error(
                    "Prioritize this account for immediate "
                    "customer-success intervention. Review "
                    "support escalation, contract status, "
                    "and recent customer issues."
                )

            elif account["Risk Level"] == "Medium":

                st.warning(
                    "Schedule a proactive customer-success "
                    "review and monitor support activity."
                )

            else:

                st.success(
                    "Account currently shows lower modeled "
                    "churn probability. Continue normal monitoring."
                )

            # ------------------------------------------------
            # Salesforce actions
            # ------------------------------------------------

            st.markdown(
                "### Salesforce Actions"
            )

            action_col1, action_col2 = st.columns(2)

            with action_col1:

                if st.button(
                    "☁️ Sync This Account",
                    key=f"sync_{acc_idx}"
                ):

                    count, errors = (
                        execute_salesforce_sync(
                            pd.DataFrame(
                                [account]
                            )
                        )
                    )

                    if count:

                        st.success(
                            "Prediction synchronized "
                            "to Salesforce."
                        )

                    else:

                        st.error(
                            errors[0]
                            if errors
                            else
                            "Account was not synchronized."
                        )

            with action_col2:

                if st.button(
                    "📋 Prepare Retention Task",
                    key=f"task_{acc_idx}"
                ):

                    st.session_state[
                        "task_account"
                    ] = account["Name"]

                    st.success(
                        "Retention task prepared. "
                        "Use Action Studio to create it."
                    )


# ============================================================
# TAB 3 — WHAT-IF SIMULATOR
# ============================================================

with tab3:

    st.subheader(
        "What-If Churn Simulator"
    )

    st.write(
        """
        Adjust the customer's actual model features and observe
        how the **same production XGBoost pipeline** changes the
        predicted churn probability.
        """
    )

    st.info(
        "The simulator uses the saved production model from "
        "`models/churn_model_artifact.joblib`. It does not train "
        "a separate model."
    )

    # --------------------------------------------------------
    # Select base customer
    # --------------------------------------------------------

    if df_scored.empty:

        st.warning(
            "No customer data available."
        )

    else:

        simulator_account = st.selectbox(
            "Base Customer",
            df_scored["Name"].astype(str).tolist(),
            key="simulator_account"
        )

        sim_idx_list = (
            df_scored.index[
                df_scored["Name"].astype(str)
                ==
                simulator_account
            ]
            .tolist()
        )

        if sim_idx_list:

            sim_idx = sim_idx_list[0]

            base = (
                df_scored
                .loc[sim_idx]
            )

            st.markdown(
                "### Adjust Model Inputs"
            )

            # ------------------------------------------------
            # Numeric inputs
            # ------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                tenure = st.number_input(
                    "Tenure (months)",
                    min_value=1.0,
                    max_value=240.0,
                    value=float(
                        base["Tenure_Months"]
                    ),
                    step=1.0
                )

            with c2:

                monthly_charges = st.number_input(
                    "Monthly Charges",
                    min_value=0.0,
                    value=float(
                        base["Monthly_Charges"]
                    ),
                    step=100.0
                )

            with c3:

                total_cases = st.number_input(
                    "Total Cases",
                    min_value=0,
                    max_value=500,
                    value=int(
                        base["Total_Cases"]
                    ),
                    step=1
                )

            c4, c5, c6 = st.columns(3)

            with c4:

                escalated_cases = st.number_input(
                    "Escalated Cases",
                    min_value=0,
                    max_value=500,
                    value=int(
                        base["Escalated_Cases"]
                    ),
                    step=1
                )

            with c5:

                critical_cases = st.number_input(
                    "Critical Cases",
                    min_value=0,
                    max_value=500,
                    value=int(
                        base["Critical_Cases"]
                    ),
                    step=1
                )

            with c6:

                high_cases = st.number_input(
                    "High Priority Cases",
                    min_value=0,
                    max_value=500,
                    value=int(
                        base["High_Cases"]
                    ),
                    step=1
                )

            c7, c8, c9 = st.columns(3)

            with c7:

                avg_resolution = st.number_input(
                    "Average Resolution Days",
                    min_value=0.0,
                    max_value=365.0,
                    value=float(
                        base["Avg_Days_To_Resolve"]
                    ),
                    step=0.5
                )

            with c8:

                industry = st.selectbox(
                    "Industry",
                    sorted(
                        df_scored[
                            "Industry"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    ),
                    index=(
                        sorted(
                            df_scored[
                                "Industry"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                        .index(
                            str(
                                base["Industry"]
                            )
                        )
                        if str(
                            base["Industry"]
                        ) in sorted(
                            df_scored[
                                "Industry"
                            ]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                        else 0
                    )
                )

            with c9:

                contract_options = [
                    "Month-to-Month",
                    "One Year",
                    "Two Year"
                ]

                current_contract = str(
                    base["Contract_Type"]
                )

                contract = st.selectbox(
                    "Contract Type",
                    contract_options,
                    index=(
                        contract_options.index(
                            current_contract
                        )
                        if current_contract
                        in contract_options
                        else 0
                    )
                )

            # ------------------------------------------------
            # Derived values
            # ------------------------------------------------

            total_charges = (
                monthly_charges
                *
                tenure
            )

            escalation_rate = (
                escalated_cases
                /
                total_cases
                if total_cases > 0
                else 0
            )

            avg_monthly_case_load = (
                total_cases
                /
                max(
                    tenure,
                    1
                )
            )

            est_annual_value = (
                monthly_charges
                *
                12
            )

            simulation_row = pd.DataFrame(
                [
                    {
                        "Tenure_Months": tenure,
                        "Monthly_Charges": monthly_charges,
                        "Total_Charges": total_charges,
                        "Total_Cases": total_cases,
                        "Escalated_Cases": escalated_cases,
                        "Avg_Days_To_Resolve": avg_resolution,
                        "Critical_Cases": critical_cases,
                        "High_Cases": high_cases,
                        "Escalation_Rate": escalation_rate,
                        "Avg_Monthly_Case_Load": avg_monthly_case_load,
                        "Est_Annual_Value": est_annual_value,
                        "Industry": industry,
                        "Contract_Type": contract
                    }
                ]
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            simulation_probability = float(
                pipeline
                .predict_proba(
                    simulation_row[
                        FEATURE_COLS
                    ]
                )[:, 1][0]
            )

            if simulation_probability >= HIGH_THRESHOLD:

                simulation_risk = "High"

            elif simulation_probability >= LOW_THRESHOLD:

                simulation_risk = "Medium"

            else:

                simulation_risk = "Low"

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            st.divider()

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Simulated Churn Probability",
                f"{simulation_probability:.1%}"
            )

            r2.metric(
                "Simulated Risk",
                simulation_risk
            )

            change = (
                simulation_probability
                -
                float(
                    base[
                        "Churn Risk Score"
                    ]
                )
            )

            r3.metric(
                "Change vs Current",
                f"{change:+.1%}"
            )

            # ------------------------------------------------
            # Gauge
            # ------------------------------------------------

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=simulation_probability * 100,
                    title={
                        "text":
                        "Simulated Churn Probability"
                    },
                    number={
                        "suffix": "%"
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                100
                            ]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value":
                                HIGH_THRESHOLD
                                * 100
                        }
                    }
                )
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )

            st.caption(
                "This simulator is a scenario analysis tool. "
                "Changing a feature does not mean the customer's "
                "real Salesforce record has changed."
            )


# ============================================================
# TAB 4 — ACTION STUDIO
# ============================================================

with tab4:

    st.subheader(
        "⚡ Salesforce Action Studio"
    )

    st.write(
        """
        Use this area to create Salesforce records or prepare
        retention actions based on the churn analysis.
        """
    )

    action_tab1, action_tab2 = st.tabs(
        [
            "Create Account",
            "Create Case"
        ]
    )

    # ========================================================
    # CREATE ACCOUNT
    # ========================================================

    with action_tab1:

        st.markdown(
            "### Create Salesforce Account"
        )

        with st.form(
            "create_account_form"
        ):

            new_name = st.text_input(
                "Account Name"
            )

            new_industry = st.selectbox(
                "Industry",
                [
                    "Technology",
                    "Healthcare",
                    "Finance",
                    "Manufacturing",
                    "Retail",
                    "Other"
                ]
            )

            new_revenue = st.number_input(
                "Annual Revenue",
                min_value=0.0,
                value=500000.0,
                step=50000.0
            )

            submitted = st.form_submit_button(
                "Create Account"
            )

        if submitted:

            if not new_name.strip():

                st.error(
                    "Account name is required."
                )

            elif SalesforceClient is None:

                st.error(
                    "SalesforceClient is unavailable."
                )

            else:

                try:

                    sf = SalesforceClient()

                    payload = {
                        "Name": new_name.strip(),
                        "Industry": new_industry,
                        "AnnualRevenue": float(
                            new_revenue
                        )
                    }

                    new_id = sf.create_record(
                        "Account",
                        payload
                    )

                    st.success(
                        f"Account created successfully. "
                        f"Salesforce ID: {new_id}"
                    )

                except Exception as e:

                    st.error(
                        f"Account creation failed: {e}"
                    )

    # ========================================================
    # CREATE CASE
    # ========================================================

    with action_tab2:

        st.markdown(
            "### Create Salesforce Case"
        )

        account_options = (
            df_scored[
                [
                    "Id",
                    "Name"
                ]
            ]
            .dropna()
        )

        if account_options.empty:

            st.warning(
                "No accounts available."
            )

        else:

            with st.form(
                "create_case_form"
            ):

                selected_case_account = st.selectbox(
                    "Account",
                    account_options[
                        "Name"
                    ].tolist()
                )

                case_subject = st.text_input(
                    "Subject",
                    value="Churn Risk Follow-up"
                )

                case_priority = st.selectbox(
                    "Priority",
                    [
                        "Low",
                        "Medium",
                        "High",
                        "Critical"
                    ]
                )

                case_origin = st.selectbox(
                    "Origin",
                    [
                        "Phone",
                        "Email",
                        "Web"
                    ]
                )

                case_description = st.text_area(
                    "Description",
                    value=(
                        "Customer churn risk review "
                        "initiated from Customer 360 Pulse."
                    )
                )

                create_case = st.form_submit_button(
                    "Create Case"
                )

            if create_case:

                selected_row = (
                    account_options[
                        account_options[
                            "Name"
                        ]
                        ==
                        selected_case_account
                    ]
                    .iloc[0]
                )

                if SalesforceClient is None:

                    st.error(
                        "SalesforceClient is unavailable."
                    )

                else:

                    try:

                        sf = SalesforceClient()

                        payload = {
                            "AccountId": str(
                                selected_row["Id"]
                            ),
                            "Subject": case_subject,
                            "Priority": case_priority,
                            "Origin": case_origin,
                            "Description": case_description
                        }

                        case_id = sf.create_record(
                            "Case",
                            payload
                        )

                        st.success(
                            f"Case created successfully. "
                            f"Salesforce ID: {case_id}"
                        )

                    except Exception as e:

                        st.error(
                            f"Case creation failed: {e}"
                        )


# ============================================================
# TAB 5 — SOQL STUDIO
# ============================================================

with tab5:

    st.subheader(
        "🧪 SOQL Studio"
    )

    st.write(
        "Run a read-only SOQL query against the connected Salesforce org."
    )

    default_query = """
SELECT
    Id,
    Name,
    Industry,
    AnnualRevenue
FROM Account
LIMIT 20
""".strip()

    soql_query = st.text_area(
        "SOQL Query",
        value=default_query,
        height=180
    )

    execute_query = st.button(
        "▶ Execute SOQL"
    )

    if execute_query:

        cleaned_query = (
            soql_query
            .strip()
        )

        # ----------------------------------------------------
        # Basic read-only protection
        # ----------------------------------------------------

        normalized = (
            cleaned_query
            .lower()
            .replace("\n", " ")
            .strip()
        )

        if not normalized.startswith(
            "select"
        ):

            st.error(
                "SOQL Studio only permits SELECT queries."
            )

        elif any(
            keyword in normalized
            for keyword in [
                " insert ",
                " update ",
                " delete ",
                " upsert ",
                " merge ",
                " undelete "
            ]
        ):

            st.error(
                "Only read-only SELECT queries are permitted."
            )

        elif SalesforceClient is None:

            st.error(
                "SalesforceClient is unavailable."
            )

        else:

            try:

                sf = SalesforceClient()

                result = sf.query(
                    cleaned_query
                )

                if result.empty:

                    st.info(
                        "Query executed successfully, "
                        "but returned no records."
                    )

                else:

                    st.success(
                        f"{len(result)} records returned."
                    )

                    st.dataframe(
                        result,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as e:

                st.error(
                    f"SOQL execution failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer 360 Pulse • XGBoost Churn Intelligence • "
    "Salesforce + Python + SHAP"
)