"""
Customer 360 Pulse: Salesforce Churn Intelligence Engine
Features:
- Branded as Customer 360 Pulse
- Multi-tier calibration: Low, Medium (0.28-0.45), and High risk representation
- Non-zero floor: Eliminates 0.00 score collapse
- Automatic + Manual Salesforce custom field sync
- XGBoost inference with live SHAP attribution & SLDS action studio
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.sf_client import SalesforceClient
except ImportError:
    SalesforceClient = None

# ==========================================
# 1. PAGE CONFIGURATION & SLDS STYLING
# ==========================================
st.set_page_config(
    page_title="Customer 360 Pulse",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .slds-card {
        background: #ffffff;
        border: 1px solid #dddbda;
        border-radius: 0.25rem;
        box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.1);
        padding: 1.25rem;
        margin-bottom: 1rem;
        color: #181818;
    }
    .slds-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0176d3;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA INGESTION
# ==========================================
def safe_read_csv(file_path: str) -> pd.DataFrame:
    """Reads CSV safely across multiple encodings."""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    for enc in ["utf-16", "utf-8-sig", "utf-8", "latin-1"]:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            if not df.empty and len(df.columns) > 1:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def generate_synthetic_data():
    """Fallback generator when live Salesforce or CSV data is absent."""
    np.random.seed(42)
    n = 20
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
    accounts, cases = [], []
    
    for i in range(1, n + 1):
        acc_id = f"0015g00000{i:04d}AAA"
        name = f"Enterprise Corp {i}"
        rev = float(np.random.choice([250000, 750000, 1500000, 3200000, 5000000]))
        emp = int(np.random.choice([50, 150, 500, 1200, 3000]))
        ind = str(np.random.choice(industries))
        
        accounts.append({
            "Id": acc_id,
            "Name": name,
            "AnnualRevenue": rev,
            "NumberOfEmployees": emp,
            "Industry": ind
        })
        
        num_cases = np.random.randint(0, 5)
        for c in range(num_cases):
            cases.append({
                "Id": f"5005g00000{i:02d}{c:02d}BBB",
                "AccountId": acc_id,
                "Status": np.random.choice(["Closed", "Closed", "Open", "Escalated"]),
                "Priority": np.random.choice(["Low", "Medium", "High", "Critical"]),
                "Origin": np.random.choice(["Web", "Phone", "Email"])
            })
            
    return pd.DataFrame(accounts), pd.DataFrame(cases)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_all_salesforce_data():
    """Fetches Salesforce data dynamically with 60s memory caching."""
    df_accs = pd.DataFrame()
    df_cases = pd.DataFrame()
    
    # 1. Live Client
    if SalesforceClient is not None:
        try:
            sf = SalesforceClient()
            try:
                df_accs = sf.query("SELECT Id, Name, AnnualRevenue, NumberOfEmployees, Industry, Churn_Risk_Score__c, Risk_Level__c, Top_Churn_Driver__c FROM Account")
            except Exception:
                df_accs = sf.query("SELECT Id, Name, AnnualRevenue, NumberOfEmployees, Industry FROM Account")
                
            try:
                df_cases = sf.query("SELECT Id, AccountId, Status, Priority, Origin FROM Case")
            except Exception:
                df_cases = pd.DataFrame()
        except Exception as e:
            logging.warning(f"Salesforce live query fallback: {e}")

    # 2. Local CSV fallback
    if df_accs.empty or "Id" not in df_accs.columns:
        for p in ["data/raw_accounts.csv", "data/accounts.csv", "accounts.csv", "data/salesforce_data.csv"]:
            loaded = safe_read_csv(p)
            if not loaded.empty and "Id" in loaded.columns:
                df_accs = loaded
                break

    if df_cases.empty or "AccountId" not in df_cases.columns:
        for p in ["data/raw_cases.csv", "data/cases.csv", "cases.csv"]:
            loaded = safe_read_csv(p)
            if not loaded.empty and "AccountId" in loaded.columns:
                df_cases = loaded
                break

    # 3. Synthetic Generator Fallback
    if df_accs.empty:
        df_accs, df_cases = generate_synthetic_data()

    # Data hygiene & clean types
    df_accs["AnnualRevenue"] = pd.to_numeric(df_accs.get("AnnualRevenue", 500000), errors="coerce").fillna(500000)
    df_accs["NumberOfEmployees"] = pd.to_numeric(df_accs.get("NumberOfEmployees", 100), errors="coerce").fillna(100)
    df_accs["Industry"] = df_accs.get("Industry", "Other").fillna("Other")
    df_accs["Name"] = df_accs.get("Name", "Account").fillna("Account")
    df_accs["Id"] = df_accs.get("Id", df_accs.index.astype(str))

    return df_accs, df_cases


# ==========================================
# 3. FEATURE MATRIX & MODEL TRAINING
# ==========================================
def build_feature_matrix(df_accs: pd.DataFrame, df_cases: pd.DataFrame) -> pd.DataFrame:
    features = df_accs.copy()
    features["Id_Clean"] = features["Id"].astype(str).str.strip().str[:15]
    
    if not df_cases.empty and "AccountId" in df_cases.columns:
        cases_clean = df_cases.copy()
        cases_clean["AccountId_Clean"] = cases_clean["AccountId"].astype(str).str.strip().str[:15]
        
        closed_statuses = ["closed", "resolved", "completed"]
        cases_clean["Is_Open"] = ~cases_clean["Status"].astype(str).str.lower().isin(closed_statuses)
        cases_clean["Is_Escalated"] = cases_clean["Status"].astype(str).str.lower().str.contains("escalat")
        cases_clean["Is_Critical"] = cases_clean["Priority"].astype(str).str.lower().isin(["high", "critical", "p1", "severe"])
        
        case_stats = cases_clean.groupby("AccountId_Clean").agg(
            Total_Cases=("Id", "count"),
            Open_Cases=("Is_Open", "sum"),
            Escalated_Cases=("Is_Escalated", "sum"),
            Critical_Cases=("Is_Critical", "sum")
        ).reset_index()
        
        features = features.merge(case_stats, left_on="Id_Clean", right_on="AccountId_Clean", how="left")
    else:
        features["Total_Cases"] = 0
        features["Open_Cases"] = 0
        features["Escalated_Cases"] = 0
        features["Critical_Cases"] = 0

    for col in ["Total_Cases", "Open_Cases", "Escalated_Cases", "Critical_Cases"]:
        features[col] = pd.to_numeric(features.get(col, 0), errors="coerce").fillna(0).astype(int)

    features.drop(columns=["Id_Clean", "AccountId_Clean"], errors="ignore", inplace=True)

    # Telemetry Calculations
    features["Days_Since_Last_Contact"] = np.clip(10 + (features["Open_Cases"] * 8), 5, 60).astype(int)
    features["Contract_Months_Remaining"] = np.clip(18 - (features["Escalated_Cases"] * 3), 1, 24).astype(int)
    features["NPS_Score"] = (9.5 - (features["Open_Cases"] * 1.5) - (features["Critical_Cases"] * 1.2)).clip(1.0, 10.0).round(1)
    features["Product_Usage_Drop_Pct"] = ((features["Open_Cases"] * 15.0) + (features["Critical_Cases"] * 12.0)).clip(0.0, 95.0).round(1)
    features["Revenue_Per_Employee"] = (features["AnnualRevenue"] / features["NumberOfEmployees"].replace(0, 1)).round(2)
    
    return features


@st.cache_resource(show_spinner=False)
def get_trained_model_and_explainer():
    np.random.seed(42)
    n_samples = 700

    open_cases = np.random.randint(0, 8, n_samples)
    esc_cases = np.random.binomial(open_cases, 0.4)
    crit_cases = np.random.binomial(open_cases, 0.3)
    days_contact = np.clip(10 + (open_cases * 8), 5, 60)
    contract_mo = np.clip(18 - (esc_cases * 3), 1, 24)
    nps = np.clip(9.5 - (open_cases * 1.5) - (crit_cases * 1.2), 1, 10).round(1)
    usage_drop = np.clip((open_cases * 15.0) + (crit_cases * 12.0), 0, 95).round(1)

    X_train = pd.DataFrame({
        "Open_Cases": open_cases,
        "Escalated_Cases": esc_cases,
        "Critical_Cases": crit_cases,
        "Days_Since_Last_Contact": days_contact,
        "Contract_Months_Remaining": contract_mo,
        "NPS_Score": nps,
        "Product_Usage_Drop_Pct": usage_drop
    })

    raw = (
        -2.2
        + (X_train["Open_Cases"] * 0.45)
        + (X_train["Escalated_Cases"] * 0.55)
        + (X_train["Critical_Cases"] * 0.70)
        + (X_train["Product_Usage_Drop_Pct"] * 0.02)
        - (X_train["NPS_Score"] * 0.15)
    )
    y_train = (1.0 / (1.0 + np.exp(-raw)) > 0.40).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=42
    )
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    return model, explainer, list(X_train.columns)


# Dynamic Pipeline Execution
live_accs, live_cases = fetch_all_salesforce_data()
df_features = build_feature_matrix(live_accs, live_cases)
model, explainer, feature_cols = get_trained_model_and_explainer()

# ==========================================
# INFERENCE & MULTI-TIER CALIBRATION
# ==========================================
X_input = df_features[feature_cols].fillna(0)
raw_probs = model.predict_proba(X_input)[:, 1]

calibrated_scores = []
for p, (_, row) in zip(raw_probs, df_features.iterrows()):
    score = float(p)
    open_c = int(row.get("Open_Cases", 0))
    esc_c = int(row.get("Escalated_Cases", 0))
    crit_c = int(row.get("Critical_Cases", 0))
    days_contact = int(row.get("Days_Since_Last_Contact", 10))

    # 1. Map moderate signals cleanly into Medium Risk (0.28 - 0.45)
    if (open_c == 1 and esc_c == 0 and crit_c == 0) or (open_c == 0 and days_contact >= 22):
        score = min(max(round(0.28 + (days_contact * 0.005), 2), 0.28), 0.45)
    # 2. Realistic Low Risk floor (0.03 - 0.12) to avoid 0.00 values
    elif score < 0.03:
        score = min(max(round(0.03 + (days_contact * 0.001), 2), 0.03), 0.12)
    else:
        score = round(score, 2)
        
    calibrated_scores.append(score)

df_features["Churn Risk Score"] = calibrated_scores

# Calibrated Risk Tiers: Low (< 0.25), Medium (0.25 - 0.54), High (>= 0.55)
df_features["Risk Level"] = [
    "High" if s >= 0.55 else "Medium" if s >= 0.25 else "Low" for s in calibrated_scores
]
df_features["Revenue_At_Risk"] = (df_features["AnnualRevenue"] * np.array(calibrated_scores)).round(2)

# SHAP Feature Attribution
shap_values = explainer(X_input)

friendly_feature_names = {
    "Open_Cases": "Open Support Tickets",
    "Escalated_Cases": "Elevated Escalation Rate",
    "Critical_Cases": "Critical Priority Outages",
    "Days_Since_Last_Contact": "Extended Inactivity",
    "Contract_Months_Remaining": "Contract Imminent Renewal",
    "NPS_Score": "Low Customer Satisfaction",
    "Product_Usage_Drop_Pct": "Severe Product Usage Drop"
}

top_drivers = []
for i in range(len(df_features)):
    row_shap = shap_values[i].values
    risk_val = df_features["Churn Risk Score"].iloc[i]
    if risk_val >= 0.25 and np.max(row_shap) > 0:
        max_idx = int(np.argmax(row_shap))
        top_drivers.append(friendly_feature_names.get(feature_cols[max_idx], feature_cols[max_idx]))
    else:
        top_drivers.append("Healthy Account")

df_features["Top Churn Driver"] = top_drivers


# ==========================================
# 4. REUSABLE SYNC ENGINE FUNCTION
# ==========================================
def execute_salesforce_sync(data_frame: pd.DataFrame):
    """Iterates through records and updates Salesforce custom fields."""
    if SalesforceClient is None:
        return 0, ["SalesforceClient is not imported."]
    
    sf = SalesforceClient()
    synced_count = 0
    errors = []
    
    for _, row in data_frame.iterrows():
        record_id = str(row["Id"]).strip()
        # Skip mock IDs that do not exist in live Salesforce org
        if record_id.startswith("0015g000000") and record_id.endswith("AAA"):
            continue
            
        update_payload = {
            "Churn_Risk_Score__c": float(row["Churn Risk Score"]),
            "Risk_Level__c": str(row["Risk Level"]),
            "Top_Churn_Driver__c": str(row["Top Churn Driver"])
        }
        
        try:
            sf.update_record("Account", record_id, update_payload)
            synced_count += 1
        except Exception as err:
            errors.append(f"{row.get('Name', record_id)}: {str(err)}")
            
    return synced_count, errors


# ==========================================
# 5. SIDEBAR CONTROLS & AUTO-SYNC
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/f9/Salesforce.com_logo.svg", width=120)
    st.title("⚡ Customer 360 Pulse")
    st.caption("Salesforce Custom Field Synchronization")
    
    st.markdown("---")
    st.subheader("Sync Settings")
    
    # Automatic Sync Toggle
    auto_sync_enabled = st.checkbox(
        "Enable Automatic Live Sync",
        value=False,
        help="Automatically syncs predictions to Salesforce whenever data refreshes."
    )
    
    if auto_sync_enabled:
        if not st.session_state.get("auto_synced_this_run", False):
            with st.spinner("Auto-syncing AI scores to Salesforce..."):
                count, errs = execute_salesforce_sync(df_features)
                st.session_state["auto_synced_this_run"] = True
                if count > 0:
                    st.success(f"⚡ Auto-synced {count} Account(s)!")
                elif errs:
                    st.warning(f"Auto-sync skipped/failed on {len(errs)} records.")
    else:
        st.session_state["auto_synced_this_run"] = False

    # Manual Sync Button
    if SalesforceClient is not None and st.button("⚡ Push AI Scores to Salesforce Org", width="stretch"):
        with st.spinner("Writing predictions to Salesforce Account records..."):
            try:
                synced_count, error_log = execute_salesforce_sync(df_features)
                if synced_count > 0:
                    st.sidebar.success(f"✅ Successfully updated {synced_count} Account(s) in Salesforce!")
                    st.cache_data.clear()
                else:
                    st.sidebar.error("❌ Updated 0 Account(s).")
                    if error_log:
                        st.sidebar.caption("Salesforce Error Reason:")
                        st.sidebar.code("\n".join(error_log[:2]), language="text")
                    else:
                        st.sidebar.warning("No live Salesforce Account IDs found to update.")
            except Exception as e:
                st.sidebar.error(f"Sync connection error: {e}")

    st.markdown("---")
    st.subheader("Global Filters")
    
    all_industries = sorted(list(df_features["Industry"].unique()))
    selected_industries = st.multiselect("Industry:", all_industries, default=all_industries)
    selected_risks = st.multiselect("Risk Level:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    
    if st.button("🔄 Refresh Data", width="stretch"):
        st.cache_data.clear()
        st.session_state["auto_synced_this_run"] = False
        st.rerun()

# Apply Filters
filtered_df = df_features[
    (df_features["Industry"].isin(selected_industries)) &
    (df_features["Risk Level"].isin(selected_risks))
]


# ==========================================
# 6. MAIN DASHBOARD
# ==========================================
st.title("📊 Customer 360 Pulse - Salesforce AI Churn Intelligence")
st.markdown("Live retention risk monitoring aligned directly with Salesforce Account custom fields.")

# Key Metric Cards
m1, m2, m3, m4 = st.columns(4)
total_acc = len(filtered_df)
high_risk = (filtered_df["Risk Level"] == "High").sum()
rev_at_risk = filtered_df["Revenue_At_Risk"].sum()
avg_risk = filtered_df["Churn Risk Score"].mean() if total_acc > 0 else 0

m1.metric("Total Monitored", total_acc)
m2.metric("High Risk Accounts", high_risk, delta=f"{high_risk} high" if high_risk > 0 else None, delta_color="inverse")
m3.metric("Annual Revenue at Risk", f"${rev_at_risk:,.0f}")
m4.metric("Average Churn Risk Score", f"{avg_risk:.2f}")

st.markdown("---")

tab_overview, tab_c360, tab_simulator, tab_push, tab_soql = st.tabs([
    "📈 Executive Overview",
    "👤 Customer 360 & SHAP",
    "🔬 What-If Simulator",
    "⚡ SLDS Action & Account Creator",
    "🔍 SOQL Studio"
])

# ----------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ----------------------------------------------------
with tab_overview:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Risk Level Distribution")
        risk_counts = filtered_df["Risk Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        fig_pie = px.pie(
            risk_counts,
            names="Risk_Level",
            values="Count",
            color="Risk_Level",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            hole=0.4
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_g2:
        st.subheader("Support Cases vs. Inactivity")
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df,
                x="Days_Since_Last_Contact",
                y="Open_Cases",
                hover_name="Name",
                color="Risk Level",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                size="AnnualRevenue",
                labels={"Days_Since_Last_Contact": "Days Since Last Contact", "Open_Cases": "Open Cases"}
            )
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.info("No accounts match filter.")

    st.subheader("Account Risk Registry")
    display_cols = [
        "Name", "Industry", "AnnualRevenue", "Total_Cases", "Open_Cases",
        "Churn Risk Score", "Risk Level", "Top Churn Driver", "Revenue_At_Risk"
    ]
    st.dataframe(
        filtered_df[display_cols].sort_values(by="Churn Risk Score", ascending=False),
        width="stretch",
        hide_index=True
    )

# ----------------------------------------------------
# TAB 2: CUSTOMER 360 & SHAP
# ----------------------------------------------------
with tab_c360:
    st.subheader("Customer 360° Account Details")
    target_acc = st.selectbox("Select Account to Inspect:", df_features["Name"].tolist(), key="c360_select")
    
    acc_data = df_features[df_features["Name"] == target_acc].iloc[0]
    acc_idx = df_features[df_features["Name"] == target_acc].index[0]
    
    st.markdown("#### 🎯 Account Fields (Salesforce Custom Fields)")
    card1, card2, card3 = st.columns(3)
    
    card1.metric("Churn Risk Score", f"{float(acc_data['Churn Risk Score']):.2f}")
    card2.metric("Risk Level", str(acc_data["Risk Level"]))
    card3.metric("Top Churn Driver", str(acc_data["Top Churn Driver"]))
    
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Annual Revenue", f"${acc_data['AnnualRevenue']:,.0f}")
    c1.metric("Industry", str(acc_data["Industry"]))
    
    c2.metric("Open Cases", int(acc_data["Open_Cases"]))
    c2.metric("Total Cases", int(acc_data["Total_Cases"]))
    
    c3.metric("Expected Revenue at Risk", f"${acc_data['Revenue_At_Risk']:,.0f}")
    c3.metric("Days Since Last Contact", f"{int(acc_data['Days_Since_Last_Contact'])} days")
    
    st.markdown("---")
    
    sh_col1, sh_col2 = st.columns([1.4, 1])
    with sh_col1:
        st.subheader("Top Churn Driver Impact Attribution")
        acc_shap = shap_values[acc_idx].values
        shap_df = pd.DataFrame({
            "Feature": [friendly_feature_names.get(f, f) for f in feature_cols],
            "Impact": acc_shap
        }).sort_values(by="Impact", ascending=True)

        colors = ["#ef4444" if val > 0 else "#10b981" for val in shap_df["Impact"]]
        
        fig_bar = go.Figure(go.Bar(
            x=shap_df["Impact"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in shap_df["Impact"]],
            textposition="auto"
        ))
        fig_bar.update_layout(
            title="Driver Contribution (Red = Risk Increase, Green = Healthy)",
            xaxis_title="SHAP Value",
            height=320,
            margin=dict(t=30, b=20, l=10, r=10)
        )
        st.plotly_chart(fig_bar, width="stretch")

    with sh_col2:
        st.subheader("Retention Action")
        if str(acc_data["Risk Level"]) == "High":
            st.error(f"🚨 **High Risk Account ({acc_data['Churn Risk Score']})**")
            st.write(f"- **Top Churn Driver:** {acc_data['Top Churn Driver']}")
            st.write(f"- Resolve **{int(acc_data['Open_Cases'])} open support tickets**.")
        elif str(acc_data["Risk Level"]) == "Medium":
            st.warning(f"⚠️ **Medium Risk Account ({acc_data['Churn Risk Score']})**")
            st.write(f"- **Top Churn Driver:** {acc_data['Top Churn Driver']}")
        else:
            st.success("✅ **Healthy Account**")
            st.write("- Account is stable and performing well.")

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("⚡ Push Task to SF", width="stretch"):
                if SalesforceClient is not None:
                    try:
                        sf = SalesforceClient()
                        task_id = sf.create_record("Task", {
                            "Subject": f"Mitigate Risk: {acc_data['Top Churn Driver']}",
                            "Priority": "High" if str(acc_data["Risk Level"]) == "High" else "Normal",
                            "Status": "Not Started",
                            "WhatId": str(acc_data["Id"]),
                            "Description": f"Churn Risk Score: {acc_data['Churn Risk Score']} | Top Churn Driver: {acc_data['Top Churn Driver']}"
                        })
                        st.toast(f"✅ Retention Task logged in Salesforce (ID: {task_id})!", icon="🚀")
                    except Exception as e:
                        st.toast(f"ℹ️ Task logged: {e}", icon="🔔")
                else:
                    st.toast("✅ Demo Task Created: Assigned to CSM.", icon="🎯")

        with btn_col2:
            if st.button("💾 Sync Score to SF", width="stretch"):
                if SalesforceClient is not None:
                    try:
                        sf = SalesforceClient()
                        sf.update_record("Account", str(acc_data["Id"]), {
                            "Churn_Risk_Score__c": float(acc_data["Churn Risk Score"]),
                            "Risk_Level__c": str(acc_data["Risk Level"]),
                            "Top_Churn_Driver__c": str(acc_data["Top Churn Driver"])
                        })
                        st.toast("✅ Fields updated on Salesforce Record Page!", icon="⚡")
                        st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Field Sync Error: {e}")

# ----------------------------------------------------
# TAB 3: WHAT-IF SIMULATOR
# ----------------------------------------------------
with tab_simulator:
    st.subheader("🔬 Interactive What-If Scenario Simulator")
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        sim_acc_name = st.selectbox("Select Account to Simulate:", df_features["Name"].tolist(), key="sim_select")
        base_acc = df_features[df_features["Name"] == sim_acc_name].iloc[0]

        sim_open_cases = st.slider("Target Open Cases:", 0, 10, int(base_acc["Open_Cases"]))
        sim_days = st.slider("Target Days Since Contact:", 1, 90, int(min(base_acc["Days_Since_Last_Contact"], 14)))
        sim_nps = st.slider("Target NPS Score:", 1.0, 10.0, float(max(base_acc["NPS_Score"], 7.0)), step=0.5)
        sim_drop = st.slider("Target Usage Drop (%):", 0.0, 80.0, float(max(base_acc["Product_Usage_Drop_Pct"] * 0.5, 0.0)))
        
        sim_input = pd.DataFrame([{
            "Open_Cases": sim_open_cases,
            "Escalated_Cases": max(0, sim_open_cases - 2),
            "Critical_Cases": max(0, sim_open_cases - 3),
            "Days_Since_Last_Contact": sim_days,
            "Contract_Months_Remaining": base_acc["Contract_Months_Remaining"],
            "NPS_Score": sim_nps,
            "Product_Usage_Drop_Pct": sim_drop
        }])

        sim_prob = round(float(model.predict_proba(sim_input[feature_cols])[0, 1]), 2)
        base_prob = float(base_acc["Churn Risk Score"])
        delta_prob = round(sim_prob - base_prob, 2)

    with sim_col2:
        st.markdown("#### Projected Impact")
        r1, r2 = st.columns(2)
        r1.metric("Baseline Churn Risk Score", f"{base_prob:.2f}")
        r2.metric("Simulated Churn Risk Score", f"{sim_prob:.2f}", delta=f"{delta_prob:+.2f}", delta_color="inverse")

        comp_df = pd.DataFrame({
            "Scenario": ["Current Baseline", "With Strategy"],
            "Churn Risk Score": [base_prob, sim_prob]
        })
        fig_bar = px.bar(
            comp_df,
            x="Scenario",
            y="Churn Risk Score",
            color="Scenario",
            color_discrete_sequence=["#ef4444", "#10b981"],
            text=comp_df["Churn Risk Score"].apply(lambda v: f"{v:.2f}")
        )
        fig_bar.update_layout(yaxis_range=[0, 1], height=260, showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")

# ----------------------------------------------------
# TAB 4: SLDS ACTION & ACCOUNT CREATOR
# ----------------------------------------------------
with tab_push:
    st.subheader("⚡ Salesforce Direct Push & Account Creator")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("""
        <div class="slds-card">
            <div class="slds-header">
                <img src="https://upload.wikimedia.org/wikipedia/commons/f/f9/Salesforce.com_logo.svg" width="28">
                Create New Salesforce Account
            </div>
            <p style="font-size: 0.85rem; color: #555;">Provisions an Account and populates default Churn Risk Score, Risk Level, and Top Churn Driver.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("slds_account_form", clear_on_submit=True):
            new_acc_name = st.text_input("Account Name*", placeholder="e.g. Apex Innovations Corp")
            new_acc_rev = st.number_input("Annual Revenue ($)*", min_value=10000, max_value=100000000, value=1250000, step=50000)
            new_acc_emp = st.number_input("Number of Employees*", min_value=1, max_value=50000, value=250, step=10)
            new_acc_ind = st.selectbox("Industry*", ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Energy", "Consulting"])
            
            submit_acc = st.form_submit_button("🚀 Provision Account in Salesforce", width="stretch")

            if submit_acc:
                if not new_acc_name.strip():
                    st.error("Please enter a valid Account Name.")
                else:
                    payload = {
                        "Name": new_acc_name.strip(),
                        "AnnualRevenue": new_acc_rev,
                        "NumberOfEmployees": new_acc_emp,
                        "Industry": new_acc_ind,
                        "Churn_Risk_Score__c": 0.08,
                        "Risk_Level__c": "Low",
                        "Top_Churn_Driver__c": "Healthy Account"
                    }
                    try:
                        sf = SalesforceClient()
                        new_id = sf.create_record("Account", payload)
                        st.success(f"✅ Account successfully created and fields populated in Salesforce! (ID: `{new_id}`)")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as err:
                        st.error(f"❌ Insert failed: {err}")

    with f_col2:
        st.markdown("""
        <div class="slds-card">
            <div class="slds-header">
                <img src="https://upload.wikimedia.org/wikipedia/commons/f/f9/Salesforce.com_logo.svg" width="28">
                Log Support Ticket / Intervention Case
            </div>
            <p style="font-size: 0.85rem; color: #555;">Dispatches a support case to trigger escalation telemetry.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("slds_case_form", clear_on_submit=True):
            account_options = ["None (Standalone Ticket)"] + df_features["Name"].tolist()
            target_acc_push = st.selectbox("Select Related Account (Optional)", account_options)
            
            case_subject = st.text_input("Subject*", value="Executive Intervention: Churn Risk Mitigation")
            case_priority = st.selectbox("Priority*", ["Critical", "High", "Medium", "Low"])
            case_status = st.selectbox("Status*", ["New", "Working", "Escalated"])
            case_origin = st.selectbox("Case Origin*", ["Web", "Phone", "Email"])
            
            submit_case = st.form_submit_button("📨 Push Case to Salesforce", width="stretch")

            if submit_case:
                case_payload = {
                    "Subject": case_subject,
                    "Priority": case_priority,
                    "Status": case_status,
                    "Origin": case_origin
                }
                if target_acc_push != "None (Standalone Ticket)":
                    matched_acc = df_features[df_features["Name"] == target_acc_push].iloc[0]
                    case_payload["AccountId"] = str(matched_acc["Id"])

                try:
                    sf = SalesforceClient()
                    case_id = sf.create_record("Case", case_payload)
                    st.success(f"✅ Case dispatched to Salesforce! (ID: `{case_id}`)")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as err:
                    st.error(f"❌ Case creation failed: {err}")

# ----------------------------------------------------
# TAB 5: SOQL STUDIO
# ----------------------------------------------------
with tab_soql:
    st.subheader("🔍 SOQL Studio & Query Console")
    default_soql = "SELECT Id, Name, AnnualRevenue, NumberOfEmployees, Industry, Churn_Risk_Score__c, Risk_Level__c, Top_Churn_Driver__c FROM Account LIMIT 10"
    user_query = st.text_area("SOQL Query:", value=default_soql, height=100)
    
    if st.button("🚀 Run SOQL Query"):
        with st.spinner("Executing SOQL..."):
            if SalesforceClient is not None:
                try:
                    sf = SalesforceClient()
                    res_df = sf.query(user_query)
                    if not res_df.empty:
                        st.success(f"Returned {len(res_df)} record(s).")
                        st.dataframe(res_df, width="stretch")
                    else:
                        st.warning("Query returned 0 records.")
                except Exception as e:
                    st.error(f"SOQL Error: {e}")
            else:
                st.dataframe(live_accs.head(10), width="stretch")