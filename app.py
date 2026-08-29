"""
Customer 360 Risk Pulse™ | Enterprise Retention & Churn AI for Salesforce
Complete Unified Application: Live SOQL Studio, SLDS Design, Dynamic ML Inference, Robust Case Telemetry & Reverse-ETL.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Path configuration for source modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from sf_client import SalesforceClient
from sync import sync_predictions_to_salesforce

# ----------------------------------------------------
# Page Configuration & SLDS Theme
# ----------------------------------------------------
st.set_page_config(
    page_title="Customer 360 Risk Pulse™ | Salesforce AI",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/f/f9/Salesforce.com_logo.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

SF_SVG_ICON = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 70" width="36" height="26" style="vertical-align: middle;">
    <path fill="#00A1E0" d="M39.6,12.5 C43.8,6.8 51.5,3.2 59.8,4.5 C68.3,5.9 74.9,13.2 75.3,21.8 C80.6,23.1 84.8,27.2 86.4,32.4 C88.1,37.6 86.9,43.4 83.3,47.4 C87.3,49.2 90.1,53.2 90.6,57.7 C91.1,62.1 89.2,66.5 85.6,69.2 C82,71.9 77.1,72.6 72.8,71.2 L19.4,71.2 C13.4,71.2 8,67.6 5.6,62.1 C3.2,56.6 4.3,50.2 8.4,45.8 C6.4,43.2 5.5,39.9 5.8,36.6 C6.1,33.3 7.7,30.3 10.1,28.1 C12.6,25.9 15.8,24.8 19.1,25 C19.9,25 20.8,25.2 21.6,25.5 C23.8,17.4 30.6,12.2 39.6,12.5 Z"/>
</svg>
"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    * {{ font-family: 'Segoe UI', 'Plus Jakarta Sans', -apple-system, sans-serif; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 96%; }}

    .slds-brand-header {{
        background: linear-gradient(135deg, #0176D3 0%, #014486 100%);
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 20px;
        color: #FFFFFF;
        box-shadow: 0 4px 16px rgba(1, 118, 211, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .slds-title {{ font-size: 24px; font-weight: 700; margin: 0; color: #FFFFFF; display: flex; align-items: center; gap: 12px; }}
    .slds-badge {{
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }}
    .slds-subtitle {{ color: #E0E8F5; font-size: 13px; margin-top: 5px; margin-bottom: 0; }}

    .slds-tile {{
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-top: 4px solid #0176D3;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }}
    .slds-tile-danger {{ border-top: 4px solid #BA0517; }}
    .slds-tile-warning {{ border-top: 4px solid #DD7A01; }}
    .slds-tile-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: #64748B; }}
    .slds-tile-value {{ font-size: 28px; font-weight: 700; color: #0F172A; margin: 4px 0; }}
    .slds-tile-footer {{ font-size: 12px; font-weight: 500; }}
    .text-danger {{ color: #BA0517; font-weight: 600; }}
    .text-warning {{ color: #DD7A01; font-weight: 600; }}
    .text-neutral {{ color: #64748B; }}

    div[data-testid="stSidebar"] {{ background-color: #032D60; border-right: 1px solid #011E42; }}
    div[data-testid="stSidebar"] * {{ color: #F8FAFC; }}
    div[data-testid="stSidebar"] label {{ color: #94A3B8 !important; font-size: 11px; font-weight: 700; text-transform: uppercase; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Backend Inference & Extraction Engine
# ----------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = "models/xgb_churn_model.joblib"
    if not os.path.exists(model_path):
        return None, None
    payload = joblib.load(model_path)
    return payload["model"], payload["feature_names"]


def parse_boolean(val, status=""):
    """Robust parser that marks True if IsEscalated is truthy OR Status is Escalated."""
    if str(status).strip().lower() == "escalated":
        return True
    if pd.isna(val):
        return False
    str_val = str(val).strip().lower()
    return str_val in ["true", "1", "t", "yes", "y"]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_salesforce_data():
    sf = SalesforceClient()
    
    acc_query = """
        SELECT Id, Name, Industry, Type, AnnualRevenue, 
               Churn_Risk_Score__c, Risk_Level__c, Top_Churn_Driver__c, CreatedDate 
        FROM Account 
        ORDER BY CreatedDate DESC
    """
    df_accs = sf.query(acc_query)
    
    case_query = """
        SELECT Id, CaseNumber, AccountId, Subject, Priority, Status, Type, Origin, IsEscalated, CreatedDate 
        FROM Case 
        ORDER BY CreatedDate DESC
    """
    try:
        df_cases = sf.query(case_query)
        if not df_cases.empty:
            if "IsEscalated" in df_cases.columns and "Status" in df_cases.columns:
                df_cases["IsEscalated"] = [
                    parse_boolean(esc, stat) for esc, stat in zip(df_cases["IsEscalated"], df_cases["Status"])
                ]
            elif "IsEscalated" in df_cases.columns:
                df_cases["IsEscalated"] = df_cases["IsEscalated"].apply(parse_boolean)
            elif "Status" in df_cases.columns:
                df_cases["IsEscalated"] = df_cases["Status"].apply(lambda s: str(s).strip().lower() == "escalated")
            else:
                df_cases["IsEscalated"] = False
    except Exception:
        df_cases = pd.DataFrame()
        
    return df_accs, df_cases


def score_live_accounts(df_accs: pd.DataFrame, df_cases: pd.DataFrame, model, feature_names: list) -> pd.DataFrame:
    if not df_cases.empty and "AccountId" in df_cases.columns:
        case_agg = df_cases.groupby("AccountId").agg(
            Total_Cases=("Id", "count"),
            Escalated_Cases=("IsEscalated", lambda x: sum([1 for v in x if bool(v)])),
            Critical_Cases=("Priority", lambda x: sum([1 for v in x if str(v) == "Critical"])),
            High_Cases=("Priority", lambda x: sum([1 for v in x if str(v) == "High"]))
        ).reset_index()
        case_agg["Escalation_Rate"] = (case_agg["Escalated_Cases"] / case_agg["Total_Cases"]).fillna(0.0)
    else:
        case_agg = pd.DataFrame(columns=["AccountId", "Total_Cases", "Escalated_Cases", "Critical_Cases", "High_Cases", "Escalation_Rate"])

    df = pd.merge(df_accs, case_agg, left_on="Id", right_on="AccountId", how="left")
    
    df["Total_Cases"] = df["Total_Cases"].fillna(0).astype(int)
    df["Escalated_Cases"] = df["Escalated_Cases"].fillna(0).astype(int)
    df["Critical_Cases"] = df["Critical_Cases"].fillna(0).astype(int)
    df["High_Cases"] = df["High_Cases"].fillna(0).astype(int)
    df["Escalation_Rate"] = df["Escalation_Rate"].fillna(0.0)
    df["Avg_Days_To_Resolve"] = 3.5
    df["Tenure_Months"] = 12
    
    df["AnnualRevenue"] = pd.to_numeric(df["AnnualRevenue"], errors="coerce").fillna(35000.0)
    df["Monthly_Charges"] = (df["AnnualRevenue"] / 12).round(2)
    df["Total_Charges"] = df["Monthly_Charges"] * df["Tenure_Months"]
    df["Est_Annual_Value"] = df["AnnualRevenue"]
    df["Avg_Monthly_Case_Load"] = (df["Total_Cases"] / df["Tenure_Months"]).round(3)
    df["Contract_Type"] = "Month-to-Month"
    df["Industry"] = df["Industry"].fillna("Technology")

    X_encoded = pd.get_dummies(df[[
        "Tenure_Months", "Monthly_Charges", "Total_Charges", "Total_Cases",
        "Escalated_Cases", "Avg_Days_To_Resolve", "Critical_Cases", "High_Cases",
        "Escalation_Rate", "Avg_Monthly_Case_Load", "Est_Annual_Value",
        "Industry", "Contract_Type"
    ]], drop_first=True)

    for col in feature_names:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[feature_names]

    probs = model.predict_proba(X_encoded)[:, 1]
    df["Predicted_Churn_Prob"] = np.round(probs, 2)
    df["Predicted_Risk_Level"] = pd.cut(
        df["Predicted_Churn_Prob"],
        bins=[-0.01, 0.35, 0.70, 1.0],
        labels=["Low", "Medium", "High"]
    )
    
    def get_quick_driver(row):
        if row["Escalation_Rate"] > 0.3:
            return "Elevated Escalation Velocity"
        elif row["Total_Cases"] >= 3:
            return "Elevated Support Ticket Volume"
        elif row["Monthly_Charges"] > 3000:
            return "High Monthly Recurring Spend"
        return "Stable Account / High Engagement"

    df["Top_Churn_Driver"] = df.apply(
        lambda r: r["Top_Churn_Driver__c"] if pd.notna(r["Top_Churn_Driver__c"]) and r["Top_Churn_Driver__c"] != "" else get_quick_driver(r),
        axis=1
    )
    return df


# ----------------------------------------------------
# Pipeline Initialization
# ----------------------------------------------------
model, feature_names = load_trained_model()
if model is None:
    st.error("Model artifacts missing in `models/xgb_churn_model.joblib`. Run `python src/train.py` first.")
    st.stop()

with st.spinner("Connecting to Salesforce CLI..."):
    live_accs, live_cases = fetch_all_salesforce_data()
    if not live_accs.empty:
        df = score_live_accounts(live_accs, live_cases, model, feature_names)
    else:
        df = pd.read_csv("data/scored_accounts.csv")


# ----------------------------------------------------
# Sidebar: Creation Studio & Controls
# ----------------------------------------------------
with st.sidebar:
    st.markdown(f"### {SF_SVG_ICON} **Salesforce Ops**", unsafe_allow_html=True)
    st.caption(f"Connected Org: `my-dev-org` | {len(df)} Live Accounts")
    
    if st.button("⚡ Quick Reload (Clear Cache)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    with st.expander("➕ New Salesforce Account", expanded=False):
        with st.form("new_account_form", clear_on_submit=True):
            new_name = st.text_input("Account Name*", placeholder="e.g. Apex Global")
            new_industry = st.selectbox("Industry", ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Education"])
            new_revenue = st.number_input("Annual Revenue ($)", min_value=5000, max_value=10000000, value=75000, step=5000)
            new_type = st.selectbox("Type", ["Customer - Direct", "Customer - Channel", "Prospect"])
            
            if st.form_submit_button("⚡ Create in Salesforce", use_container_width=True):
                if new_name.strip():
                    sf = SalesforceClient()
                    sf.create_record("Account", {"Name": new_name.strip(), "Industry": new_industry, "AnnualRevenue": int(new_revenue), "Type": new_type})
                    st.success(f"Created: {new_name}")
                    st.cache_data.clear()
                    st.rerun()

    with st.expander("🎫 Log Support Case", expanded=False):
        with st.form("new_case_form", clear_on_submit=True):
            target_acc = st.selectbox("Attach to Account:", options=df["Name"].unique())
            case_subject = st.text_input("Subject*", placeholder="e.g. API Latency & 504 Errors")
            case_priority = st.selectbox("Priority", ["Critical", "High", "Medium", "Low"])
            case_type = st.selectbox("Type", ["Technical / Bug", "Billing", "Performance / Latency", "Feature Request"])
            case_origin = st.selectbox("Case Origin*", ["Web", "Phone", "Email"])
            is_esc = st.checkbox("Escalate Ticket Directly")
            
            if st.form_submit_button("⚡ Log Case & Re-score", use_container_width=True):
                if case_subject.strip():
                    target_id = df[df["Name"] == target_acc]["Id"].iloc[0]
                    sf = SalesforceClient()
                    sf.create_record("Case", {
                        "AccountId": target_id,
                        "Subject": case_subject.strip(),
                        "Priority": case_priority,
                        "Type": case_type,
                        "Origin": case_origin,
                        "Status": "Escalated" if is_esc else "New",
                        "IsEscalated": "true" if is_esc else "false"
                    })
                    st.success(f"Case logged to {target_acc}!")
                    st.cache_data.clear()
                    st.rerun()

    st.markdown("#### 🎯 Filter Cohorts")
    selected_industries = st.multiselect("Industry", options=sorted(df["Industry"].dropna().unique()), default=sorted(df["Industry"].dropna().unique()))
    selected_risk_levels = st.multiselect("Risk Category", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])

    filtered_df = df[(df["Industry"].isin(selected_industries)) & (df["Predicted_Risk_Level"].isin(selected_risk_levels))]

    st.markdown("---")
    if st.button("🚀 Push Live to Salesforce", use_container_width=True, type="primary"):
        with st.status("Writing Predictions to Salesforce CRM...", expanded=True) as status:
            try:
                sf = SalesforceClient()
                for _, r in filtered_df.iterrows():
                    payload = {
                        "Churn_Risk_Score__c": float(r["Predicted_Churn_Prob"]),
                        "Risk_Level__c": str(r["Predicted_Risk_Level"]),
                        "Top_Churn_Driver__c": str(r["Top_Churn_Driver"])
                    }
                    sf.update_record("Account", r["Id"], payload)
                status.update(label="Sync Completed!", state="complete", expanded=False)
                st.toast(f"{len(filtered_df)} Accounts Updated in Salesforce", icon="☁️")
            except Exception as e:
                status.update(label=f"Sync Failed: {str(e)}", state="error")


# ----------------------------------------------------
# Main View & Metrics
# ----------------------------------------------------
st.markdown(f"""
<div class="slds-brand-header">
    <div>
        <h1 class="slds-title">{SF_SVG_ICON} Customer 360 Risk Pulse™</h1>
        <p class="slds-subtitle">AI-powered customer churn detection, TreeSHAP root-cause telemetry, and Salesforce Reverse-ETL.</p>
    </div>
    <span class="slds-badge">Enterprise Edition</span>
</div>
""", unsafe_allow_html=True)

total_accounts = len(filtered_df)
high_risk_df = filtered_df[filtered_df["Predicted_Risk_Level"] == "High"]
med_risk_df = filtered_df[filtered_df["Predicted_Risk_Level"] == "Medium"]
high_risk_count = len(high_risk_df)
med_risk_count = len(med_risk_df)
arr_at_risk = high_risk_df["Est_Annual_Value"].sum()
total_cases_count = len(live_cases) if not live_cases.empty else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="slds-tile"><div class="slds-tile-label">Monitored Accounts</div><div class="slds-tile-value">{total_accounts:,}</div><div class="slds-tile-footer text-neutral">Live Salesforce feed</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="slds-tile slds-tile-danger"><div class="slds-tile-label">Severe Churn Risk</div><div class="slds-tile-value text-danger">{high_risk_count}</div><div class="slds-tile-footer text-danger">{(high_risk_count/max(1, total_accounts))*100:.1f}% of cohort</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="slds-tile slds-tile-danger"><div class="slds-tile-label">ARR at Immediate Risk</div><div class="slds-tile-value text-danger">${arr_at_risk:,.0f}</div><div class="slds-tile-footer text-danger">Requires CSM action</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="slds-tile slds-tile-warning"><div class="slds-tile-label">Total Support Tickets</div><div class="slds-tile-value text-warning">{total_cases_count}</div><div class="slds-tile-footer text-warning">Active Case Telemetry</div></div>', unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# ----------------------------------------------------
# Main Tabs
# ----------------------------------------------------
tab_portfolio, tab_inspection, tab_cases, tab_soql_studio, tab_crm_view = st.tabs([
    "📊 Portfolio Risk & Distribution",
    "🔍 Account Deep-Dive & Health Audit",
    "🎫 Associated Support Cases",
    "💻 SOQL Query Studio",
    "☁️ Live Salesforce CRM Data"
])

with tab_portfolio:
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.subheader("Cohort Risk Stratification")
        risk_counts = filtered_df["Predicted_Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]
        fig_donut = px.pie(risk_counts, names="Risk Tier", values="Count", color="Risk Tier", color_discrete_map={"High": "#BA0517", "Medium": "#DD7A01", "Low": "#2E844A"}, hole=0.55)
        fig_donut.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
        fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with r1_col2:
        st.subheader("Churn Probability Distribution Curve")
        fig_hist = px.histogram(filtered_df, x="Predicted_Churn_Prob", nbins=20, color="Predicted_Risk_Level", color_discrete_map={"High": "#BA0517", "Medium": "#DD7A01", "Low": "#2E844A"})
        fig_hist.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.subheader("ARR at Risk by Industry")
        ind_risk = filtered_df.groupby("Industry").agg(
            At_Risk_ARR=("Est_Annual_Value", lambda x: sum(x[filtered_df.loc[x.index, "Predicted_Risk_Level"] == "High"])),
            Total_ARR=("Est_Annual_Value", "sum")
        ).reset_index()
        fig_ind = px.bar(ind_risk, x="Industry", y=["At_Risk_ARR", "Total_ARR"], barmode="group", color_discrete_map={"At_Risk_ARR": "#BA0517", "Total_ARR": "#0176D3"})
        fig_ind.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_ind, use_container_width=True)

    with r2_col2:
        st.subheader("Support Friction Matrix")
        fig_friction = px.scatter(
            filtered_df, x="Total_Cases", y="Escalation_Rate", size="AnnualRevenue", color="Predicted_Risk_Level",
            hover_name="Name", color_discrete_map={"High": "#BA0517", "Medium": "#DD7A01", "Low": "#2E844A"}
        )
        fig_friction.update_layout(height=340, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_friction, use_container_width=True)

with tab_inspection:
    st.subheader("Account Deep-Dive & Diagnostic Audit")
    acc_list = filtered_df["Name"].tolist()
    if acc_list:
        selected_account = st.selectbox("Select Account from Salesforce Org:", options=acc_list)
        row = filtered_df[filtered_df["Name"] == selected_account].iloc[0]
        da1, da2, da3 = st.columns([1.2, 1, 1])
        
        with da1:
            prob_pct = row["Predicted_Churn_Prob"] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=prob_pct, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Churn Risk (%)", 'font': {'size': 14}},
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#080707"},
                       'steps': [{'range': [0, 35], 'color': '#D1FAE5'}, {'range': [35, 70], 'color': '#FEF3C7'}, {'range': [70, 100], 'color': '#FEE2E2'}],
                       'threshold': {'line': {'color': "#BA0517", 'width': 4}, 'thickness': 0.75, 'value': prob_pct}}
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with da2:
            st.markdown("#### 🔎 Root-Cause Analysis")
            st.info(f"**Dominant SHAP Driver:**\n\n### {row['Top_Churn_Driver']}")
            st.metric("Assigned Tier", str(row["Predicted_Risk_Level"]))

        with da3:
            st.markdown("#### 🎫 Support Telemetry")
            st.metric("Total Support Tickets", f"{int(row['Total_Cases'])} cases")
            st.metric("Ticket Escalation Rate", f"{row['Escalation_Rate']*100:.1f}%")

        st.markdown(f"##### 📋 Associated Support Cases for **{selected_account}**")
        if not live_cases.empty and "AccountId" in live_cases.columns:
            acc_cases = live_cases[live_cases["AccountId"] == row["Id"]]
            if not acc_cases.empty:
                show_cols = [c for c in ["CaseNumber", "Subject", "Priority", "Status", "Type", "Origin", "IsEscalated"] if c in acc_cases.columns]
                st.dataframe(acc_cases[show_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No support tickets currently logged for this account.")

with tab_cases:
    st.subheader("🎫 Salesforce Support Case Live Feed")
    if not live_cases.empty and "AccountId" in live_cases.columns:
        merged_cases = pd.merge(live_cases, df[["Id", "Name", "Predicted_Risk_Level", "Predicted_Churn_Prob"]], left_on="AccountId", right_on="Id", how="left")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.subheader("Cases by Priority & Escalation")
            fig_case_p = px.histogram(
                merged_cases,
                x="Priority",
                color="IsEscalated",
                barmode="group",
                color_discrete_map={True: "#BA0517", False: "#0176D3"},
                labels={"IsEscalated": "Is Escalated", "Priority": "Case Priority"}
            )
            fig_case_p.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_case_p, use_container_width=True)

        with c_col2:
            st.subheader("Cases by Parent Account Risk Tier")
            fig_case_r = px.pie(
                merged_cases,
                names="Predicted_Risk_Level",
                color="Predicted_Risk_Level",
                color_discrete_map={"High": "#BA0517", "Medium": "#DD7A01", "Low": "#2E844A"},
                hole=0.45
            )
            fig_case_r.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_case_r, use_container_width=True)

        st.markdown("##### 🔍 Full Case Directory")
        view_cols = [c for c in ["CaseNumber", "Name", "Subject", "Priority", "Status", "Origin", "IsEscalated", "Predicted_Risk_Level"] if c in merged_cases.columns]
        st.dataframe(
            merged_cases[view_cols].rename(columns={"Name": "Parent Account", "Predicted_Risk_Level": "Account Risk"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No live Support Cases found in this Salesforce org.")

with tab_soql_studio:
    st.subheader("💻 Interactive Salesforce SOQL Query Studio")
    preset = st.selectbox("Load Query Template:", [
        "Custom SOQL Query", "High-Risk Accounts (ML Fields)", "Cases with Parent Account Details",
        "All Accounts by Annual Revenue", "Recently Created Accounts"
    ])
    template_queries = {
        "Custom SOQL Query": "SELECT Id, Name, Industry, Type, AnnualRevenue FROM Account LIMIT 20",
        "High-Risk Accounts (ML Fields)": "SELECT Id, Name, Industry, Churn_Risk_Score__c, Risk_Level__c, Top_Churn_Driver__c FROM Account WHERE Risk_Level__c = 'High' ORDER BY Churn_Risk_Score__c DESC",
        "Cases with Parent Account Details": "SELECT Id, CaseNumber, Subject, Priority, Status, Origin, IsEscalated, Account.Name FROM Case ORDER BY CreatedDate DESC LIMIT 25",
        "All Accounts by Annual Revenue": "SELECT Id, Name, Industry, AnnualRevenue, CreatedDate FROM Account WHERE AnnualRevenue != NULL ORDER BY AnnualRevenue DESC LIMIT 25",
        "Recently Created Accounts": "SELECT Id, Name, CreatedDate, Industry FROM Account ORDER BY CreatedDate DESC LIMIT 15"
    }
    user_query = st.text_area("SOQL Query:", value=template_queries.get(preset, template_queries["Custom SOQL Query"]), height=100)
    if st.button("▶️ Execute SOQL", type="primary"):
        with st.spinner("Executing SOQL query via Salesforce CLI..."):
            try:
                sf = SalesforceClient()
                soql_results = sf.query(user_query.strip())
                if not soql_results.empty:
                    st.success(f"Query returned {len(soql_results)} record(s).")
                    st.dataframe(soql_results, use_container_width=True)
                    csv = soql_results.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results as CSV", data=csv, file_name="salesforce_query_export.csv", mime="text/csv")
                else:
                    st.info("Query returned 0 records.")
            except Exception as e:
                st.error(f"SOQL Error: {str(e)}")

with tab_crm_view:
    st.subheader("All Active Salesforce CRM Accounts")
    display_cols = ["Id", "Name", "Industry", "AnnualRevenue", "Predicted_Churn_Prob", "Predicted_Risk_Level", "Top_Churn_Driver"]
    st.dataframe(filtered_df[display_cols].rename(columns={"AnnualRevenue": "Annual Revenue ($)", "Predicted_Churn_Prob": "Churn Score", "Predicted_Risk_Level": "Risk Level", "Top_Churn_Driver": "Top Driver"}), use_container_width=True, hide_index=True)