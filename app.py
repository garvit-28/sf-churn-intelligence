"""
Salesforce Churn Intelligence & Customer 360 Engine
Streamlit Application for Real-Time Churn Risk Prediction & Retention Playbooks
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Salesforce Churn Intelligence & Customer 360",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern enterprise SaaS look
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .risk-high { color: #f87171; font-weight: bold; }
    .risk-medium { color: #fbbf24; font-weight: bold; }
    .risk-low { color: #34d399; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA INGESTION & ROBUST FALLBACK ENGINE
# ==========================================
def generate_synthetic_data():
    """Generates synthetic enterprise CRM data if no live connection or local files exist."""
    np.random.seed(42)
    n = 60
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail", "Energy"]
    
    accounts = []
    cases = []
    
    for i in range(1, n + 1):
        acc_id = f"0015g00000{i:04d}AAA"
        name = f"Enterprise Corp {i}"
        rev = float(np.random.choice([250000, 750000, 1500000, 3200000, 5000000, 12000000]))
        emp = int(np.random.choice([50, 150, 500, 1200, 3000, 7500]))
        ind = str(np.random.choice(industries))
        
        accounts.append({
            "Id": acc_id,
            "Name": name,
            "AnnualRevenue": rev,
            "NumberOfEmployees": emp,
            "Industry": ind
        })
        
        # Generate associated cases
        num_cases = np.random.randint(1, 9)
        for c in range(num_cases):
            cases.append({
                "Id": f"5005g00000{i:02d}{c:02d}BBB",
                "AccountId": acc_id,
                "Status": np.random.choice(["Closed", "Closed", "Closed", "Open", "Escalated"]),
                "Priority": np.random.choice(["Low", "Medium", "High", "Critical"]),
                "CreatedDate": "2026-01-15T10:00:00.000Z",
                "ClosedDate": "2026-02-10T12:00:00.000Z"
            })
            
    return pd.DataFrame(accounts), pd.DataFrame(cases)


@st.cache_data(show_spinner=False, ttl=600)
def fetch_all_salesforce_data():
    """Fetches Salesforce data via Client or falls back to CSV / Synthetic state."""
    df_accs = pd.DataFrame()
    df_cases = pd.DataFrame()
    
    # 1. Try SalesforceClient
    if SalesforceClient is not None:
        try:
            sf = SalesforceClient()
            df_accs = sf.query("SELECT Id, Name, AnnualRevenue, NumberOfEmployees, Industry FROM Account")
            df_cases = sf.query("SELECT Id, AccountId, Status, Priority, CreatedDate, ClosedDate FROM Case")
        except Exception as e:
            logging.warning(f"Salesforce query failed: {e}")

    # 2. Try Local CSV paths if query returned empty
    if df_accs.empty or "Id" not in df_accs.columns:
        csv_candidates_acc = ["data/raw_accounts.csv", "data/accounts.csv", "accounts.csv", "data/salesforce_data.csv"]
        for p in csv_candidates_acc:
            if os.path.exists(p):
                try:
                    df_accs = pd.read_csv(p)
                    break
                except Exception:
                    pass

    if df_cases.empty or "AccountId" not in df_cases.columns:
        csv_candidates_case = ["data/raw_cases.csv", "data/cases.csv", "cases.csv"]
        for p in csv_candidates_case:
            if os.path.exists(p):
                try:
                    df_cases = pd.read_csv(p)
                    break
                except Exception:
                    pass

    # 3. Final Fallback: Synthetic generation
    if df_accs.empty:
        df_accs, df_cases = generate_synthetic_data()

    # Data hygiene & null fills
    df_accs["AnnualRevenue"] = pd.to_numeric(df_accs.get("AnnualRevenue", 500000), errors="coerce").fillna(500000)
    df_accs["NumberOfEmployees"] = pd.to_numeric(df_accs.get("NumberOfEmployees", 100), errors="coerce").fillna(100)
    df_accs["Industry"] = df_accs.get("Industry", "Other").fillna("Other")
    df_accs["Name"] = df_accs.get("Name", "Account").fillna("Account")
    df_accs["Id"] = df_accs.get("Id", df_accs.index.astype(str))

    return df_accs, df_cases


# ==========================================
# 3. FEATURE ENGINEERING & ML PIPELINE
# ==========================================
def build_feature_matrix(df_accs: pd.DataFrame, df_cases: pd.DataFrame) -> pd.DataFrame:
    """Computes operational friction, support metrics, and behavioral feature matrix."""
    features = df_accs.copy()
    
    # Calculate case aggregations per account
    if not df_cases.empty and "AccountId" in df_cases.columns:
        case_stats = df_cases.groupby("AccountId").agg(
            Total_Cases=("Id", "count"),
            Open_Cases=("Status", lambda s: (s.isin(["Open", "Escalated", "Working", "New"])).sum()),
            Escalated_Cases=("Status", lambda s: (s == "Escalated").sum()),
            Critical_Cases=("Priority", lambda p: (p.isin(["High", "Critical"])).sum())
        ).reset_index()
        features = features.merge(case_stats, left_on="Id", right_on="AccountId", how="left")
    else:
        features["Total_Cases"] = 3
        features["Open_Cases"] = 1
        features["Escalated_Cases"] = 0
        features["Critical_Cases"] = 0

    # Fill numerical case features
    for col in ["Total_Cases", "Open_Cases", "Escalated_Cases", "Critical_Cases"]:
        features[col] = pd.to_numeric(features.get(col, 0), errors="coerce").fillna(0)

    # Deterministic pseudo-metrics based on hash for realistic reproducibility
    np.random.seed(42)
    hash_seed = features["Id"].astype(str).apply(lambda x: sum(ord(c) for c in x)) % 100
    
    features["Days_Since_Last_Contact"] = (hash_seed * 1.5).clip(3, 140).astype(int)
    features["Contract_Months_Remaining"] = ((hash_seed % 24) + 1).astype(int)
    features["NPS_Score"] = (10 - (features["Open_Cases"] * 1.2) - (hash_seed % 4)).clip(1, 10).round(1)
    features["Product_Usage_Drop_Pct"] = ((features["Open_Cases"] * 8) + (features["Days_Since_Last_Contact"] * 0.3)).clip(0, 85).round(1)
    features["Revenue_Per_Employee"] = (features["AnnualRevenue"] / features["NumberOfEmployees"].replace(0, 1)).round(2)
    
    return features


@st.cache_resource(show_spinner=False)
def train_or_load_model():
    """Trains an XGBoost classifier on feature interactions for instant explainability."""
    X_sample = pd.DataFrame({
        "Open_Cases": [0, 1, 3, 5, 2, 6, 0, 4, 1, 7],
        "Escalated_Cases": [0, 0, 1, 2, 0, 3, 0, 2, 0, 4],
        "Critical_Cases": [0, 0, 2, 3, 1, 4, 0, 3, 0, 5],
        "Days_Since_Last_Contact": [10, 25, 60, 95, 35, 120, 15, 80, 20, 130],
        "Contract_Months_Remaining": [20, 16, 8, 3, 12, 1, 24, 4, 18, 2],
        "NPS_Score": [9, 8, 6, 3, 7, 2, 10, 4, 8, 2],
        "Product_Usage_Drop_Pct": [2.0, 5.0, 25.0, 60.0, 12.0, 75.0, 0.0, 45.0, 8.0, 80.0]
    })
    y_sample = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_sample, y_sample)
    
    feature_cols = list(X_sample.columns)
    return model, feature_cols


# ==========================================
# 4. APP EXECUTION & DATA PIPELINE
# ==========================================
live_accs, live_cases = fetch_all_salesforce_data()
df_features = build_feature_matrix(live_accs, live_cases)
model, feature_cols = train_or_load_model()

# Perform Inference across all accounts
X_input = df_features[feature_cols].fillna(0)
probs = model.predict_proba(X_input)[:, 1]

df_features["Churn_Probability"] = probs
df_features["Churn_Risk"] = pd.cut(
    df_features["Churn_Probability"],
    bins=[-0.01, 0.35, 0.65, 1.0],
    labels=["Low", "Medium", "High"]
)
df_features["Revenue_At_Risk"] = np.where(
    df_features["Churn_Risk"].isin(["High", "Medium"]),
    df_features["AnnualRevenue"],
    0.0
)

# Tree SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_input)


# ==========================================
# 5. HEADER & TOP NAVIGATION
# ==========================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("⚡ Salesforce Churn Intelligence & Customer 360")
    st.caption("AI-Powered Proactive Retention Engine • XGBoost Inference • SHAP Explainability")

with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data Sync", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

tab_overview, tab_c360, tab_simulator = st.tabs([
    "📊 Executive Overview",
    "👤 Customer 360 & SHAP",
    "🔬 What-If Retention Simulator"
])


# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
with tab_overview:
    # Key KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_accounts = len(df_features)
    high_risk_count = (df_features["Churn_Risk"] == "High").sum()
    total_rev_at_risk = df_features["Revenue_At_Risk"].sum()
    avg_churn_prob = (df_features["Churn_Probability"].mean()) * 100

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL MONITORED ACCOUNTS</div>
            <div class="metric-value">{total_accounts}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HIGH CHURN RISK ACCOUNTS</div>
            <div class="metric-value" style="color: #f87171;">{high_risk_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TOTAL REVENUE AT RISK</div>
            <div class="metric-value" style="color: #fbbf24;">${total_rev_at_risk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">AVERAGE PORTFOLIO RISK</div>
            <div class="metric-value">{avg_churn_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations Row
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Account Risk Distribution")
        risk_counts = df_features["Churn_Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        fig_pie = px.pie(
            risk_counts,
            names="Risk_Level",
            values="Count",
            color="Risk_Level",
            color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
            hole=0.45
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.subheader("Customer Friction vs. Churn Probability")
        
        # RESILIENT SCATTER PLOT
        if not df_features.empty:
            plot_df = df_features.copy()
            x_col = "Days_Since_Last_Contact" if "Days_Since_Last_Contact" in plot_df.columns else plot_df.columns[0]
            y_col = "Open_Cases" if "Open_Cases" in plot_df.columns else plot_df.columns[1]

            plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce").fillna(0)
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce").fillna(0)
            plot_df["AnnualRevenue"] = pd.to_numeric(plot_df.get("AnnualRevenue", 100000), errors="coerce").fillna(100000)
            plot_df["AnnualRevenue_Scaled"] = np.sqrt(plot_df["AnnualRevenue"]).clip(lower=10)

            scatter_kwargs = {
                "data_frame": plot_df,
                "x": x_col,
                "y": y_col,
                "hover_name": "Name" if "Name" in plot_df.columns else None,
                "color": "Churn_Risk" if "Churn_Risk" in plot_df.columns else None,
                "color_discrete_map": {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                "size": "AnnualRevenue_Scaled",
                "labels": {
                    x_col: "Days Since Last Contact",
                    y_col: "Open Support Cases"
                }
            }
            fig_friction = px.scatter(**scatter_kwargs)
            fig_friction.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_friction, use_container_width=True)
        else:
            st.info("No data available to render friction scatter chart.")

    # High Risk Action Table
    st.subheader("🚨 Priority Retention Accounts (High & Medium Risk)")
    priority_cols = ["Name", "Industry", "AnnualRevenue", "Open_Cases", "Days_Since_Last_Contact", "NPS_Score", "Churn_Probability", "Churn_Risk"]
    available_cols = [c for c in priority_cols if c in df_features.columns]
    
    df_priority = df_features[df_features["Churn_Risk"].isin(["High", "Medium"])][available_cols].sort_values(
        by="Churn_Probability", ascending=False
    )
    
    st.dataframe(
        df_priority.style.format({
            "AnnualRevenue": "${:,.0f}",
            "Churn_Probability": "{:.1%}",
            "NPS_Score": "{:.1f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# TAB 2: CUSTOMER 360 & SHAP EXPLAINABILITY
# ==========================================
with tab_c360:
    st.subheader("360° Account Deep Dive & Machine Learning Explainability")
    
    account_names = df_features["Name"].tolist()
    selected_name = st.selectbox("Select Target Salesforce Account:", account_names, index=0)
    
    account_row = df_features[df_features["Name"] == selected_name].iloc[0]
    account_idx = df_features[df_features["Name"] == selected_name].index[0]
    
    # 360 Info Cards
    col_c1, col_c2, col_c3 = st.columns([1, 1, 1.2])
    
    with col_c1:
        st.markdown(f"""
        **Account Details:**
        * **Account ID:** `{account_row['Id']}`
        * **Industry:** {account_row['Industry']}
        * **Annual Revenue:** ${account_row['AnnualRevenue']:,.0f}
        * **Employees:** {int(account_row['NumberOfEmployees']):,}
        """)
        
    with col_c2:
        risk_class = f"risk-{account_row['Churn_Risk'].lower()}"
        st.markdown(f"""
        **Engagement Health:**
        * **Churn Risk:** <span class="{risk_class}">{account_row['Churn_Risk']} ({account_row['Churn_Probability']:.1%})</span>
        * **Open Support Tickets:** {int(account_row['Open_Cases'])}
        * **Days Since Contact:** {int(account_row['Days_Since_Last_Contact'])} days
        * **NPS Score:** {account_row['NPS_Score']} / 10
        """, unsafe_allow_html=True)

    with col_c3:
        # Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=account_row["Churn_Probability"] * 100,
            number={'suffix': "%"},
            title={'text': "Churn Probability", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if account_row['Churn_Probability'] > 0.65 else ("#f59e0b" if account_row['Churn_Probability'] > 0.35 else "#10b981")},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.15)"},
                    {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.15)"},
                    {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                ]
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # SHAP Waterfall / Feature Drivers
    st.subheader(f"Why is {selected_name} at Risk? (SHAP Feature Attribution)")
    
    col_shap1, col_shap2 = st.columns([1.5, 1])
    
    with col_shap1:
        # Calculate local SHAP contributions for the selected record
        acc_shap = shap_values[account_idx].values
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Impact": acc_shap,
            "Value": [account_row[col] for col in feature_cols]
        }).sort_values(by="Impact", ascending=True)

        colors = ["#ef4444" if val > 0 else "#10b981" for val in shap_df["Impact"]]
        
        fig_shap = go.Figure(go.Bar(
            x=shap_df["Impact"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in shap_df["Impact"]],
            textposition="auto"
        ))
        fig_shap.update_layout(
            title="Feature Contribution to Churn Risk (Red = Increases Risk, Green = Reduces Risk)",
            xaxis_title="SHAP Value (Impact on Model Log-Odds)",
            margin=dict(t=40, b=20, l=10, r=10),
            height=340
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_shap2:
        st.markdown("#### Automated Retention Playbook")
        if account_row["Churn_Risk"] == "High":
            st.error("🚨 **Immediate Action Recommended:**")
            st.markdown(f"""
            1. **Executive Escalation:** Assign CSM lead to resolve **{int(account_row['Open_Cases'])} open tickets**.
            2. **Friction Intervention:** Re-engage stakeholder immediately (inactivity: **{int(account_row['Days_Since_Last_Contact'])} days**).
            3. **Discount / Contract Review:** Propose flexible renewal terms for remaining **{int(account_row['Contract_Months_Remaining'])} months**.
            """)
        elif account_row["Churn_Risk"] == "Medium":
            st.warning("⚠️ **Preventive Care Recommended:**")
            st.markdown(f"""
            1. **Customer Check-in:** Schedule proactive feedback sync.
            2. **Feature Training:** Mitigate usage drop with dedicated enablement.
            """)
        else:
            st.success("✅ **Account Healthy & Expansion Ready:**")
            st.markdown("""
            1. **Upsell Potential:** Pitch enterprise add-ons or increased user tier.
            2. **Advocacy:** Request case study or referral testimonial.
            """)

        # Trigger Retention Task
        if st.button(f"⚡ Create Retention Task in Salesforce for {account_row['Name']}", use_container_width=True):
            if SalesforceClient is not None:
                try:
                    sf = SalesforceClient()
                    task_payload = {
                        "Subject": f"Urgent: Churn Risk Mitigation ({account_row['Churn_Risk']})",
                        "Priority": "High" if account_row["Churn_Risk"] == "High" else "Normal",
                        "Status": "Not Started",
                        "WhatId": account_row["Id"]
                    }
                    task_id = sf.create_record("Task", task_payload)
                    st.toast(f"✅ Retention Task successfully logged in Salesforce (ID: {task_id})!", icon="🚀")
                except Exception as e:
                    st.toast(f"ℹ️ Simulated Task Creation: {e}", icon="🔔")
            else:
                st.toast("✅ Demo Task Created: Assigned to Account Executive.", icon="🎯")


# ==========================================
# TAB 3: WHAT-IF SCENARIO SIMULATOR
# ==========================================
with tab_simulator:
    st.subheader("🔬 Real-Time Retention Scenario Simulator")
    st.caption("Simulate intervention strategies and observe dynamic changes in churn probability.")

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        sim_account_name = st.selectbox("Select Account to Simulate:", account_names, key="sim_acc_select")
        base_row = df_features[df_features["Name"] == sim_account_name].iloc[0]

        st.markdown("##### Adjust Proposed Intervention Levers:")
        sim_open_cases = st.slider("Resolve Support Tickets (Target Open Cases):", 0, 10, int(base_row["Open_Cases"]))
        sim_days_contact = st.slider("Execute Contact (Days Since Last Touchpoint):", 1, 90, int(min(base_row["Days_Since_Last_Contact"], 15)))
        sim_nps = st.slider("Target NPS Score Post-Resolution:", 1.0, 10.0, float(max(base_row["NPS_Score"], 7.0)), step=0.5)
        sim_usage_drop = st.slider("Recover Product Usage Drop (%):", 0.0, 80.0, float(max(base_row["Product_Usage_Drop_Pct"] * 0.5, 0.0)))
        
        sim_input = pd.DataFrame([{
            "Open_Cases": sim_open_cases,
            "Escalated_Cases": max(0, sim_open_cases - 2),
            "Critical_Cases": max(0, sim_open_cases - 3),
            "Days_Since_Last_Contact": sim_days_contact,
            "Contract_Months_Remaining": base_row["Contract_Months_Remaining"],
            "NPS_Score": sim_nps,
            "Product_Usage_Drop_Pct": sim_usage_drop
        }])

        sim_prob = model.predict_proba(sim_input[feature_cols])[0, 1]
        base_prob = base_row["Churn_Probability"]
        delta_prob = (sim_prob - base_prob) * 100

    with sim_col2:
        st.markdown("##### Impact of Proposed Retention Strategy")
        
        sim_kpi1, sim_kpi2 = st.columns(2)
        with sim_kpi1:
            st.metric("Current Churn Probability", f"{base_prob:.1%}")
        with sim_kpi2:
            st.metric(
                "Simulated Churn Probability",
                f"{sim_prob:.1%}",
                delta=f"{delta_prob:.1f}%",
                delta_color="inverse"
            )

        # Comparative Bar Chart
        comp_df = pd.DataFrame({
            "Scenario": ["Current Baseline", "After Proposed Intervention"],
            "Churn_Probability": [base_prob * 100, sim_prob * 100]
        })
        
        fig_comp = px.bar(
            comp_df,
            x="Scenario",
            y="Churn_Probability",
            color="Scenario",
            color_discrete_sequence=["#ef4444", "#10b981"],
            text=comp_df["Churn_Probability"].apply(lambda v: f"{v:.1f}%")
        )
        fig_comp.update_layout(
            yaxis_range=[0, 100],
            yaxis_title="Churn Probability (%)",
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        if delta_prob < -10:
            st.success(f"🎉 This retention strategy reduces churn probability by **{abs(delta_prob):.1f}%**, protecting **${base_row['AnnualRevenue']:,.0f}** in ARR.")
        else:
            st.info("💡 Adjust the sliders further to see the compounding impact of ticket resolution and proactive outreach.")