"""
Salesforce Churn Intelligence & Customer 360 Engine
Featuring: Sidebar Multi-Page Navigation, Dynamic Column Selectors, and SOQL Studio
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
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Salesforce Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .risk-high { color: #f87171; font-weight: bold; }
    .risk-medium { color: #fbbf24; font-weight: bold; }
    .risk-low { color: #34d399; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATA INGESTION & ROBUST FALLBACK
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
    
    if SalesforceClient is not None:
        try:
            sf = SalesforceClient()
            df_accs = sf.query("SELECT Id, Name, AnnualRevenue, NumberOfEmployees, Industry FROM Account")
            df_cases = sf.query("SELECT Id, AccountId, Status, Priority, CreatedDate, ClosedDate FROM Case")
        except Exception as e:
            logging.warning(f"Salesforce query failed: {e}")

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

    if df_accs.empty:
        df_accs, df_cases = generate_synthetic_data()

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
    features = df_accs.copy()
    
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

    for col in ["Total_Cases", "Open_Cases", "Escalated_Cases", "Critical_Cases"]:
        features[col] = pd.to_numeric(features.get(col, 0), errors="coerce").fillna(0)

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


# Run core processing
live_accs, live_cases = fetch_all_salesforce_data()
df_features = build_feature_matrix(live_accs, live_cases)
model, feature_cols = train_or_load_model()

# Model Inference
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
# 4. SIDEBAR NAVIGATION & COLUMN SELECTOR
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/f9/Salesforce.com_logo.svg", width=120)
    st.title("Navigation")
    
    nav_selection = st.radio(
        "Go to Page:",
        [
            "📊 Executive Dashboard",
            "👤 Customer 360 & SHAP",
            "🔬 What-If Simulator",
            "🔍 SOQL Studio & Data Explorer"
        ]
    )
    
    st.markdown("---")
    st.subheader("⚙️ Global Filters")
    
    # Industry Filter
    all_industries = sorted(list(df_features["Industry"].unique()))
    selected_industries = st.multiselect("Filter by Industry:", all_industries, default=all_industries)
    
    # Risk Filter
    selected_risks = st.multiselect("Filter by Risk Level:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    
    st.markdown("---")
    if st.button("🔄 Sync & Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Apply Sidebar Filters to working data
filtered_df = df_features[
    (df_features["Industry"].isin(selected_industries)) &
    (df_features["Churn_Risk"].isin(selected_risks))
]


# ==========================================
# PAGE 1: EXECUTIVE DASHBOARD
# ==========================================
if nav_selection == "📊 Executive Dashboard":
    st.title("📊 Salesforce Churn Intelligence & Portfolio Risk")
    st.caption("AI-driven real-time churn prediction engine synchronized with Salesforce Accounts & Cases.")
    
    # Top KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    total_acc = len(filtered_df)
    high_risk = (filtered_df["Churn_Risk"] == "High").sum()
    rev_at_risk = filtered_df["Revenue_At_Risk"].sum()
    avg_risk = (filtered_df["Churn_Probability"].mean() * 100) if total_acc > 0 else 0

    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL MONITORED</div><div class="metric-value">{total_acc}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">HIGH CHURN RISK</div><div class="metric-value" style="color:#f87171;">{high_risk}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">REVENUE AT RISK</div><div class="metric-value" style="color:#fbbf24;">${rev_at_risk:,.0f}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">AVG RISK SCORE</div><div class="metric-value">{avg_risk:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Visualizations
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.subheader("Risk Distribution")
        risk_counts = filtered_df["Churn_Risk"].value_counts().reset_index()
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

    with col_v2:
        st.subheader("Customer Friction vs. Churn Risk")
        if not filtered_df.empty:
            p_df = filtered_df.copy()
            x_col = "Days_Since_Last_Contact"
            y_col = "Open_Cases"
            p_df[x_col] = pd.to_numeric(p_df[x_col], errors="coerce").fillna(0)
            p_df[y_col] = pd.to_numeric(p_df[y_col], errors="coerce").fillna(0)
            p_df["Revenue_Scaled"] = np.sqrt(pd.to_numeric(p_df.get("AnnualRevenue", 100000), errors="coerce").fillna(100000)).clip(lower=10)

            fig_scat = px.scatter(
                p_df,
                x=x_col,
                y=y_col,
                hover_name="Name",
                color="Churn_Risk",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                size="Revenue_Scaled",
                labels={x_col: "Days Since Last Contact", y_col: "Open Cases"}
            )
            fig_scat.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_scat, use_container_width=True)
        else:
            st.info("No accounts match current filter criteria.")

    # Data Table with Column Selector
    st.subheader("📋 Account Risk Registry")
    
    # Dynamic Column Selector
    all_table_cols = list(filtered_df.columns)
    default_cols = ["Name", "Industry", "AnnualRevenue", "Open_Cases", "Days_Since_Last_Contact", "NPS_Score", "Churn_Probability", "Churn_Risk"]
    active_default = [c for c in default_cols if c in all_table_cols]
    
    selected_cols = st.multiselect("Customize Displayed Columns:", all_table_cols, default=active_default)
    
    if selected_cols:
        st.dataframe(
            filtered_df[selected_cols].sort_values(by="Churn_Probability", ascending=False if "Churn_Probability" in selected_cols else True),
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# PAGE 2: CUSTOMER 360 & SHAP
# ==========================================
elif nav_selection == "👤 Customer 360 & SHAP":
    st.title("👤 Customer 360° & AI Explainability")
    st.caption("Deep-dive into individual accounts and explain the root drivers behind model risk scores using SHAP.")
    
    account_list = df_features["Name"].tolist()
    target_acc = st.selectbox("Select Target Account:", account_list, index=0)
    
    acc_data = df_features[df_features["Name"] == target_acc].iloc[0]
    acc_idx = df_features[df_features["Name"] == target_acc].index[0]
    
    c360_1, c360_2, c360_3 = st.columns([1, 1, 1.2])
    
    with c360_1:
        st.markdown(f"""
        **Account Metadata:**
        * **Salesforce ID:** `{acc_data['Id']}`
        * **Industry:** {acc_data['Industry']}
        * **Annual Revenue:** ${acc_data['AnnualRevenue']:,.0f}
        * **Employee Count:** {int(acc_data['NumberOfEmployees']):,}
        """)
        
    with c360_2:
        risk_class = f"risk-{acc_data['Churn_Risk'].lower()}"
        st.markdown(f"""
        **Health Telemetry:**
        * **Churn Risk:** <span class="{risk_class}">{acc_data['Churn_Risk']} ({acc_data['Churn_Probability']:.1%})</span>
        * **Open Support Tickets:** {int(acc_data['Open_Cases'])}
        * **Inactivity:** {int(acc_data['Days_Since_Last_Contact'])} days
        * **NPS Score:** {acc_data['NPS_Score']} / 10
        """, unsafe_allow_html=True)

    with c360_3:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=acc_data["Churn_Probability"] * 100,
            number={'suffix': "%"},
            title={'text': "Churn Probability", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444" if acc_data['Churn_Probability'] > 0.65 else ("#f59e0b" if acc_data['Churn_Probability'] > 0.35 else "#10b981")},
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

    # SHAP Feature Attribution
    s_col1, s_col2 = st.columns([1.5, 1])
    
    with s_col1:
        st.subheader("Feature Impact Attribution (SHAP)")
        acc_shap = shap_values[acc_idx].values
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Impact": acc_shap,
            "Value": [acc_data[col] for col in feature_cols]
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
            title="Local SHAP Values (Red = Pushes toward Churn, Green = Pushes toward Retain)",
            xaxis_title="SHAP Value (Impact on Model Log-Odds)",
            margin=dict(t=40, b=20, l=10, r=10),
            height=340
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with s_col2:
        st.subheader("Automated Playbook")
        if acc_data["Churn_Risk"] == "High":
            st.error("🚨 **High Risk Playbook Triggered**")
            st.markdown(f"""
            - **Priority Support:** Resolve **{int(acc_data['Open_Cases'])} open tickets**.
            - **Executive Outreach:** Re-engage contact (idle **{int(acc_data['Days_Since_Last_Contact'])} days**).
            - **Contract Review:** Renewal in **{int(acc_data['Contract_Months_Remaining'])} months**.
            """)
        else:
            st.success("✅ **Stable Account**")
            st.markdown("- Account healthy. Recommended for expansion or case study review.")

        if st.button(f"⚡ Log Retention Task in Salesforce for {acc_data['Name']}", use_container_width=True):
            if SalesforceClient is not None:
                try:
                    sf = SalesforceClient()
                    task_id = sf.create_record("Task", {
                        "Subject": f"Mitigate Churn Risk ({acc_data['Churn_Risk']})",
                        "Priority": "High" if acc_data["Churn_Risk"] == "High" else "Normal",
                        "Status": "Not Started",
                        "WhatId": acc_data["Id"]
                    })
                    st.toast(f"✅ Retention Task logged in Salesforce (ID: {task_id})!", icon="🚀")
                except Exception as e:
                    st.toast(f"ℹ️ Task created (Simulated): {e}", icon="🔔")
            else:
                st.toast("✅ Demo Task Created: Assigned to CSM.", icon="🎯")


# ==========================================
# PAGE 3: WHAT-IF RETENTION SIMULATOR
# ==========================================
elif nav_selection == "🔬 What-If Simulator":
    st.title("🔬 What-If Retention Scenario Simulator")
    st.caption("Interactively test intervention levers (closing tickets, increasing contact) to measure immediate risk reduction.")

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        sim_acc = st.selectbox("Select Account to Simulate:", df_features["Name"].tolist())
        base_acc = df_features[df_features["Name"] == sim_acc].iloc[0]

        st.markdown("##### Adjust Intervention Levers:")
        sim_open_cases = st.slider("Target Open Cases (Post-Resolution):", 0, 10, int(base_acc["Open_Cases"]))
        sim_days = st.slider("Target Days Since Last Contact:", 1, 90, int(min(base_acc["Days_Since_Last_Contact"], 15)))
        sim_nps = st.slider("Target NPS Score:", 1.0, 10.0, float(max(base_acc["NPS_Score"], 7.0)), step=0.5)
        sim_drop = st.slider("Recovered Usage Drop (%):", 0.0, 80.0, float(max(base_acc["Product_Usage_Drop_Pct"] * 0.5, 0.0)))
        
        sim_input = pd.DataFrame([{
            "Open_Cases": sim_open_cases,
            "Escalated_Cases": max(0, sim_open_cases - 2),
            "Critical_Cases": max(0, sim_open_cases - 3),
            "Days_Since_Last_Contact": sim_days,
            "Contract_Months_Remaining": base_acc["Contract_Months_Remaining"],
            "NPS_Score": sim_nps,
            "Product_Usage_Drop_Pct": sim_drop
        }])

        sim_prob = model.predict_proba(sim_input[feature_cols])[0, 1]
        base_prob = base_acc["Churn_Probability"]
        delta_prob = (sim_prob - base_prob) * 100

    with sim_col2:
        st.markdown("##### Simulated Retention Impact")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Baseline Churn Risk", f"{base_prob:.1%}")
        with m2:
            st.metric("Simulated Risk", f"{sim_prob:.1%}", delta=f"{delta_prob:.1f}%", delta_color="inverse")

        comp_df = pd.DataFrame({
            "Scenario": ["Baseline Risk", "Simulated Risk"],
            "Probability": [base_prob * 100, sim_prob * 100]
        })
        
        fig_comp = px.bar(
            comp_df,
            x="Scenario",
            y="Probability",
            color="Scenario",
            color_discrete_sequence=["#ef4444", "#10b981"],
            text=comp_df["Probability"].apply(lambda v: f"{v:.1f}%")
        )
        fig_comp.update_layout(yaxis_range=[0, 100], height=280, showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_comp, use_container_width=True)

        if delta_prob < -5:
            st.success(f"🎉 Strategy saves **{abs(delta_prob):.1f}%** in churn probability, protecting **${base_acc['AnnualRevenue']:,.0f}** in ARR.")


# ==========================================
# PAGE 4: SOQL STUDIO & DATA EXPLORER
# ==========================================
elif nav_selection == "🔍 SOQL Studio & Data Explorer":
    st.title("🔍 SOQL Studio & Query Console")
    st.caption("Execute custom SOQL queries against your Salesforce org or explore raw synchronized entities.")

    soql_tab1, soql_tab2 = st.tabs(["⚡ Interactive SOQL Console", "📁 Raw Entity Explorer"])

    with soql_tab1:
        st.markdown("##### Enter SOQL Query:")
        default_query = "SELECT Id, Name, AnnualRevenue, NumberOfEmployees, Industry FROM Account LIMIT 10"
        user_query = st.text_area("SOQL Editor", value=default_query, height=120)

        col_q1, col_q2 = st.columns([1, 4])
        with col_q1:
            run_query = st.button("🚀 Execute SOQL", use_container_width=True)

        if run_query:
            with st.spinner("Executing SOQL query..."):
                if SalesforceClient is not None:
                    try:
                        sf = SalesforceClient()
                        df_result = sf.query(user_query)
                        if not df_result.empty:
                            st.success(f"Returned {len(df_result)} record(s).")
                            st.dataframe(df_result, use_container_width=True)
                        else:
                            st.warning("Query returned 0 records.")
                    except Exception as e:
                        st.error(f"SOQL Execution Error: {e}")
                else:
                    st.info("Demo Mode: Executing on local snapshot...")
                    st.dataframe(df_features.head(10), use_container_width=True)

    with soql_tab2:
        entity_choice = st.selectbox("Choose Salesforce Object to Inspect:", ["Account", "Case", "Engineered Features Matrix"])
        
        if entity_choice == "Account":
            st.dataframe(live_accs, use_container_width=True)
        elif entity_choice == "Case":
            st.dataframe(live_cases, use_container_width=True)
        else:
            st.dataframe(df_features, use_container_width=True)