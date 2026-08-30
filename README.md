# ⚡ Customer 360 Risk Pulse™ | Salesforce AI Churn Intelligence

An end-to-end Machine Learning and Reverse-ETL intelligence platform built to predict and prevent customer churn in B2B SaaS. 

This project extracts live CRM data from **Salesforce**, combines billing history with support ticket friction, predicts churn risk using **XGBoost**, calculates the exact root cause using **TreeSHAP (Explainable AI)**, and writes the predictions back into custom Salesforce fields.

---

## 📌 Table of Contents
- [What Problem Does This Solve?](#-what-problem-does-this-solve)
- [How It Works (System Architecture)](#-how-it-works-system-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Step-by-Step Setup & Installation](#-step-by-step-setup--installation)
- [Machine Learning Performance & Metrics](#-machine-learning-performance--metrics)
- [Explainable AI (TreeSHAP)](#-explainable-ai-treeshap)
- [Screenshots & Visuals](#-screenshots--visuals)

---

## 💡 What Problem Does This Solve?

In subscription software (B2B SaaS), customer churn rarely happens without warning. Early signs appear weeks before cancellation:
1. **Support ticket spikes:** A client suddenly submits multiple cases.
2. **Escalations:** Tickets are marked *Critical* or *Escalated*.
3. **Slow resolutions:** Mean time to resolve tickets increases.
4. **Contract friction:** Short-tenure accounts on expensive month-to-month contracts.

Customer Success Managers (CSMs) cannot manually audit hundreds of accounts daily. **Customer 360 Risk Pulse™** automates this by continuously calculating churn probability and displaying the dollar value of **ARR (Annual Recurring Revenue) at risk** directly inside Salesforce.

---

## 🏗️ How It Works (System Architecture)

```text
[ Salesforce CRM ]
   ├── Account Object (Revenue, Industry, Contract Type, Tenure)
   └── Case Object (Priority, Status, Escalation Flag, Resolution Time)
           │
           ▼
[ Feature Engineering (src/etl.py) ]
   └── Calculates: Escalation Rate, Ticket Velocity, Monthly Spend
           │
           ▼
[ ML Model & Explainability (src/train.py) ]
   ├── XGBoost Classifier -> Churn Probability (0.0 to 1.0) & Risk Tier (Low/Med/High)
   └── TreeSHAP Explainer -> Primary Root-Cause Driver
           │
           ├──────────────────────────────┐
           ▼                              ▼
[ Reverse-ETL Sync (src/sync.py) ]   [ Streamlit Dashboard (app.py) ]
   Writes back to Salesforce fields:     ├── Portfolio Risk & ARR Exposure
   • Churn_Risk_Score__c                 ├── Account Deep-Dive Gauge & Metrics
   • Risk_Level__c                       ├── Live Support Case Telemetry
   • Top_Churn_Driver__c                 └── Interactive SOQL Query Studio
