import streamlit as st
import pandas as pd
import joblib
from agent import explain_transaction, decision_from_score

model = joblib.load("fraud_model.pkl")

st.title("Agent IA Local — Détection de Fraude Bancaire")

st.sidebar.header("Transaction")

amount = st.sidebar.number_input("Montant", min_value=0.0, value=100.0)
country_risk = st.sidebar.slider("Risque pays (0-1)", 0.0, 1.0, 0.2)
device_risk = st.sidebar.slider("Risque device (0-1)", 0.0, 1.0, 0.1)
new_device = st.sidebar.selectbox("Nouveau device ?", [0,1])
geo_anomaly = st.sidebar.selectbox("Anomalie géographique ?", [0,1])
velocity_24h = st.sidebar.number_input("Transactions 24h", min_value=0, value=2)
hour = st.sidebar.slider("Heure", 0, 23, 14)

st.sidebar.header("Profil client")

age = st.sidebar.slider("Âge", 18, 90, 40)
account_age_years = st.sidebar.slider("Ancienneté compte (années)", 0.0, 30.0, 5.0)
income = st.sidebar.number_input("Revenu annuel", min_value=5000, value=30000)
avg_transaction_amount = st.sidebar.number_input("Montant moyen client", min_value=0.0, value=60.0)
preferred_device_risk = st.sidebar.slider("Risque device habituel", 0.0, 1.0, 0.2)
fraud_history = st.sidebar.selectbox("Historique fraude ?", [0,1])

if st.button("Analyser"):
    row = pd.DataFrame([{
        "amount": amount,
        "country_risk": country_risk,
        "device_risk": device_risk,
        "new_device": new_device,
        "geo_anomaly": geo_anomaly,
        "velocity_24h": velocity_24h,
        "hour": hour,
        "age": age,
        "account_age_years": account_age_years,
        "income": income,
        "avg_transaction_amount": avg_transaction_amount,
        "preferred_device_risk": preferred_device_risk,
        "fraud_history": fraud_history
    }])

    score = model.predict_proba(row)[0][1]
    decision = decision_from_score(score)
    explanations = explain_transaction(row.iloc[0])

    st.subheader("Résultat")
    st.write(f"Probabilité de fraude : **{score:.2f}**")
    st.write(f"Décision : **{decision}**")

    st.subheader("Explication")
    for e in explanations:
        st.write("- " + e)
