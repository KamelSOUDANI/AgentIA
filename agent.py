import numpy as np

def explain_transaction(row):
    explanations = []

    if row["amount"] > 300:
        explanations.append("Montant élevé inhabituel.")

    if row["country_risk"] > 0.5:
        explanations.append("Pays de transaction à risque.")

    if row["device_risk"] > 0.6:
        explanations.append("Device suspect.")

    if row["new_device"] == 1:
        explanations.append("Nouveau device jamais vu.")

    if row["geo_anomaly"] == 1:
        explanations.append("Distance géographique impossible.")

    if row["velocity_24h"] > 6:
        explanations.append("Trop de transactions récentes (velocity).")

    if row["fraud_history"] == 1:
        explanations.append("Client avec historique de fraude.")

    if not explanations:
        explanations.append("Aucun signal fort détecté.")

    return explanations


def decision_from_score(score):
    if score > 0.8:
        return "BLOCK"
    elif score > 0.5:
        return "REVIEW"
    else:
        return "ALLOW"
