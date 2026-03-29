import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("dataset_merged.csv")

# Features utiles
features = [
    "amount", "country_risk", "device_risk", "new_device",
    "geo_anomaly", "velocity_24h", "hour",
    "age", "account_age_years", "income",
    "avg_transaction_amount", "preferred_device_risk",
    "fraud_history"
]

X = df[features]
y = df["is_fraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Sauvegarde
joblib.dump(model, "fraud_model.pkl")
print("fraud_model.pkl sauvegardé.")
