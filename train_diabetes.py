import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ================= LOAD DATA =================
df = pd.read_csv("diabetes.csv")

FEATURES = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']
TARGET = 'Outcome'

X = df[FEATURES]
y = df[TARGET]

print("Class distribution:")
print(y.value_counts(normalize=True))

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================= SCALE =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= MODEL (IMPORTANT FIXES) =================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=10,
    class_weight="balanced",   # 🔥 KEY FIX
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ================= EVALUATE =================
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================= SAVE =================
joblib.dump(model, "final_rf_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("\n✅ Model and scaler saved successfully")
