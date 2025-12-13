import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes.csv")   # change name if needed

# Features & target
FEATURES = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']
X = df[FEATURES]
y = df['Outcome']   # 0 = No diabetes, 1 = Diabetes

# Train-test split (same logic used in training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load("final_rf_model.joblib")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

# Extra evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))