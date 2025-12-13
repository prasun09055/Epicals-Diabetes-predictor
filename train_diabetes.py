# train_diabetes.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Load
df = pd.read_csv("diabetes.csv")
print("Loaded shape:", df.shape)

# 2) Replace zeros with medians for some cols
cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for c in cols_with_zero:
    if c in df.columns:
        med = df.loc[df[c] > 0, c].median()
        df[c] = df[c].replace(0, med)

# 3) Features and split
features = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']
X = df[features]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Scale (for models that need it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.joblib")

# 5) Train Logistic Regression baseline
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, "logreg_model.joblib")

# 6) Train Random Forest (on unscaled features)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "rf_model.joblib")
joblib.dump(rf, "final_rf_model.joblib")

# 7) Evaluate RF
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]
print("RF accuracy:", accuracy_score(y_test, y_pred))
print("RF ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# 8) Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - RF")
plt.savefig("confusion_matrix.png", bbox_inches='tight')
plt.close()

# 9) ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"RF (AUC={roc_auc_score(y_test,y_proba):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png", bbox_inches='tight')
plt.close()

# 10) Feature importance
import pandas as pd
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature importances:\n", fi)
fi.to_csv("feature_importances.csv")
