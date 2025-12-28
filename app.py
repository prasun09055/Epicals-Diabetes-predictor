import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Epicals — Diabetes Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLING =================
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.04);
    padding: 1rem;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ================= PATHS =================
MODEL_PATH = "final_rf_model.joblib"
SCALER_PATH = "scaler.joblib"

FEATURES = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# ================= SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= SIDEBAR =================
with st.sidebar:
    st.title("🧑 Patient Inputs")

    input_mode = st.radio(
        "Input Mode",
        ["Slider (Drag)", "Manual (Type)"],
        horizontal=True
    )

    if input_mode == "Slider (Drag)":
        preg = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 50, 300, 120)
        bp = st.slider("Blood Pressure (mm Hg)", 40, 200, 70)
        insulin = st.slider("Insulin (µU/mL)", 0.0, 600.0, 80.0)
        bmi = st.slider("BMI", 10.0, 60.0, 25.0, step=0.1)
        age = st.slider("Age", 1, 120, 33)
    else:
        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
        bp = st.number_input("Blood Pressure (mm Hg)", 40, 200, 70)
        insulin = st.number_input("Insulin (µU/mL)", 0.0, 600.0, 80.0)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
        age = st.number_input("Age", 1, 120, 33)

# ================= INPUT DATA =================
input_df = pd.DataFrame(
    [[preg, glucose, bp, insulin, bmi, age]],
    columns=FEATURES
)

X = scaler.transform(input_df)
prob = model.predict_proba(X)[0][1]
pred = model.predict(X)[0]

# ================= SAVE TO HISTORY =================
record = {
    "Pregnancies": preg,
    "Glucose": glucose,
    "BloodPressure": bp,
    "Insulin": insulin,
    "BMI": bmi,
    "Age": age,
    "Risk_Probability": round(prob, 3),
    "Prediction": "Diabetic" if pred == 1 else "Not Diabetic"
}

if not st.session_state.history or st.session_state.history[-1] != record:
    st.session_state.history.append(record)

# ================= MAIN =================
st.title("🩺 Epicals — Diabetes Risk Predictor")
st.caption("Educational use only — not a medical diagnosis")

col1, col2 = st.columns(2)

# ---------- PREDICTION ----------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔮 Live Risk Prediction")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "#ff4b4b" if prob >= 0.5 else "#00ff9c"},
            'steps': [
                {'range': [0,30], 'color': "#00ff9c"},
                {'range': [30,60], 'color': "#ffe066"},
                {'range': [60,100], 'color': "#ff4b4b"}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'value': 50}
        }
    ))
    fig.update_layout(height=260)
    st.plotly_chart(fig, use_container_width=True)

    if pred == 1:
        st.error(f"⚠️ High diabetes risk detected ({prob:.1%})")
    else:
        st.success(f"✅ Low diabetes risk detected ({prob:.1%})")

    st.progress(int(prob * 100))
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- SUGGESTIONS ----------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 What can you do?")

    if prob >= 0.75:
        st.error("""
**High Risk – Immediate Action Needed**
- Consult a doctor (HbA1c, fasting glucose)
- Reduce sugar & refined carbs
- Regular physical activity
- Weight management
- Monitor blood glucose
""")
    elif prob >= 0.4:
        st.warning("""
**Moderate Risk – Be Careful**
- Improve diet quality
- Exercise regularly
- Reduce stress
- Periodic health check-ups
""")
    else:
        st.success("""
**Low Risk – You’re Good 👍**
- Maintain healthy lifestyle
- Balanced diet & exercise
- Annual screening recommended
""")

    st.caption("Suggestions are general guidance only.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- FEATURE IMPORTANCE ----------
st.markdown("---")
st.subheader("📊 Feature Importance")

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
fig_fi = px.bar(
    fi,
    orientation="h",
    labels={"value": "Importance", "index": "Feature"}
)
st.plotly_chart(fig_fi, use_container_width=True)

# ---------- HISTORY ----------
st.markdown("---")
st.subheader("📁 Prediction History")

hist_df = pd.DataFrame(st.session_state.history)
st.dataframe(hist_df, use_container_width=True)

st.download_button(
    "⬇️ Download History CSV",
    hist_df.to_csv(index=False).encode("utf-8"),
    "prediction_history.csv",
    "text/csv"
)

if st.button("🗑️ Clear History"):
    st.session_state.history = []
    st.rerun()


st.caption("Epicals Diabetes Risk Predictor • Educational Use Only")
