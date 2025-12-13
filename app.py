# app.py - Streamlit app (SHAP and Batch CSV removed)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ---------------- Page & caching ----------------
st.set_page_config(page_title="Interactive Diabetes Predictor", layout="wide")

MODEL_PATH = "final_rf_model.joblib"
SCALER_PATH = "scaler.joblib"
ROC_IMAGE = "roc_curve.png"
CM_IMAGE = "confusion_matrix.png"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        st.error(f"Model file not found at `{path}`. Run training first and place the model file here.")
        st.stop()
    return joblib.load(path)

@st.cache_resource
def load_scaler(path=SCALER_PATH):
    if Path(path).exists():
        return joblib.load(path)
    return None

model = load_model()
scaler = load_scaler()

# FEATURES order must match the model's training features
FEATURES = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','Age']

# ---------------- Utilities ----------------
def predict_from_df(df_input):
    """Return preds, probs (numpy arrays). Handles scaling toggle if needed."""
    if scaler is not None:
        # If your model was trained on scaled features, uncomment the next line
        # X = scaler.transform(df_input)
        X = df_input.values
    else:
        X = df_input.values
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return preds, probs

def feature_contributions_proxy(row, model, baseline=None):
    """Lightweight proxy for feature contribution: (x - baseline) * importance"""
    importances = model.feature_importances_
    if baseline is None:
        baseline = np.zeros_like(row)
    contrib = (row - baseline) * importances
    return contrib, importances

# ---------------- Session-state for history ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts
if "last_input" not in st.session_state:
    st.session_state["last_input"] = None

# ---------------- Sidebar (input mode + controls) ----------------

with st.sidebar:
    st.image("logo.png", width=120)
    st.markdown("## EPICALS")
    st.caption("AI Health Intelligence")
    st.markdown("---")
    st.title("Patient Input")
    input_mode = st.radio("Input mode", options=["Slider (drag)", "Manual (number)"], index=0)
    st.markdown("**Controls**")
    if input_mode == "Slider (drag)":
        preg = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose (mg/dL)", 1, 300, 120)
        bp = st.slider("Blood Pressure (mm Hg)", 1, 200, 70)
        ins = st.slider("Insulin (µU/mL)", 0.0, 1000.0, 80.0)
        bmi = st.slider("BMI", 10.0, 70.0, 25.0, step=0.1)
        age = st.slider("Age", 1, 120, 33)
    else:
        preg = st.number_input("Pregnancies", min_value=0, max_value=100, value=1, step=1)
        glucose = st.number_input("Glucose (mg/dL)", min_value=1, max_value=1000, value=120, step=1)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=1, max_value=500, value=70, step=1)
        ins = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=10000.0, value=80.0, step=0.1, format="%.1f")
        bmi = st.number_input("BMI", min_value=1.0, max_value=200.0, value=25.0, step=0.1, format="%.1f")
        age = st.number_input("Age", min_value=0, max_value=200, value=33, step=1)

    st.markdown("---")
    st.checkbox("Show raw model input table", value=False, key="show_input")
    st.markdown(f"Model file: `{MODEL_PATH}`")

# Build input DataFrame
input_df = pd.DataFrame([[preg, glucose, bp, ins, bmi, age]], columns=FEATURES)

# ---------------- Main layout ----------------
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = get_base64_image("logo.png")

st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:center; gap:20px;">
        <img src="data:image/png;base64,{logo}" width="90">
        <h1 style='font-size:42px; font-weight:900;
        background: -webkit-linear-gradient(0deg, #4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Epicals — Diabetes Risk Predictor
        </h1>
    </div>

    <h4 style='text-align:center; margin-top:-10px; opacity:0.8;'>
        Smart • Fast • Interactive Health Insight
    </h4>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns([2,3])

# ------------ Left column: live prediction & history controls ------------
with col1:
    st.subheader("Live prediction")

    if st.session_state.get("show_input"):
        st.dataframe(input_df)

    # Predict
    preds, probs = predict_from_df(input_df)
    prob = float(probs[0])
    pred = int(preds[0])

    # Save to history (avoid duplicates)
    record = {
        "Pregnancies": int(preg),
        "Glucose": float(glucose),
        "BloodPressure": float(bp),
        "Insulin": float(ins),
        "BMI": float(bmi),
        "Age": int(age),
        "Probability": float(prob),
        "Prediction": "Diabetic" if pred == 1 else "Not Diabetic"
    }
    if st.session_state["last_input"] != record:
        st.session_state["history"].append(record)
        st.session_state["last_input"] = record

    # Gauge (Plotly)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob*100,
        number={'suffix': '%', 'valueformat': '.1f'},
        title={'text': "Predicted Diabetes Risk"},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "darkred" if prob >= 0.5 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "#d4f0d4"},
                {'range': [30, 60], 'color': "#fff7b2"},
                {'range': [60, 100], 'color': "#ffd2d2"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Textual prediction and progress
    st.markdown("### Prediction")
    if pred == 1:
        st.error("**Diabetic** — model suggests elevated risk.")
    else:
        st.success("**Not Diabetic** — model suggests lower risk.")
    st.markdown("#### Probability")
    st.progress(int(prob*100))

    st.markdown("---")
    # History controls
    st.subheader("History controls")
    hist_len = len(st.session_state["history"])
    st.write(f"Saved predictions: **{hist_len}**")
    if hist_len > 0:
        # Download button uses its own widget (not nested inside if), so provide on-demand download
        hist_df_dl = pd.DataFrame(st.session_state["history"])
        st.download_button("Download history CSV", data=hist_df_dl.to_csv(index=False).encode("utf-8"),
                           file_name="prediction_history.csv", mime="text/csv")
        if st.button("Clear history"):
            st.session_state["history"] = []
            st.session_state["last_input"] = None
            st.experimental_rerun()
    else:
        st.info("No history yet — adjust inputs to record predictions.")

# ------------ Right column: importance & contributions ------------
with col2:
    st.subheader("Feature importance & contributions")

    # Feature importance bar (build stable dataframe)
    try:
        fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
    except Exception:
        fi = pd.Series(np.ones(len(FEATURES))/len(FEATURES), index=FEATURES)

    df_fi = fi.reset_index().rename(columns={'index': 'Feature'})
    # create Importance column explicitly (the values will be in the second column)
    value_col_fi = df_fi.columns[1]
    df_fi = df_fi.rename(columns={value_col_fi: 'Importance'})

    fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    fig_fi.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Contribution proxy (stable dataframe)
    contrib_vals, importances = feature_contributions_proxy(input_df.values.flatten(), model, baseline=np.zeros(len(FEATURES)))
    contrib_series = pd.Series(contrib_vals, index=FEATURES).sort_values(ascending=True)
    df_contrib = contrib_series.reset_index().rename(columns={'index': 'Feature'})
    value_col_contrib = df_contrib.columns[1]
    df_contrib = df_contrib.rename(columns={value_col_contrib: 'Contribution'})

    st.markdown("#### Contribution (proxy): (value - baseline) * importance")
    fig_contrib = px.bar(df_contrib, x='Contribution', y='Feature', orientation='h', height=300)
    st.plotly_chart(fig_contrib, use_container_width=True)

# ---------------- History panel (table + trend + download/clear) ----------------
st.markdown("---")
st.subheader(" Prediction History (Automatic)")

history = st.session_state["history"]
if len(history) == 0:
    st.info("No history yet. Adjust inputs to make your first prediction.")
else:
    hist_df = pd.DataFrame(history)
    st.dataframe(hist_df, use_container_width=True)

    st.markdown("### Risk Trend")
    fig_hist = px.line(hist_df.reset_index(), y="Probability", title="Prediction Risk Trend", markers=True)
    fig_hist.update_layout(xaxis_title="Record index", yaxis=dict(range=[0,1]))
    st.plotly_chart(fig_hist, use_container_width=True)

    csv_history = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download History CSV", data=csv_history, file_name="history.csv", mime="text/csv")

# ---------------- Compare two records ----------------
st.markdown("---")
st.subheader(" Compare Two Predictions")

if len(history) < 2:
    st.info("Need at least two predictions to compare.")
else:
    hist_df = pd.DataFrame(history)
    idx_options = list(range(len(hist_df)))
    c1, c2 = st.columns(2)
    with c1:
        idx1 = st.selectbox("Select Record 1", idx_options, format_func=lambda i: f"#{i+1} - {hist_df.iloc[i]['Prediction']}, p={hist_df.iloc[i]['Probability']:.2f}")
    with c2:
        idx2 = st.selectbox("Select Record 2", idx_options, index=1, format_func=lambda i: f"#{i+1} - {hist_df.iloc[i]['Prediction']}, p={hist_df.iloc[i]['Probability']:.2f}")

    if idx1 == idx2:
        st.warning("Pick two different records to compare.")
    else:
        rec1 = hist_df.iloc[idx1]
        rec2 = hist_df.iloc[idx2]
        st.markdown("### Side-by-side")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Record 1")
            st.write(rec1)
        with right:
            st.markdown("#### Record 2")
            st.write(rec2)

        diff = rec2["Probability"] - rec1["Probability"]
        st.markdown("### Risk change")
        if diff > 0:
            st.error(f"Risk increased by {diff:.2f}")
        elif diff < 0:
            st.success(f"Risk decreased by {abs(diff):.2f}")
        else:
            st.info("Risk did not change.")

        # Feature differences (fixed)
        diffs = (rec2[FEATURES].astype(float) - rec1[FEATURES].astype(float))

        # Build a clean dataframe with explicit column names
        df_diffs = diffs.reset_index().rename(columns={'index': 'Feature'})
        # After reset_index numeric values are usually in second column; rename it to 'Change'
        if df_diffs.shape[1] >= 2:
            value_col = df_diffs.columns[1]
            df_diffs = df_diffs.rename(columns={value_col: 'Change'})
        else:
            # fallback (shouldn't happen): compute Change explicitly
            df_diffs['Change'] = df_diffs.iloc[:, 1:].sum(axis=1)

        df_diffs['Change'] = df_diffs['Change'].astype(float)

        fig_diff = px.bar(df_diffs, x='Change', y='Feature', orientation='h',
                          labels={'Change': 'Change (Record2 - Record1)', 'Feature': 'Feature'},
                          title="Feature Differences (Record2 - Record1)")
        fig_diff.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_diff, use_container_width=True)

# ---------------- "What to do" actionable recommendations ----------------
st.markdown("---")
st.subheader(" What to do — personalized suggestions (educational)")

def get_recommendations(inputs: dict, contrib: pd.Series, prob: float):
    recs = []
    advice_map = {
        "Glucose": [
            "Check fasting glucose and HbA1c with your doctor to confirm blood sugar control.",
            "Reduce intake of refined carbohydrates & sugary drinks; prefer whole grains and fiber-rich foods.",
            "Monitor glucose regularly if advised (use a glucometer)."
        ],
        "BMI": [
            "Aim for modest weight loss (5–10%) if overweight — it often reduces diabetes risk.",
            "Increase daily activity: try 30 minutes of moderate exercise most days.",
            "Follow a balanced diet (smaller portions, more vegetables, lean protein)."
        ],
        "BloodPressure": [
            "High blood pressure increases cardiovascular risk; consult your doctor for targets and treatment.",
            "Reduce salt intake, manage stress, and increase aerobic activity.",
            "If on medication, take as prescribed and monitor regularly."
        ],
        "Insulin": [
            "Insulin values may reflect metabolic changes — discuss with a clinician or endocrinologist.",
            "Regular physical activity improves insulin sensitivity.",
            "Avoid long periods of inactivity; break sitting time frequently."
        ],
        "Pregnancies": [
            "If pregnancy-related: ensure obstetric follow-up for gestational diabetes screening.",
            "Maintain healthy weight gain and follow nutritional guidance from your healthcare team."
        ],
        "Age": [
            "Age is non-modifiable — focus on controllable factors (weight, diet, activity, BP).",
            "Annual check-ups and screening tests become more important with age."
        ]
    }

    # Header based on probability
    if prob >= 0.8:
        recs.append(("Urgent", "High model-predicted risk — seek medical evaluation promptly."))
    elif prob >= 0.5:
        recs.append(("Important", "Moderate to high model-predicted risk — schedule a check-up and tests (fasting glucose, HbA1c)."))
    else:
        recs.append(("Note", "Low model-predicted risk. Maintain healthy lifestyle and routine monitoring."))

    # Prioritize top contributors
    contrib_sorted = contrib.sort_values(ascending=False)
    topn = min(3, len(contrib_sorted))
    for feat in contrib_sorted.index[:topn]:
        val = inputs.get(feat, None)
        if val is None:
            continue
        if contrib_sorted.loc[feat] > 0 or (feat == "Glucose" and val > 140) or (feat == "BMI" and val >= 25) or (feat == "BloodPressure" and val >= 130):
            tips = advice_map.get(feat, ["Focus on general healthy habits."])
            recs.append((feat, tips[0]))
            recs.append((feat + "_details", " · ".join(tips[1:])))
    recs.append(("General", "Adopt regular exercise, a fiber-rich diet, reduce sugary foods, and have routine health checks."))
    recs.append(("Disclaimer", "This tool provides educational suggestions only — not a medical diagnosis. Consult a healthcare professional for personalized care."))
    return recs

# Build inputs dict and contrib_series
inputs_dict = {f: input_df.iloc[0][f] for f in FEATURES}
contrib_vals, _ = feature_contributions_proxy(input_df.values.flatten(), model, baseline=np.zeros(len(FEATURES)))
contrib_series = pd.Series(contrib_vals, index=FEATURES)

recommendations = get_recommendations(inputs_dict, contrib_series, prob)

for tag, text in recommendations:
    if tag == "Urgent":
        st.error(f"**{text}**")
    elif tag == "Important":
        st.warning(f"**{text}**")
    elif tag == "Note":
        st.info(text)
    elif tag == "Disclaimer":
        st.caption(text)
    elif tag.endswith("_details"):
        st.markdown(f"<small style='color:#555'>{text}</small>", unsafe_allow_html=True)
    else:
        st.markdown(f"**{tag}:** {text}")

# ---------------- Footer: show ROC & CM if available (responsive images) ----------------
st.markdown("---")
st.write("Model artifacts (if available):")
cols = st.columns(2)
if Path(ROC_IMAGE).exists():
    cols[0].image(ROC_IMAGE, caption="ROC Curve")
if Path(CM_IMAGE).exists():
    cols[1].image(CM_IMAGE, caption="Confusion Matrix")

st.caption("Model is for demonstration and educational purposes only — not a medical diagnosis tool.")