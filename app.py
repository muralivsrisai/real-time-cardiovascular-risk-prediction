import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import deque
import os

st.set_page_config(page_title="Wearable Cardiovascular Risk Predictor", layout="centered")

TIME_STEPS = 10
SHAP_SAMPLES = 100

def get_dynamic_suggestion(feature, value):
    if feature == "Blood_Oxygen_Level" and value < 95:
        return "Your blood oxygen level is low. Practice deep breathing and consult a professional if it persists."

    if feature == "Sleep_Duration" and value < 7:
        return "Your sleep duration is below recommended levels. Aim for 7–9 hours of sleep."

    if feature == "REM_Sleep_Duration" and value < 1:
        return "Low REM sleep detected. Reducing screen time before bed may help improve sleep quality."

    if feature == "Deep_Sleep_Duration" and value < 1:
        return "Deep sleep duration is low. Maintain a consistent sleep schedule and avoid late caffeine."

    if feature == "Wakeups" and value > 2:
        return "Frequent night awakenings detected. Consider improving sleep environment and bedtime habits."

    if feature == "Water_Intake" and value < 2:
        return "Your water intake appears low. Increasing hydration can help overall cardiovascular health."

    if feature == "Smoker" and value == 1:
        return "Smoking is a significant cardiovascular risk factor. Reducing or quitting can greatly lower risk."

    if feature == "Alcohol_Consumption" and value > 1:
        return "High alcohol consumption detected. Limiting intake can reduce cardiovascular strain."

    if feature == "Age" and value > 50:
        return "Regular moderate physical activity is recommended to maintain cardiovascular health as you age."

    if feature == "Anomaly_Flag" and value == 1:
        return "Irregular physiological readings detected. Monitoring trends and seeking professional advice is recommended."

    return None


# -------------------------
# Load model + scaler + features
# -------------------------
@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("models/wearable_risk_model.keras")
    scaler = joblib.load("models/wearable_scaler.pkl")
    features = pd.read_csv("models/features.csv")["feature"].tolist()
    return model, scaler, features

model, scaler, features = load_all()
N_FEATURES = len(features)

# -------------------------
# Sliding Window Buffer
# -------------------------
if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=TIME_STEPS)

# -------------------------
# Title
# -------------------------
st.title("Wearable-Based Cardiovascular Risk Predictor")
st.caption("Temporal LSTM-based real-time cardiovascular risk assessment")

# -------------------------
# Input Form
# -------------------------
# -------------------------
# Input Form (REALISTIC VALUES)
# -------------------------
with st.form("input_form"):
    inputs = {}
    cols = st.columns(2)

    for i, feat in enumerate(features):
        col = cols[i % 2]

        if feat == "Blood_Oxygen_Level":
            inputs[feat] = col.slider(
                "Blood Oxygen Level (SpO₂ %)", 85.0, 100.0, 98.0
            )

        elif feat == "Sleep_Duration":
            inputs[feat] = col.slider(
                "Total Sleep Duration (hrs)", 0.0, 12.0, 7.0
            )

        elif feat == "REM_Sleep_Duration":
            inputs[feat] = col.slider(
                "REM Sleep Duration (hrs)", 0.0, 4.0, 1.5
            )

        elif feat == "Deep_Sleep_Duration":
            inputs[feat] = col.slider(
                "Deep Sleep Duration (hrs)", 0.0, 4.0, 1.2
            )

        elif feat == "Wakeups":
            inputs[feat] = col.slider(
                "Night Wakeups (count)", 0, 10, 1
            )

        elif feat == "Water_Intake":
            inputs[feat] = col.slider(
                "Daily Water Intake (liters)", 0.5, 6.0, 2.5
            )

        elif feat == "Height":
            inputs[feat] = col.slider(
                "Height (cm)", 130.0, 210.0, 170.0
            )

        elif feat == "Age":
            inputs[feat] = col.slider(
                "Age (years)", 10, 100, 45
            )

        elif feat == "Smoker":
            sm = col.selectbox(
                "Smoker", ["No", "Yes"], index=0
            )
            inputs[feat] = 1 if sm == "Yes" else 0

        elif feat == "Alcohol_Consumption":
            alc = col.selectbox(
                "Alcohol Consumption",
                ["None", "Occasional", "Regular"],
                index=1
            )
            inputs[feat] = {"None": 0, "Occasional": 1, "Regular": 2}[alc]

        elif feat == "Anomaly_Flag":
            an = col.selectbox(
                "Physiological Anomaly Detected",
                ["No", "Yes"],
                index=0
            )
            inputs[feat] = 1 if an == "Yes" else 0

        else:
            # Safe fallback
            inputs[feat] = col.number_input(feat, value=0.0)

    submitted = st.form_submit_button("Add Reading")


# -------------------------
# Prediction
# -------------------------
if submitted:
    x = np.array([inputs[f] for f in features], dtype=float)

    # Handle missing values
    x = np.nan_to_num(x, nan=np.nanmean(x))

    # Scale
    x_scaled = scaler.transform([x])[0]

    # Add to temporal buffer
    st.session_state.buffer.append(x_scaled)

    if len(st.session_state.buffer) < TIME_STEPS:
        st.warning(f"⏳ Collecting data ({len(st.session_state.buffer)}/{TIME_STEPS})")
        st.stop()

    # Prepare LSTM input
    X_window = np.array(st.session_state.buffer).reshape(
        1, TIME_STEPS, N_FEATURES
    )

    # Predict
    pred = model.predict(X_window, verbose=0)[0][0]
    risk = pred * 100

    # -------------------------
    # Gauge
    # -------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "Cardiovascular Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkred" if risk >= 50 else "green"},
            "steps": [
                {"range": [0, 50], "color": "lightgreen"},
                {"range": [50, 100], "color": "lightcoral"},
            ],
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    if pred >= 0.5:
        st.error(f"⚠ High Cardiovascular Risk: {risk:.2f}%")
    else:
        st.success(f"✅ Low Cardiovascular Risk: {risk:.2f}%")

    # =====================================================
    # 🔍 SHAP EXPLAINABILITY (STABLE TEMPORAL VERSION)
    # =====================================================
    st.subheader("Feature Contribution to Prediction")

    # -------------------------
    # Background data (2D)
    # -------------------------
    try:
        train_data = pd.read_csv("data/personal_health_data.csv")
        bg = train_data[features].values
        bg = scaler.transform(bg)
        bg = bg[:SHAP_SAMPLES]
    except Exception:
        bg = np.tile(scaler.mean_, (SHAP_SAMPLES, 1))


    # -------------------------
    # Wrapper: repeat timestep
    # -------------------------
    def model_predict_2d(X_2d):
        # Repeat current state across TIME_STEPS
        X_3d = np.repeat(
            X_2d[:, None, :],
            TIME_STEPS,
            axis=1
        )
        return model.predict(X_3d, verbose=0).reshape(-1)


    # Current timestep only
    current_step = X_window[:, -1, :]

    explainer = shap.KernelExplainer(model_predict_2d, bg)

    shap_vals = explainer.shap_values(
        current_step, nsamples=150
    )

    shap_vals = np.array(shap_vals).reshape(-1)

    # Normalize importance
    importance = np.abs(shap_vals)
    importance = importance / importance.sum()

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    y_pos = np.arange(N_FEATURES)

    ax.barh(y_pos, importance, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized SHAP Importance")
    ax.set_title("Feature Contribution (Current Time Step)")

    st.pyplot(fig)

    st.caption(
        "SHAP explanation computed for the most recent time step. "
        "Temporal context is preserved by repeating the state across the LSTM window."
    )

    # =====================================================
    # 🩺 Dynamic Risk Reduction Suggestions
    # ====================================================
    st.subheader("Personalized Risk Reduction Suggestions")

    if pred >= 0.5:
        top_indices = np.argsort(importance)[-3:][::-1]
        shown = set()

        for idx in top_indices:
            feat = features[idx]
            current_value = inputs[feat]

            suggestion = get_dynamic_suggestion(feat, current_value)
            if suggestion and feat not in shown:
                st.markdown(f"• **{suggestion}**")
                shown.add(feat)

        if not shown:
            st.info(
                "No specific lifestyle risks detected from current readings. Maintain healthy habits."
            )

        st.caption(
            "Suggestions are general wellness guidance and not a medical diagnosis."
        )
    else:
        st.success(
            "Your current readings indicate low cardiovascular risk. Keep maintaining a healthy lifestyle."
        )

