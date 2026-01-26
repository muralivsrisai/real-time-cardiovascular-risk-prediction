# app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Wearable Cardiovascular Risk Predictor", layout="centered")

# -------------------------
# Load model + scaler + features
# -------------------------
@st.cache_resource
def load_model_and_scaler():
    model_path_saved = "models/wearable_risk_model.keras"   # SavedModel format
    model_path_h5 = "models/wearable_risk_model.h5"   # HDF5 fallback

    if os.path.exists(model_path_saved):
        model = tf.keras.models.load_model(model_path_saved)
    elif os.path.exists(model_path_h5):
        model = tf.keras.models.load_model(model_path_h5, compile=False)
    else:
        st.stop()  # stop if no model found

    scaler = joblib.load("models/wearable_scaler.pkl")

    try:
        features = pd.read_csv("models/features.csv")["feature"].tolist()
    except:
        # fallback if features.csv missing
        features = [
            "Heart_Rate", "Blood_Oxygen_Level", "ECG", "Skin_Temperature",
            "Sleep_Duration", "Stress_Level", "Age", "Weight",
            "Body_Fat_Percentage", "Calories_Intake"
        ]

    return model, scaler, features

model, scaler, features = load_model_and_scaler()

# -------------------------
# Title & info
# -------------------------
st.title("Wearable-Based Cardiovascular Risk Predictor")
st.markdown("Enter your health data for the selected top 10 features to predict cardiovascular risk.")

# -------------------------
# Manual Input Form
# -------------------------
with st.form("input_form"):
    inputs = {}
    cols = st.columns(2)  # 2-column layout

    for i, feat in enumerate(features):
        col = cols[i % 2]
        
        if feat == "Age":
            inputs[feat] = col.number_input("Age", min_value=1, max_value=120, value=45)
        elif feat == "Gender":
            gender = col.selectbox("Gender", ["Male", "Female", "Other"], index=0)  # default Male
            inputs[feat] = {"Male": 0, "Female": 1, "Other": 2}[gender]
        elif feat == "Medical_Conditions":
            mc = col.selectbox("Medical Conditions", ["None", "Diabetes", "Hypertension", "Other"], index=0)
            inputs[feat] = {"None": 0, "Diabetes": 1, "Hypertension": 2, "Other": 3}[mc]
        elif feat == "Medication":
            med = col.selectbox("Medication", ["None", "Yes"], index=0)
            inputs[feat] = {"None": 0, "Yes": 1}[med]
        elif feat == "Smoker":
            sm = col.selectbox("Smoker", ["No", "Yes"], index=0)
            inputs[feat] = {"No": 0, "Yes": 1}[sm]
        elif feat == "Alcohol_Consumption":
            alc = col.selectbox("Alcohol Consumption", ["None", "Occasional", "Regular"], index=1)  # default occasional
            inputs[feat] = {"None": 0, "Occasional": 1, "Regular": 2}[alc]
        elif feat == "ECG":
            ecg_sel = col.selectbox("ECG", ["Normal", "Abnormal"], index=0)
            inputs[feat] = {"Normal": 1, "Abnormal": 0}[ecg_sel]
        elif feat == "Stress_Level":
            stress = col.selectbox("Stress Level", ["Low", "Moderate", "High"], index=1)  # default Moderate
            inputs[feat] = {"Low": 0, "Moderate": 1, "High": 2}[stress]
        elif feat == "Mood":
            mood = col.selectbox("Mood", ["Happy", "Neutral", "Sad", "Stressed", "Anxious"], index=1)  # default Neutral
            inputs[feat] = {"Happy": 0, "Neutral": 1, "Sad": 2, "Stressed": 3, "Anxious": 4}[mood]
            
        else:
            defaults = {
        "Weight": 70.0,
        "Height": 170.0,
        "Heart_Rate": 75.0,
        "Blood_Oxygen_Level": 98.0,
        "Skin_Temperature": 36.7,
        "Sleep_Duration": 7.0,
        "Deep_Sleep_Duration": 1.5,
        "REM_Sleep_Duration": 2.0,
        "Wakeups": 1.0,
        "Snoring": 0.0,
        "Calories_Intake": 2200.0,
        "Water_Intake": 2.5,
        "Body_Fat_Percentage": 20.0,
        "Muscle_Mass": 30.0,
    }
            inputs[feat] = col.number_input(feat, value=defaults.get(feat, 0.0))


    submitted = st.form_submit_button("Predict Risk")

# -------------------------
# Prediction
# -------------------------
if submitted:
    # Ensure correct order
    input_list = [inputs[feat] for feat in features]
    input_arr = np.array(input_list).reshape(1, -1)

    # Scale & reshape
    input_scaled = scaler.transform(input_arr)
    input_3d = input_scaled.reshape((1, 1, input_scaled.shape[1]))

    # Prediction
    pred = model.predict(input_3d, verbose=0)[0][0]
    risk = float(pred) * 100

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text': "Cardiovascular Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if risk >= 50 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if pred >= 0.5:
        st.error(f"⚠ High Cardiovascular Risk: {risk:.2f}%")
    else:
        st.success(f"✅ Low Cardiovascular Risk: {risk:.2f}%")

    # -------------------------
    # SHAP Feature Importance
    # -------------------------
    st.subheader("Feature Contribution to Prediction")

    try:
        train_data = pd.read_csv("wearables_train_data.csv")
        available_rows = min(50, len(train_data))
        background_data = train_data[features].values
        background_scaled = scaler.transform(background_data)
        background = background_scaled[np.random.choice(background_scaled.shape[0], available_rows, replace=False)]
    except Exception:
        background = np.tile(scaler.mean_.reshape(1, -1), (50, 1))

    def model_predict(data_2d):
        data_3d = data_2d.reshape(data_2d.shape[0], 1, data_2d.shape[1])
        preds = model.predict(data_3d, verbose=0)
        return preds.reshape(-1,)

    explainer = shap.KernelExplainer(model_predict, background)
    shap_vals = explainer.shap_values(input_scaled, nsamples=200)  # reduce nsamples for speed
    sv = np.array(shap_vals).reshape(-1,)
    abs_vals = np.abs(sv)

    if abs_vals.sum() > 0:
        abs_vals = abs_vals / abs_vals.sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, abs_vals, align='center', color='skyblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("|SHAP value| (normalized)")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.caption("Note: SHAP values are approximate. For production, increase background data & nsamples.")
