import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Network Intrusion Detection System", layout="wide")

# ------------------------------
# Load trained models with caching
# ------------------------------
@st.cache_resource
def load_models():
    rf_model = joblib.load("rf_model.pkl")
    gb_model = joblib.load("gb_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    return rf_model, gb_model, xgb_model

rf_model, gb_model, xgb_model = load_models()

# Load label encoders
le = joblib.load("label_encoder.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ------------------------------
# Background Styling
# ------------------------------
detect_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://wallpaperaccess.com/full/1267581.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
}
.stTextInput label,
.stNumberInput label {
    color: white;
    font-weight: bold;
}
</style>
'''

shap_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://www.analyticsinsight.net/wp-content/uploads/2021/06/Explainable-AI-XAI.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
}
</style>
'''

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detect", "SHAP Analysis"])

# ------------------------------
# Feature Names
# ------------------------------
feature_names = [
    "Communication Duration", "Communication Protocol", "Service Type",
    "Source Data Volume", "Destination Data Volume", "Communication Count",
    "Service Communication Count", "Session Error Rate", "Service Error Rate",
    "Reception Error Rate", "Service Reception Error Rate",
    "Same Service Communication Rate", "Different Service Communication Rate",
    "Destination Host Count"
]

# ------------------------------
# Detect Page
# ------------------------------
if page == "Detect":
    st.markdown(detect_bg_img, unsafe_allow_html=True)
    st.markdown('<h1 style="color:white;">Network Intrusion Detection in VANETs</h1>', unsafe_allow_html=True)

    with st.form("detect_form"):
        inputs = {}
        for feature in feature_names:
            if feature in ["Communication Protocol", "Service Type"]:
                inputs[feature] = st.text_input(feature, key=feature)
            else:
                inputs[feature] = st.number_input(feature, key=feature)

        if st.form_submit_button("Submit"):
            # Check if all inputs are empty
            if (inputs["Communication Protocol"] == "" and inputs["Service Type"] == "") and \
               all(value == 0 for key, value in inputs.items() if key not in ["Communication Protocol", "Service Type"]):
                st.warning("All the input features must be filled.")
                st.stop()

            # Encode categorical inputs
            d1 = label_encoders["Communication Protocol"].transform([inputs["Communication Protocol"]])[0] \
                if inputs["Communication Protocol"] in label_encoders["Communication Protocol"].classes_ else -1
            d2 = label_encoders["Service Type"].transform([inputs["Service Type"]])[0] \
                if inputs["Service Type"] in label_encoders["Service Type"].classes_ else -1

            # Prepare features array
            features = np.array([[
                inputs[f] if f not in ["Communication Protocol", "Service Type"]
                else (d1 if f == "Communication Protocol" else d2)
                for f in feature_names
            ]])

            # Predictions
            rf_preds = rf_model.predict_proba(features)
            gb_preds = gb_model.predict_proba(features)
            xgb_preds = xgb_model.predict_proba(features)

            # Weighted blending
            weights = [0.4, 0.3, 0.3]
            blended_preds = (weights[0] * rf_preds + weights[1] * gb_preds + weights[2] * xgb_preds)
            final_preds = np.argmax(blended_preds, axis=1)

            # Decode predictions
            decoded_predictions = le.inverse_transform(final_preds)

            st.session_state["features"] = features
            st.session_state["prediction"] = decoded_predictions[0]

            if "attack" in decoded_predictions[0].lower():
                st.markdown(f'<p style="color:red; font-size:24px; font-weight:bold;">Alert: {decoded_predictions[0]}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:green; font-size:24px; font-weight:bold;">{decoded_predictions[0]}</p>', unsafe_allow_html=True)

# ------------------------------
# SHAP Analysis Page
# ------------------------------
elif page == "SHAP Analysis":
    st.markdown(detect_bg_img, unsafe_allow_html=True)
    st.markdown('<h1 style="color: white;">Feature Importance and SHAP Analysis</h1>', unsafe_allow_html=True)

    if "features" in st.session_state:
        explainer = shap.Explainer(rf_model)
        shap_values = explainer(st.session_state["features"])

        # Feature Importance Plot
        st.markdown('<h2 style="color:white;">Feature Importance (SHAP)</h2>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        shap.summary_plot(shap_values.values[:, :, 0], st.session_state["features"],
                          feature_names=feature_names, show=False)
        st.pyplot(fig)
    else:
        st.markdown('<p style="color:red; font-size:18px; font-weight:bold;">⚠ Please run detection first in the "Detect" section.</p>', unsafe_allow_html=True)
