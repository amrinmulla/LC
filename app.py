import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model & features
# =========================
model = joblib.load("lung_cancer_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")

st.title("🫁 Lung Cancer Prediction App")
st.write(
    "This application predicts the likelihood of lung cancer "
    "based on patient symptoms."
)

st.markdown("---")

# =========================
# Helper function
# =========================
def yes_no(label):
    return st.selectbox(label, ["No", "Yes"])

# =========================
# User Inputs (UI)
# =========================
gender_ui = st.selectbox("Gender", ["Male", "Female"])
age_ui = st.number_input("Age", min_value=1, max_value=100, value=45)

smoking_ui = yes_no("Smoking")
yellow_fingers_ui = yes_no("Yellow Fingers")
anxiety_ui = yes_no("Anxiety")
peer_pressure_ui = yes_no("Peer Pressure")
chronic_disease_ui = yes_no("Chronic Disease")
fatigue_ui = yes_no("Fatigue")
allergy_ui = yes_no("Allergy")
wheezing_ui = yes_no("Wheezing")
alcohol_consuming_ui = yes_no("Alcohol Consuming")
coughing_ui = yes_no("Coughing")
shortness_of_breath_ui = yes_no("Shortness of Breath")
swallowing_difficulty_ui = yes_no("Swallowing Difficulty")
chest_pain_ui = yes_no("Chest Pain")

st.markdown("---")

# =========================
# Encode inputs for MODEL
# =========================
if st.button("🔍 Predict"):

    encoded_data = {
        "GENDER": 1 if gender_ui == "Male" else 0,
        "AGE": age_ui,
        "SMOKING": 1 if smoking_ui == "Yes" else 0,
        "YELLOW_FINGERS": 1 if yellow_fingers_ui == "Yes" else 0,
        "ANXIETY": 1 if anxiety_ui == "Yes" else 0,
        "PEER_PRESSURE": 1 if peer_pressure_ui == "Yes" else 0,
        "CHRONIC_DISEASE": 1 if chronic_disease_ui == "Yes" else 0,
        "FATIGUE": 1 if fatigue_ui == "Yes" else 0,
        "ALLERGY": 1 if allergy_ui == "Yes" else 0,
        "WHEEZING": 1 if wheezing_ui == "Yes" else 0,
        "ALCOHOL_CONSUMING": 1 if alcohol_consuming_ui == "Yes" else 0,
        "COUGHING": 1 if coughing_ui == "Yes" else 0,
        "SHORTNESS_OF_BREATH": 1 if shortness_of_breath_ui == "Yes" else 0,
        "SWALLOWING_DIFFICULTY": 1 if swallowing_difficulty_ui == "Yes" else 0,
        "CHEST_PAIN": 1 if chest_pain_ui == "Yes" else 0
    }

    # Create DataFrame
    input_df = pd.DataFrame([encoded_data])

    # Ensure exact training feature order
    input_df = input_df.reindex(columns=features, fill_value=0)

    # =========================
    # Prediction
    # =========================
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    # =========================
    # Output
    # =========================
    if prediction[0] == 1:
        st.error(f"⚠️ Lung Cancer Detected\n\nRisk Probability: {probability:.2%}")
    else:
        st.success(f"✅ No Lung Cancer Detected\n\nRisk Probability: {probability:.2%}")

st.markdown("---")
st.caption("⚠️ This app is for educational purposes only and not a medical diagnosis.")
