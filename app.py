
import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature list
model = joblib.load("lung_cancer_model.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Lung Cancer Prediction", layout="centered")

st.title("🫁 Lung Cancer Prediction App")
st.write("Predicts the likelihood of lung cancer based on patient symptoms.")

st.markdown("---")

# Helper for Yes/No input
def yes_no(label):
    return st.selectbox(label, ["No", "Yes"])

# User inputs
GENDER = st.selectbox("Gender", ["Male", "Female"])
AGE = st.number_input("Age", min_value=1, max_value=100, value=45)

SMOKING = yes_no("Smoking")
YELLOW_FINGERS = yes_no("Yellow Fingers")
ANXIETY = yes_no("Anxiety")
PEER_PRESSURE = yes_no("Peer Pressure")
CHRONIC_DISEASE = yes_no("Chronic Disease")
FATIGUE = yes_no("Fatigue")
ALLERGY = yes_no("Allergy")
WHEEZING = yes_no("Wheezing")
ALCOHOL_CONSUMING = yes_no("Alcohol Consuming")
COUGHING = yes_no("Coughing")
SHORTNESS_OF_BREATH = yes_no("Shortness of Breath")
SWALLOWING_DIFFICULTY = yes_no("Swallowing Difficulty")
CHEST_PAIN = yes_no("Chest Pain")

st.markdown("---")

# Predict button
if st.button("🔍 Predict"):

    data = pd.DataFrame(columns=features)

    data.loc[0] = 0  # initialize row

    data["GENDER"] = 1 if GENDER == "Male" else 0
    data["AGE"] = AGE
    data["SMOKING"] = 1 if SMOKING == "Yes" else 0
    data["YELLOW_FINGERS"] = 1 if YELLOW_FINGERS == "Yes" else 0
    data["ANXIETY"] = 1 if ANXIETY == "Yes" else 0
    data["PEER_PRESSURE"] = 1 if PEER_PRESSURE == "Yes" else 0
    data["CHRONIC_DISEASE"] = 1 if CHRONIC_DISEASE == "Yes" else 0
    data["FATIGUE"] = 1 if FATIGUE == "Yes" else 0
    data["ALLERGY"] = 1 if ALLERGY == "Yes" else 0
    data["WHEEZING"] = 1 if WHEEZING == "Yes" else 0
    data["ALCOHOL_CONSUMING"] = 1 if ALCOHOL_CONSUMING == "Yes" else 0
    data["COUGHING"] = 1 if COUGHING == "Yes" else 0
    data["SHORTNESS_OF_BREATH"] = 1 if SHORTNESS_OF_BREATH == "Yes" else 0
    data["SWALLOWING_DIFFICULTY"] = 1 if SWALLOWING_DIFFICULTY == "Yes" else 0
    data["CHEST_PAIN"] = 1 if CHEST_PAIN == "Yes" else 0

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Lung Cancer Detected\n\nRisk Probability: {probability:.2%}")
    else:
        st.success(f"✅ No Lung Cancer Detected\n\nRisk Probability: {probability:.2%}")
