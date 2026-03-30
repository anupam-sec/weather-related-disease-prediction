import streamlit as st
import joblib
import numpy as np
import os

# Updated code for Streamlit upload from github
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "gradient_boost_model.pkl")

# Load the model safely
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

st.title("🌦️ Weather-Related Disease Predictor")

age = st.number_input("Enter Age")
gender = st.selectbox("Gender", ["Male", "Female"])
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed (km/h)")

nausea = st.selectbox("Nausea", ["No", "Yes"])
joint_pain = st.selectbox("Joint Pain", ["No", "Yes"])
abdominal_pain = st.selectbox("Abdominal Pain", ["No", "Yes"])
high_fever = st.selectbox("High Fever", ["No", "Yes"])
chills = st.selectbox("Chills", ["No", "Yes"])

fatigue = st.selectbox("Fatigue", ["No", "Yes"])
runny_nose = st.selectbox("Runny Nose", ["No", "Yes"])
pain_behind_the_eyes = st.selectbox("Pain Behind the Eyes", ["No", "Yes"])
dizziness = st.selectbox("Dizziness", ["No", "Yes"])
headache = st.selectbox("Headache", ["No", "Yes"])

chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
vomiting = st.selectbox("Vomitting", ["No", "Yes"])
cough = st.selectbox("Cough", ["No", "Yes"])

hiv_aids = st.selectbox("HIV AIDS", ["No", "Yes"])
nasal_polyps = st.selectbox("Nasal Polyps", ["No", "Yes"])

asthma = st.selectbox("Asthma", ["No", "Yes"])
high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])
severe_headache = st.selectbox("Severe Headache", ["No", "Yes"])
weakness = st.selectbox("Weakness", ["No", "Yes"])
trouble_seeing = st.selectbox("Trouble Seeing", ["No", "Yes"])

fever = st.selectbox("Fever", ["No", "Yes"])
body_aches = st.selectbox("Body Aches", ["No", "Yes"])
sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])
sneezing = st.selectbox("Sneezing", ["No", "Yes"])
diarrhea = st.selectbox("Diarrhea", ["No", "Yes"])

rapid_breathing = st.selectbox("Rapid Breathing", ["No", "Yes"])
rapid_heart_rate = st.selectbox("Rapid Heart Rate", ["No", "Yes"])
pain_behind_eyes = st.selectbox("Pain Behind Eyes", ["No", "Yes"])
swollen_glands = st.selectbox("Swollen Glands", ["No", "Yes"])
rashes = st.selectbox("Rashes", ["No", "Yes"])

sinus_headache = st.selectbox("Sinus Headache", ["No", "Yes"])
facial_pain = st.selectbox("Facial Pain", ["No", "Yes"])
shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
reduced_smell_and_taste = st.selectbox("Reduced Smell and Taste", ["No", "Yes"])
skin_irritation = st.selectbox("Skin Irritation", ["No", "Yes"])

itchiness = st.selectbox("Itchiness", ["No", "Yes"])
throbbing_headache = st.selectbox("Throbbing Headache", ["No", "Yes"])
confusion = st.selectbox("Confusion", ["No", "Yes"])
back_pain = st.selectbox("Back Pain", ["No", "Yes"])
knee_ache = st.selectbox("Knee Ache", ["No", "Yes"])




gender_val = 1 if gender == "Male" else 0
nausea_val = 1 if nausea == "Yes" else 0

joint_pain_val = 1 if joint_pain == "Yes" else 0
abdominal_pain_val = 1 if abdominal_pain == "Yes" else 0
high_fever_val = 1 if high_fever == "Yes" else 0
chills_val = 1 if chills == "Yes" else 0

fatigue_val = 1 if fatigue == "Yes" else 0
runny_nose_val = 1 if runny_nose == "Yes" else 0
pain_behind_the_eyes_val = 1 if pain_behind_the_eyes == "Yes" else 0
dizziness_val = 1 if dizziness == "Yes" else 0
headache_val = 1 if headache == "Yes" else 0

chest_pain_val = 1 if chest_pain == "Yes" else 0
vomiting_val = 1 if vomiting == "Yes" else 0
cough_val = 1 if cough == "Yes" else 0

hiv_aids_val = 1 if hiv_aids == "Yes" else 0
nasal_polyps_val = 1 if nasal_polyps == "Yes" else 0

asthma_val = 1 if asthma == "Yes" else 0
high_blood_pressure_val = 1 if high_blood_pressure == "Yes" else 0
severe_headache_val = 1 if severe_headache == "Yes" else 0
weakness_val = 1 if weakness == "Yes" else 0
trouble_seeing_val = 1 if trouble_seeing == "Yes" else 0

fever_val = 1 if fever == "Yes" else 0
body_aches_val = 1 if body_aches == "Yes" else 0
sore_throat_val = 1 if sore_throat == "Yes" else 0
sneezing_val = 1 if sneezing == "Yes" else 0
diarrhea_val = 1 if diarrhea == "Yes" else 0

rapid_breathing_val = 1 if rapid_breathing == "Yes" else 0
rapid_heart_rate_val = 1 if rapid_heart_rate == "Yes" else 0
pain_behind_eyes_val = 1 if pain_behind_eyes == "Yes" else 0
swollen_glands_val = 1 if swollen_glands == "Yes" else 0
rashes_val = 1 if rashes == "Yes" else 0

sinus_headache_val = 1 if sinus_headache == "Yes" else 0
facial_pain_val = 1 if facial_pain == "Yes" else 0
shortness_of_breath_val = 1 if shortness_of_breath == "Yes" else 0
reduced_smell_and_taste_val = 1 if reduced_smell_and_taste == "Yes" else 0
skin_irritation_val = 1 if skin_irritation == "Yes" else 0

itchiness_val = 1 if itchiness == "Yes" else 0
throbbing_headache_val = 1 if throbbing_headache == "Yes" else 0
confusion_val = 1 if confusion == "Yes" else 0
back_pain_val = 1 if back_pain == "Yes" else 0
knee_ache_val = 1 if knee_ache == "Yes" else 0

if st.button("Click Here to Predict The Weather Releted Disease !"):
    features = np.array([age, gender_val, temperature, humidity, wind_speed, nausea_val, joint_pain_val, abdominal_pain_val, high_fever_val,chills_val,
                         fatigue_val, runny_nose_val, pain_behind_the_eyes_val, dizziness_val, headache_val, chest_pain_val, vomiting_val, cough_val,
                         hiv_aids_val, nasal_polyps_val, asthma_val, high_blood_pressure_val, severe_headache_val, weakness_val, trouble_seeing_val,
                         fever_val, body_aches_val, sore_throat_val, sneezing_val, diarrhea_val, rapid_breathing_val, rapid_heart_rate_val,
                         pain_behind_eyes_val, swollen_glands_val, rashes_val, sinus_headache_val, facial_pain_val, shortness_of_breath_val,
                         reduced_smell_and_taste_val, skin_irritation_val, itchiness_val, throbbing_headache_val, confusion_val, back_pain_val, knee_ache_val]).reshape(1, -1)

    predicted_code = model.predict(features)[0]

    disease_map = {
    0: "Arthritis",
    1: "Common Cold",
    2: "Dengue",
    3: "Eczema",
    4: "Heart Attack",
    5: "Heat Stroke",
    6: "Influenza",
    7: "Malaria",
    8: "Migraine",
    9: "Sinusitis",
    10: "Stroke"
    }
    predicted_disease = disease_map.get(predicted_code, "Unknown Disease")
    st.success(f"Predicted Disease Name : {predicted_disease}")
