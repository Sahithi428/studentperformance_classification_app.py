import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Grade Prediction")
st.write(
    "Predict the student's Grade Class using a trained "
    "Random Forest Classification model."
)

st.divider()


# -----------------------------
# Student input
# -----------------------------
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=30,
        value=17
    )

    gender = st.selectbox(
        "Gender",
        [0, 1]
    )

    ethnicity = st.number_input(
        "Ethnicity",
        min_value=0,
        max_value=10,
        value=0
    )

    parental_education = st.number_input(
        "Parental Education",
        min_value=0,
        max_value=10,
        value=0
    )

    study_time = st.number_input(
        "Study Time Weekly",
        min_value=0.0,
        max_value=50.0,
        value=10.0
    )


with col2:

    absences = st.number_input(
        "Absences",
        min_value=0,
        max_value=100,
        value=5
    )

    tutoring = st.selectbox(
        "Tutoring",
        [0, 1]
    )

    parental_support = st.number_input(
        "Parental Support",
        min_value=0,
        max_value=10,
        value=2
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        [0, 1]
    )

    sports = st.selectbox(
        "Sports",
        [0, 1]
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Grade", use_container_width=True):

    input_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender],

        "Ethnicity": [ethnicity],

        "ParentalEducation": [parental_education],

        "StudyTimeWeekly": [study_time],

        "Absences": [absences],

        "Tutoring": [tutoring],

        "ParentalSupport": [parental_support],

        "Extracurricular": [extracurricular],

        "Sports": [sports]
    })


    # Apply the same scaler used during training
    input_scaled = scaler.transform(input_data)


    # Make prediction
    prediction = model.predict(input_scaled)


    # Convert prediction to integer
    grade_class = int(prediction[0])


    # Grade mapping
    grade_mapping = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E"
    }


    predicted_grade = grade_mapping.get(
        grade_class,
        "Unknown"
    )


    # Display result
    st.success(
        f"🎓 Predicted Grade Class: **{predicted_grade}**"
    )