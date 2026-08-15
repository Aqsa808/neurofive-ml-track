import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("titanic_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)


# Title
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter the passenger details below to predict "
    "whether the passenger would have survived."
)


# Input fields
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)


# Prediction
if st.button("Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("🎉 Prediction: The passenger would likely survive!")
    else:
        st.error("❌ Prediction: The passenger would likely not survive.")