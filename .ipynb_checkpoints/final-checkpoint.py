import streamlit as st
import pickle  # For loading saved ML models
import numpy as np




# Load your trained model (update the filename accordingly)
model = pickle.load(open("naive_bayes_model.pkl", "rb"))

st.title("Diabetes Prediction App")

# User input form
preg = st.number_input("Enter number of pregnancies:")
gluc = st.number_input("Enter Glucose:")
BP= st.number_input("Enter :blood pressure")
SkinThickness = st.number_input("Enter SkinThickness :")
Insulin= st.number_input("Enter Insulin:")
BMI = st.number_input("Enter BMI:")
DiabetesPedigreeFunction= st.number_input("Enter DiabetesPedigreeFunction:")
Age = st.number_input("Enter Age :")

# Convert inputs to model format
input_data = np.array([[preg,gluc,BP,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

# Predict on button click
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction)
