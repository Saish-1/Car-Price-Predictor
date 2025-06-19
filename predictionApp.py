import streamlit as st
import numpy as np
import joblib

# Load the scaler and model
scaler = joblib.load('Scaler.pkl')
model = joblib.load('Model.pkl')

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("Car Price Prediction App")
st.write("Enter the car specifications below to estimate its price.")

# Input fields
age = st.number_input("Age of the Car (in months)", min_value=1, max_value=80, step=1)
km = st.number_input("Kilometers Driven", min_value=1, step=1)
fuel_type = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol"])
hp = st.number_input("Horsepower", min_value=50, max_value=200, step=1)
automatic = st.selectbox("Automatic Transmission", ["No", "Yes"])
cc = st.number_input("Engine CC", min_value=1000, max_value=20000, step=100)
doors = st.selectbox("Number of Doors", [2, 3, 4, 5])
gears = st.selectbox("Number of Gears", [3, 4, 5, 6])
weight = st.number_input("Car Weight (kg)", min_value=900, step=10)

# Encode categorical variables
fuel_map = {"CNG": 0, "Diesel": 1, "Petrol": 2}
fuel_encoded = fuel_map[fuel_type]
automatic_encoded = 1 if automatic == "Yes" else 0

# Prepare and scale data
numeric_features = np.array([[age, km, cc, hp, weight]])
scaled_features = scaler.transform(numeric_features)

X = [
    scaled_features[0][0],  # Age
    scaled_features[0][1],  # KM
    fuel_encoded,           # Fuel_Type
    scaled_features[0][3],  # HP
    automatic_encoded,      # Automatic
    scaled_features[0][2],  # cc
    doors,                  # Doors
    gears,                  # Gears
    scaled_features[0][4],  # Weight
]

# Prediction button
if st.button("Predict Price"):
    input_data = np.array(X).reshape(1, -1)
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")
else:
    st.write('Enter the featurs of the Car and press the Predict button.')