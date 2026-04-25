import pandas as pd
import joblib
import streamlit as st

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("Falcon 9 Landing Prediction")

# Only 5 inputs shown
flight_number = st.number_input("Flight Number", 1)
payload_mass = st.number_input("Payload Mass (kg)", 0.0)
grid_fins = st.selectbox("Grid Fins", [0, 1])
legs = st.selectbox("Legs", [0, 1])
reused_count = st.number_input("Reused Count", 0)

def create_full_input():
    row = pd.DataFrame(0, index=[0], columns=features)

    # fill only visible features
    fill_map = {
        "FlightNumber": flight_number,
        "PayloadMass": payload_mass,
        "GridFins": grid_fins,
        "Legs": legs,
        "ReusedCount": reused_count
    }

    for k, v in fill_map.items():
        if k in row.columns:
            row[k] = v

    return row

if st.button("Predict Landing Outcome"):

    input_df = create_full_input()
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.success("Landing Successful")
    else:
        st.error("Landing Failed")