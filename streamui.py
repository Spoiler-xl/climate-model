import streamlit as st
import urllib.request
import gzip
import pickle
import numpy as np  
import os

# --- Model Download URL ---
MODEL_URL = "https://github.com/Spoiler-xl/climate-model/releases/download/V1.0/rf.pkl.gz"
MODEL_FILE = "rf.pkl.gz"

# --- Load Model Function ---
@st.cache_data
def load_model():
    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    with gzip.open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
dew_point = st.number_input("🌫️ Dew Point (°C)", min_value=-30.0, max_value=50.0, value=10.0)
wind_speed = st.number_input("🌬️ Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=3.0)
solar_radiation = st.number_input("☀️ Solar Radiation (W/m²)", min_value=0.0, max_value=1200.0, value=500.0)
pressure = st.number_input("📊 Pressure (hPa)", min_value=800.0, max_value=1100.0, value=1010.0)

# --- Ensure proper shape for model prediction ---
if st.button("🔮 Predict Temperature"):
    try:
        input_data = np.array([[humidity, dew_point, wind_speed, solar_radiation, pressure]])

        # Check the shape of input_data
        if input_data.shape[1] != 5:
            st.error(f"Error: The input data should have 5 features. Current shape: {input_data.shape}")
        else:
            prediction = model.predict(input_data)
            st.success(f"🌡️ Predicted Temperature: **{prediction[0]:.2f}°C**")
    except Exception as e:
        st.error(f"An error occurred: {e}")
