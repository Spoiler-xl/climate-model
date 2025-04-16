import streamlit as st
import urllib.request
import joblib
import numpy as np
import os

# --- Model Download URL ---
MODEL_URL = "https://github.com/Spoiler-xl/climate-model/releases/download/V1.0/rf_compressed.pkl"
MODEL_FILE = "rf_compressed.pkl"

# --- Load Model Function ---
@st.cache_data
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.info("Model not found locally. Downloading...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

    try:
        model = joblib.load(MODEL_FILE)
        st.success("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

# --- Check model type ---
if model is not None:
    st.write(f"Model Type: {type(model)}")
    if hasattr(model, 'predict'):
        st.write("Model is valid and has a 'predict' method.")
    else:
        st.error("The loaded model does not have a 'predict' method.")
else:
    st.stop()

# --- Streamlit App UI ---
st.title("🌦️ Climate Temperature Predictor")
st.markdown("Enter climate parameters to predict the temperature (°C).")

# --- Input fields for selected features ---
humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 50.0)
dew_point = st.number_input("🌫️ Dew Point (°C)", -30.0, 50.0, 10.0)
wind_speed = st.number_input("🌬️ Wind Speed (m/s)", 0.0, 30.0, 3.0)
solar_radiation = st.number_input("☀️ Solar Radiation (W/m²)", 0.0, 1200.0, 500.0)
pressure = st.number_input("📊 Pressure (hPa)", 800.0, 1100.0, 1010.0)

# --- Prediction ---
if st.button("🔮 Predict Temperature"):
    try:
        input_data = np.array([[humidity, dew_point, wind_speed, solar_radiation, pressure]])
        if input_data.shape[1] != 5:
            st.error(f"Error: Input shape is incorrect: {input_data.shape}")
        else:
            prediction = model.predict(input_data)
            st.success(f"🌡️ Predicted Temperature: **{prediction[0]:.2f}°C**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
