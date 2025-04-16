import streamlit as st
import urllib.request
import gzip
import pickle
import os


MODEL_URL ="https://github.com/Spoiler-xl/climate-model/releases/download/V1.0/rf.pkl"
MODEL_FILE = "rf.pkl"

@st.cache_data
def load_model():
    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()
# --- UI Inputs for 5 Selected Features ---
feature1 = st.number_input("Feature 1 (e.g., humidity)")
feature2 = st.number_input("Feature 2 (e.g., dew point)")
feature3 = st.number_input("Feature 3 (e.g., wind speed)")
feature4 = st.number_input("Feature 4 (e.g., solar radiation)")
feature5 = st.number_input("Feature 5 (e.g., pressure)")

# --- Predict Button ---
if st.button("🔮 Predict Temperature"):
    input_data = np.array([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(input_data)
    st.success(f"🌡️ Predicted Temperature: {prediction[0]:.2f}°C")
