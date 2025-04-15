import streamlit as st
import pandas as pd
import numpy as np
import pickle  

from scikitlearn.linear_model import LinearRegression

# Load your trained model (assumes you saved it as 'model.pkl')
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🌦️ Climate Data Prediction App")
st.write("Enter the climate parameters to get the predicted target value.")

# Example input features — replace these with your actual feature names
# and customize the ranges or defaults appropriately
feature_1 = st.number_input("Feature 1 (e.g. Temperature)", value=25.0)
feature_2 = st.number_input("Feature 2 (e.g. Humidity)", value=60.0)
feature_3 = st.number_input("Feature 3 (e.g. Wind Speed)", value=5.0)
feature_4 = st.number_input("Feature 4 (e.g. Hour of Day)", value=12)

# Add more features as needed

# Prediction button
if st.button("Predict"):
    # Prepare input array
    input_features = np.array([[feature_1, feature_2, feature_3, feature_4]])
    
    # Get prediction
    prediction = model.predict(input_features)

    # Display prediction
    st.success(f"📈 Predicted Value: {prediction[0]:.2f}")

# Optionally display model performance metrics
st.sidebar.header("Model Performance")
st.sidebar.write("R² Score: 0.99993")
st.sidebar.write("Mean Squared Error: 0.00846")

st.sidebar.write("Cross-validated R²: 0.99994")
st.sidebar.write("Cross-validated MSE: 0.00662")
