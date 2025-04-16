import streamlit as st
import urllib.request
import gzip
import pickle
import os


MODEL_URL ="https://github.com/climate-model/releases/download/v1.0/rf.pkl"
MODEL_FILE = "rf.pkl"

@st.cache_data
def load_model():
    if not os.path.exists(MODEL_FILE):
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    with gzip.open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()
