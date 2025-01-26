import streamlit as st
import joblib
import pickle
import numpy as np

# Configure the Streamlit page
st.set_page_config(page_title="My Clustering Test")
st.title("Mall Customer Classification")

# Load the models
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as file:
        kmeans_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return kmeans_model, scaler

kmeans_model, scaler = load_models()

# Input fields for the user
income = st.text_input("Annual Income:", "")
spend_score = st.text_input("Spending Score (1-100):", "")

# Button to trigger clustering
if st.button("Cluster Customer"):
    try:
        # Convert inputs to numeric values
        income = float(income)
        spend_score = float(spend_score)
        
        # Scale the new data
        new_data_scaled = scaler.transform([[income, spend_score]])
        
        # Predict the cluster
        cluster = kmeans_model.predict(new_data_scaled)
        
        # Display the result
        st.success(f"The new data point belongs to cluster: {cluster[0]}")
    except ValueError:
        st.error("Please enter valid numeric values for both inputs.")
