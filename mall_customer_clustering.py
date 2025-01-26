import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title = "My Clustering Test")
st.title("Mall Customer Classification")
@st.cache(allow_output_mutation = True)

with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# New data
income = st.text_input("Annual Income:", "")
spend_score = st.text_input("Spending Score (1-100):", "")  # Annual Income = 50, Spending Score = 60
if st.button("Cluster Customer"):
    new_data_scaled = scaler.transform([[income,spend_score]])
    cluster = kmeans_model.predict(new_data_scaled)
    print(f'The new data point belongs to cluster: {cluster[0]}')