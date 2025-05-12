import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model(model_dir='model_files'):
    try:
        model_path = os.path.join(model_dir, 'fraud_detection_model.h5')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None
        
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at: {scaler_path}")
            return None, None
        
        # Load files
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.write("Current directory:", os.getcwd())
        st.write("Files in directory:", os.listdir())
        return None, None

def predict_fraud(features, model, scaler):
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probability = prediction[0][0]
    
    return probability

def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("""
    This application predicts whether a credit card transaction is fraudulent based on various features.
    Please enter the transaction details below.
    """)
    
    # Load model
    model, scaler = load_model()
    
    if model is None or scaler is None:
        st.warning("Cannot proceed without model and scaler. Please check the error messages above.")
        return
    
    # Create form for input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=100.0
            )
            latitude = st.number_input(
                "Customer Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.0
            )
            longitude = st.number_input(
                "Customer Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-74.0
            )
            city_pop = st.number_input(
                "City Population",
                min_value=0,
                value=100000
            )
        
        with col2:
            unix_time = st.number_input(
                "Unix Timestamp",
                min_value=0,
                value=1577836800
            )
            merch_lat = st.number_input(
                "Merchant Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.0
            )
            merch_long = st.number_input(
                "Merchant Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-74.0
            )
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create features dataframe
            features = pd.DataFrame({
                'amt': [amount],
                'lat': [latitude],
                'long': [longitude],
                'city_pop': [city_pop],
                'unix_time': [unix_time],
                'merch_lat': [merch_lat],
                'merch_long': [merch_long]
            })
            
            probability = predict_fraud(features, model, scaler)
            
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Fraud Probability",
                    value=f"{probability:.2%}"
                )
            
            with col2:
                prediction = "🚨 FRAUDULENT" if probability > 0.3 else "✅ LEGITIMATE"
                st.metric(
                    label="Transaction Status",
                    value=prediction
                )
            
            with col3:
                risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                st.metric(
                    label="Risk Level",
                    value=risk_level
                )
            
            st.markdown("### Transaction Details")
            st.dataframe(features)

if __name__ == "__main__":
    main()