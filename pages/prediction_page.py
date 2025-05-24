import streamlit as st
import pandas as pd
import pickle
from models import (
    OHE,
    Standard_Scaler,
    combined_processed_data,
    load_model,
)
from data_utils import (
    numerical_feature,
    categorical_feature,
    extract_features_X,
)

def show(df):
    """ Display Prediction Page """

    st.header("Prediction")

    # Load trained model (if available)
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Extract features for prediction (exclude target if present)
    X = extract_features_X(df)
    numeric_cols = [col for col in numerical_feature(df) if col != 'Migration_label']
    categorical_cols = [col for col in categorical_feature(df) if col != 'Migration_label']

    # Preprocessing same as training
    X_cat_encoded_df = OHE(X, categorical_cols)
    X_num_scaled_df = Standard_Scaler(X, numeric_cols)
    X_processed_df = combined_processed_data(X_cat_encoded_df, X_num_scaled_df)

    # Show raw data preview
    if st.checkbox("Show raw input data"):
        st.write(X)

    # Predict Button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            predictions = model.predict(X_processed_df)
            st.subheader("Predictions")
            st.write(predictions)
