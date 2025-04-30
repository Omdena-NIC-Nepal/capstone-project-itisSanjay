import streamlit as st
from models import (
    combined_processed_data,
    OHE,
    Standard_Scaler,
    split_data,
    model_train,
    save_model,
)
from data_utils import (
    numerical_feature,
    categorical_feature,
    load_data,
    extract_features_X,
    extract_features_y,
)

def show(df):
    """ Display Model Training """

    st.header("Model Training")

    # Extract features and target
    X = extract_features_X(df)
    y = extract_features_y(df)

    # Set test size from slider (percentage)
    test_percent = st.slider("Test data size (%)", 10, 40, 20)
    test_size = test_percent / 100

    # Get column types
    # Avoid including the target column in feature processing
    numeric_cols = [col for col in numerical_feature(df) if col != 'Migration_label']
    categorical_cols = [col for col in categorical_feature(df) if col != 'Migration_label']

    # Preprocessing
    X_cat_encoded_df = OHE(X, categorical_cols)
    X_num_scaled_df = Standard_Scaler(X, numeric_cols)
    X_processed_df = combined_processed_data(X_cat_encoded_df, X_num_scaled_df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_processed_df, y, test_size)

    st.write(f" Training Data: {len(X_train)} samples")
    st.write(f" Test Data: {len(X_test)} samples")

    # Model selection
    model_type = st.selectbox("Select Model Type", ["logreg", "dt", "rf", "knn", "gb", "svc"])

    # Train model Button
    if st.button("Train model"):
        with st.spinner("Training in Progress..."):
            # Train the model and get evaluation metrics
            model, test_accuracy, cv_scores, report, cm = model_train(model_type, X_train, y_train, X_test, y_test)

            # Save the model
            save_model(model)
            st.success("✅ Model trained and saved successfully")

            # Save in session state
            st.session_state["model"] = model
            st.session_state["model_type"] = model_type

            # Display the evaluation metrics

            # Accuracy Score
            st.subheader("Accuracy Score")
            st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

            # Cross-Validation Score
            st.subheader("Cross-Validation Scores")
            st.write(f"Cross-Validation Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
            st.write(f"Cross-Validation Scores: {cv_scores}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            st.write(cm)

            # Optionally, you can display the classification report
            st.subheader("Classification Report")
            st.write(report)
