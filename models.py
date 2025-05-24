import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_utils import load_data, numerical_feature, categorical_feature

# Load data
df = load_data()

# Extract features and target
X = df.drop("Migration_label", axis=1)
y = df["Migration_label"]

# Extract numerical and categorical column names
numeric_cols = numerical_feature(df)
categorical_cols = categorical_feature(df)

# One-Hot Encoding
def OHE(X, categorical_cols):
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = ohe.fit_transform(X[categorical_cols])
    encoded_feature_names = ohe.get_feature_names_out(categorical_cols)
    X_cat_encoded_df = pd.DataFrame(X_cat_encoded, columns=encoded_feature_names, index=X.index)
    return X_cat_encoded_df

# Standard Scaling
def Standard_Scaler(X, numeric_cols):
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[numeric_cols])
    X_num_scaled_df = pd.DataFrame(X_num_scaled, columns=numeric_cols, index=X.index)
    return X_num_scaled_df

# Combine processed features
def combined_processed_data(X_cat_encoded_df, X_num_scaled_df):
    X_processed_df = pd.concat([X_num_scaled_df, X_cat_encoded_df], axis=1)
    return X_processed_df

# Split data
def split_data(X_processed_df, y, test_size=0.2, random_state=42):
    return train_test_split(X_processed_df, y, test_size=test_size, random_state=random_state, stratify=y)

# Model training
def model_train(model_type, X_train, y_train, X_test, y_test):
    if model_type == "logreg":
        model = LogisticRegression(solver='liblinear', max_iter=1000)
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'class_weight': [None, 'balanced']
        }
    elif model_type == "dt":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }
    elif model_type == "rf":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }
    elif model_type == "knn":
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    elif model_type == "gb":
        model = GradientBoostingClassifier()
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_type == "svc":
        model = SVC()
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        }
    else:
        raise ValueError("Invalid model_type. Choose from: 'logreg', 'dt', 'rf', 'knn', 'gb', 'svc'.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Evaluation metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)


    return best_model, test_accuracy, cv_scores, report, cm

# Save model
def save_model(best_model, filename='trained_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"✅ Model saved as '{filename}'")

# Load model
def load_model(filename='trained_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)


