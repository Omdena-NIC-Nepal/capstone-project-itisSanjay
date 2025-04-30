import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from data_utils import load_data, numerical_feature, categorical_feature, merged, load_shapefile

def pairplot(df):
    """Generate a seaborn pairplot and return a matplotlib figure"""
    numeric_df = df.select_dtypes(include='number')
    pairgrid = sns.pairplot(numeric_df)
    fig = pairgrid.fig
    return fig

def gis_visualization(merged_df, numerical_features):
    """Visualisation of features in different districts of Nepal"""
    figs = []
    for col in numerical_features:
        fig, ax = plt.subplots(figsize=(12, 12))
        merged_df.plot(column=col, cmap="Blues", legend=True, ax=ax, edgecolor="black")
        ax.set_title(f"{col} by Districts in Nepal")
        ax.axis("off")
        figs.append(fig)
    return figs

def plot_actual_vs_predicted(y_test, y_pred):
    """Actual vs predicted values"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.6, marker='o', color='blue')
    ax.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.6, marker='x', color='red')
    ax.set_title("Actual vs. Predicted (Binary Classification)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Class (0 or 1)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return fig
