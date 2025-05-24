import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import geopandas as gpd

@st.cache_data
def load_data():
    """Loading the CSV data"""
    df = pd.read_csv(r"data.csv")
    return df

def load_shapefile():
    """Loading the shapefile"""
    gdf = gpd.read_file(r"District.shp")
    return gdf

def extract_features_X(df):
    """Extracting features and targets for model training"""
    X = df.drop('Migration_label', axis=1)
    
    return X
def extract_features_y(df):
    """Extracting features and targets for model training"""
    
    y = df['Migration_label']
    return y

def numerical_feature(df):
    """Separating numerical columns"""
    numerical_features = df.select_dtypes(include='number').columns
    return numerical_features

def categorical_feature(df):
    """Separating categorical columns"""
    categorical_features = df.select_dtypes(include='object').columns
    return categorical_features

def merged(df, gdf):
    """Merging the df and gdf on District"""
    gdf["District"] = gdf["District"].str.strip().str.lower()
    df["District"] = df["District"].str.strip().str.lower()
    merged = gdf.merge(df, on="District")
    return merged
