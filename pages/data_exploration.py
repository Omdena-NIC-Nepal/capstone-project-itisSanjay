import streamlit as st
import sys
sys.path.append("C:/Users/Sanjay Sah/Desktop/sanjay_sah_git/capstone-project-itisSanjay/visualization.py")
from data_utils import merged
from data_utils import numerical_feature
from data_utils import categorical_feature
from data_utils import load_shapefile,load_data
from visualization import pairplot, gis_visualization
def show(df):
    """Dislpay the data exploration page"""


    st.header("Data Exploration")
    
    # show the raw data:
    st.subheader("Climate,Environment and Socio-economic Data")
    st.dataframe(df)
    
    # Basic Statistics
    st.subheader("statistical summary")
    st.write(df.describe())

    # Pairplot
    st.subheader("Pairplot of Variables")
    fig=pairplot(df)
    st.pyplot(fig)
    # show the shapefile
    st.subheader("Polygon geometry of 77 districts of Nepal")
    gdf=load_shapefile()
    st.dataframe(gdf)

    # GIS Visualization
    st.subheader("Visualization of districts and Features")
    gdf=load_shapefile()
    merged_df=merged(df,gdf)
    num_features=numerical_feature(df)

    for fig in gis_visualization(merged_df,num_features):
        st.pyplot(fig)
