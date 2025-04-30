import streamlit as st
# set the page configuration
st.set_page_config(
    page_title=" Migration Risk Predictor",
    page_icon="🌍",
    layout="wide")
from data_utils import load_data
import sys

sys.path.append("C:/Users/Sanjay Sah/Desktop/sanjay_sah_git/capstone-project-itisSanjay/pages")
from pages import data_exploration,model_training,prediction_page


# Title
st.title("Assessing Migration Risk in Nepal's 77 Districts")
st.markdown("Analysis of Cliamte data, Environment data and socio-economic data(census 2021), to predict migration-risk")


df = load_data()


# Sidebar for the App Navigation
st.sidebar.title("Navigation")
page=st.sidebar.radio("Go TO",["Data Exploration","Model Training","Prediction"])

# Display the selected page
if page=="Data Exploration":
  data_exploration.show(df)
elif page=="Model Training":
  model_training.show(df)
 
else: # prediction Page
 prediction_page.show(df)



