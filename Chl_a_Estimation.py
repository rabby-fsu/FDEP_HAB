# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 03:26:12 2023

@author: 88017
"""

import streamlit as st
import pandas as pd
import requests
import pickle
from io import BytesIO
import base64

# Define the URLs of the models on GitHub
#model_A_url = 'https://github.com/rabby-fsu/Chlorophyll-a_Estimation_Tool/blob/Model-A/xgb_model_Bayesian_01.pkl'  # Replace with the URL for Model A
#model_B_url = 'https://github.com/rabby-fsu/Chlorophyll-a_Estimation_Tool/blob/Model-B/rf_model_HAB_Grid_02.pkl'  # Replace with the URL for Model B

model_A_url = 'xgb_model_Bayesian_01.pkl'  
model_B_url = 'rf_model_HAB_Grid_02.pkl'  

# Load the pre-trained models locally
with open(model_A_url, "rb") as model_file_A:
    model_A = pickle.load(model_file_A)

with open(model_B_url, "rb") as model_file_B:
    model_B = pickle.load(model_file_B)
    





# Load the pre-trained models locally
#with open("xgb_model_Bayesian_01.pkl", "rb") as model_file_A:
    #model_A = pickle.load(model_file_A)

#model_B_path = 'C:/Users/88017/rf_model_HAB_Grid_02.pkl'
#with open(model_B_path, "rb") as model_file_B:



# Function to load a model from GitHub
#def load_model_from_github(url):
    #response = requests.get(url)
    #model = pickle.load(BytesIO(response.content))
    #return model

# Define the Streamlit app pages
pages = ['Home','Model A: Using Physical Chemical Water Quality Parameters', 'Model B: Using Physical Chemical Water Quality and Meteorological Parameters']

# Streamlit app
st.set_page_config(page_title='Chlorophyll-a Estimation Tool for Bay-Estuary', layout='wide')

page = st.selectbox('Select Page', pages)

st.sidebar.title('Chlorophyll-a Estimation Tool for Bay-Estuary')
st.sidebar.write("Welcome to the Chlorophyll-a Estimation Tool. This tool allows you to upload a CSV file containing some specific variables, make predictions on chlorophyll-a concentration in Apalachicola Bay using pretrained machine learning models, and download the results as a CSV file.")
st.sidebar.image("rider.png", use_column_width=True)
st.sidebar.image("famufsu.png", use_column_width=True)
st.sidebar.image("fdep.jpeg", use_column_width=True)





# Model A page
if page == 'Model A: Using Physical Chemical Water Quality Parameters':
    st.title('Estimate Chlorophyll-a (ug/l) Using Physical Chemical Water Quality Parameters')
    st.write("Model A: Pretrained XGBoost Regression model optimized using Bayesian Optimization Algorithm")
    st.write("Evaluation Metrics: For Test Set, R2=0.64, RMSE=3.043, MAE=2.256, PBIAS=-35.15")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        expected_feature_names = ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)','Salinity(ppt)', 'pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year']

        # Reorganize the columns to match the expected feature names
        test_data = test_data[expected_feature_names]

        
        # Load Model A from GitHub
        #model_A = load_model_from_github(model_A_url)
        
        # Make predictions using Model A
        predictions = model_A.predict(test_data)
        
        # Append predictions as a new column
        test_data['Model A Predictions'] = predictions
        
        # Download the results as a CSV file
        st.write("Download the results as a CSV file.")
        csv = test_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="model_a_predictions.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# Model B page
elif page == 'Model B: Using Physical Chemical Water Quality and Meteorological Parameters':
    st.title('Estimate Chlorophyll-a (ug/l) Using Physical-Chemical Water Quality Parameters and Meteorological Parameters')
    st.write("Model A: Pretrained Random Forest Regression model optimized using Grid Search Algorithm.")
    st.write("Evaluation Metrics: For Test Set, R2=0.64, RMSE=3.037, MAE=2.188, PBIAS=-37.47")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        
        # Load Model B from GitHub
        #model_B = load_model_from_github(model_B_url)
        
        # Make predictions using Model B
        predictions = model_B.predict(test_data)
        
        # Append predictions as a new column
        test_data['Model B Predictions'] = predictions
        
        # Download the results as a CSV file
        st.write("Download the results as a CSV file.")
        csv = test_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="model_b_predictions.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

