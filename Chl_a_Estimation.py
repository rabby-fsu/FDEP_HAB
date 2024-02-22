# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 03:26:12 2023

@author: 88017
"""

import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


# Define the URL to the CSV file in your GitHub repository
#github_csv_url_A = 'https://github.com/rabby-fsu/Chlorophyll-a_Estimation_Tool/blob/main/ModelA_data.csv'
df_A = pd.read_csv('ModelA_data.csv', sep=",")

# Define the URL to the CSV file in your GitHub repository
#github_csv_url_B = 'https://github.com/rabby-fsu/Chlorophyll-a_Estimation_Tool/blob/main/ModelB_data.csv'
df_B = pd.read_csv('ModelB_data.csv', sep=",")

                 
# Define the Streamlit app pages
pages = ['Home','Model A: Using Physical Chemical Water Quality Parameters', 'Model B: Using Physical Chemical Water Quality and Meteorological Parameters']

# Streamlit app
st.set_page_config(page_title='Chlorophyll-a Estimation Tool for Bay-Estuary', layout='wide')

page = st.selectbox('Select Page', pages)

st.sidebar.title('Chlorophyll-a Estimation Tool for Bay-Estuary')
st.sidebar.write("Welcome to the Chlorophyll-a Estimation Tool for Bay-Estuary. This tool was developed training Machine Learning models with data collected from Apalachicola Bay from 2003-2021. This tool allow users to estimate Chl-a levels (ug/l) in bay estuaries based on physical-chemical water quality parameters with or without meteorological parameters.")
st.sidebar.image("rider.png", use_column_width=True)
st.sidebar.image("famufsu.png", use_column_width=True)
st.sidebar.image("fdep.jpeg", use_column_width=True)


# Model A
# Step 1: Data Preprocessing (Assuming your data is in a DataFrame called 'data_CDEP')
selected_features_1 = ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)', 'Salinity(ppt)','pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year']
X_1 = df_A[selected_features_1]  # Independent variables
y_1 = df_A['Chlorophyll-a (ug/l)']  # Log-transform the target variable        # Target variable

# Perform an 70-30 train-test split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.3, random_state=42)

                 
# Load the pre-trained XGBoost model
xgb_model_Bayesian_01 = XGBRegressor(random_state=42, n_estimators=160, max_depth=4, learning_rate=0.07818940902700418)
xgb_model_Bayesian_01.fit(X_train_1, y_train_1)



#Model B:


# Step 1: Data Preprocessing (Assuming your data is in a DataFrame called 'data_CDEP')
selected_features_2 = ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)', 'Salinity(ppt)','pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year', 'ATemp_max' ,'ATemp_max_1dlag','ATemp_max_2dlag', 'ATemp_max_3dlag', 'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']
X_2 = df_B[selected_features_2]  # Independent variables
y_2 = df_B['Chlorophyll-a (ug/l)']  # target variable

# Split the data into training and test sets
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.3, random_state=42)

# Define the number of iterations for bagging.
num_iterations = 100
decision_trees_mean = []
# Train decision trees using bagging
for iteration in range(num_iterations):
    bootstrap_indices = np.random.choice(len(X_train_2), size=len(X_train_2), replace=True)
    X_train_bootstrap = X_train_2.iloc[bootstrap_indices]
    y_train_bootstrap = y_train_2.iloc[bootstrap_indices]
  
    rf_model_HAB_Grid_02 = RandomForestRegressor(
        random_state=42, 
        max_depth=26,
        min_samples_leaf=1,
        min_samples_split=3,
        n_estimators=1
    )
    rf_model_HAB_Grid_02.fit(X_train_bootstrap, y_train_bootstrap)
    decision_trees_mean.append(rf_model_HAB_Grid_02)


if page == 'Home':    
    st.write("Navigate the pages from the above to work using the following model: ")
    st.write("Model -A (Based one Physical-Chemical Water Quality Parameters: XGBoost Regression with Bayesian Optimization)")
    st.write("Model -B (Based one Physical-Chemical Water Quality and Meteorological Parameters: Decision Tree Regression optimized with Grid Search algorithm)")
    st.write("Please Note: Julian Year = Year + (Days into the Year / Total Days in the Year")

# Model A page
elif page == 'Model A: Using Physical Chemical Water Quality Parameters':
    st.title('Model A: Estimate Chlorophyll-a (ug/l) Using Physical Chemical Water Quality Parameters')
    st.write("Evaluation Metrics: For Test Set, R2=0.64, RMSE=3.043, MAE=2.256, PBIAS=-35.15")
    st.write("The uploaded csv should only contain the following columns in order: ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)', 'Salinity(ppt)','pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year']")
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        expected_feature_names = ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)','Salinity(ppt)', 'pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year']

        # Reorganize the columns to match the expected feature names
        test_data = test_data[expected_feature_names]


        # Make predictions using Model A
        predictions = xgb_model_Bayesian_01.predict(test_data)
        
        # Append predictions as a new column
        test_data['Estimated Chlorophyll-a (ug/l)'] = predictions
        
        # Download the results as a CSV file
        st.write("Download the results as a CSV file.")
        csv = test_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="model_a_predictions.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)


# Model B page
elif page == 'Model B: Using Physical Chemical Water Quality and Meteorological Parameters':
    st.title('Model B: Estimate Chlorophyll-a (ug/l) Using Physical-Chemical Water Quality Parameters and Meteorological Parameters')
    st.write("Model A: Pretrained Random Forest Regression model optimized using Grid Search Algorithm.")
    st.write("Evaluation Metrics: For Test Set, R2=0.64, RMSE=3.037, MAE=2.188, PBIAS=-37.47")
    st.write("The uploaded csv should only contain the following columns in order: ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)','Salinity(ppt)', 'pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year', 'ATemp_max' ,'ATemp_max_1dlag','ATemp_max_2dlag', 'ATemp_max_3dlag', 'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']")
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file)
        expected_feature_names = ['Secchi Depth(m)', 'DO(mg/l)', 'Temperature (deg cels)','Salinity(ppt)', 'pH', 'Turbidity(NTU)', 'Nitrate+Nitrite','Phosphate', 'N/P', 'Julian Year', 'ATemp_max' ,'ATemp_max_1dlag','ATemp_max_2dlag', 'ATemp_max_3dlag', 'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']
        
        # Reorganize the columns to match the expected feature names
        test_data = test_data[expected_feature_names]
      
        # Make predictions using Model B
        predictions = np.mean([tree.predict(test_data) for tree in decision_trees_mean], axis=0)
        
        # Append predictions as a new column
        test_data['Estimated Chlorophyll-a (ug/l)'] = predictions
        
        # Download the results as a CSV file
        st.write("Download the results as a CSV file.")
        csv = test_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
        href = f'<a href="data:file/csv;base64,{b64}" download="model_b_predictions.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

