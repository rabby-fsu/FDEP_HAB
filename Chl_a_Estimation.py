import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import streamlit as st
import pydeck as pdk

import matplotlib

# Load data
df = pd.read_csv('DataFile_ML_All.csv')

# Define selected features
selected_features = ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                     'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                     'ATemp_max_7dlag']


# Function to evaluate model per station
def training(combined_training_data):
    X_train = combined_training_data[selected_features]
    y_train = combined_training_data['Chlorophyll-a (ug/L)']
    
    # Initialize and fit the XGBoost Regressor
    xgb_regressor = XGBRegressor(n_estimators=334, max_depth=4, learning_rate=0.07818940902700418, random_state=42)
    xgb_regressor.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = xgb_regressor.predict(X_train)
    
    # Evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    return train_r2, train_rmse,




# Introduction Page
st.sidebar.title('Pages')
selected_page = st.sidebar.radio('Go to', ['Introduction', 'Apalachicola Bay-Estuary', 'Pensacola-Perdido Bay-Estuary'])

if selected_page == 'Introduction':
    st.title('Introduction')
    st.write('This is an application to evaluate the Apalachicola Bay Model.')

elif selected_page == 'Apalachicola Bay-Estuary':
    
    st.title('Gauged Stations')
    st.map(df,
    latitude='Latitude',
    longitude='Longitude',
    use_container_width=True)
      
    st.title('Evaluate the Apalachicola Bay Model')
    # Create combined training data by randomly selecting 80% data from each station
    combined_training_data = pd.DataFrame(columns=df.columns)
    for station in df['station_code'].unique():
        station_data = df[df['station_code'] == station]
        train_data, test_data = train_test_split(station_data, test_size=0.2, random_state=42)
    combined_training_data = pd.concat([combined_training_data, train_data])
    combined_testing_data = pd.concat([combined_training_data, test_data])

    
    # Button to evaluate the model
    #if st.button('Evaluate Model'):

