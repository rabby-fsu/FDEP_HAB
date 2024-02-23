import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk


# Introduction Page
st.sidebar.title('Pages')
selected_page = st.sidebar.radio('Go to', ['Introduction', 'Apalachicola Bay-Estuary', 'Pensacola-Perdido Bay-Estuary'])

if selected_page == 'Introduction':
    st.title('Introduction')
    st.write('This is an application to evaluate the Apalachicola Bay Model.')

elif selected_page == 'Apalachicola Bay-Estuary':
    # Load data
    df = pd.read_csv('DataFile_ML_All.csv')

    # Define selected features
    selected_features = ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                     'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                     'ATemp_max_7dlag']
    
    st.title('Gauged Stations')
    st.map(df,latitude='Latitude',longitude='Longitude',use_container_width=True)


    # Create combined training data by randomly selecting 80% data from each station
    combined_training_data = pd.DataFrame(columns=df.columns)
    combined_testing_data = pd.DataFrame(columns=df.columns)
    station_samples = []
    for station in df['station_code'].unique():
        station_data = df[df['station_code'] == station]
        train_data, test_data = train_test_split(station_data, test_size=0.2, random_state=42)
        combined_training_data = pd.concat([combined_training_data, train_data])
        combined_testing_data = pd.concat([combined_testing_data, test_data])
        # Store the number of samples for this station in the list
        station_samples.append({'Station': station, 'Training Samples': len(train_data), 'Testing Samples': len(test_data)})
    st.write("## Summary of the Dataset")
    st.table(station_samples)  

    # Use combined_training_data for the following model
    X_train = combined_training_data[selected_features]
    y_train = combined_training_data['Chlorophyll-a (ug/L)']
    X_test = combined_testing_data[selected_features]
    y_test = combined_testing_data['Chlorophyll-a (ug/L)']
    
    # Initialize and fit the XGBoost Regressor
    xgb_regressor = XGBRegressor(n_estimators=334, max_depth=4, learning_rate=0.07818940902700418, random_state=42)
    xgb_regressor.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = xgb_regressor.predict(X_train)
    y_test_pred = xgb_regressor.predict(X_test)
    
    # Evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Display R2 and RMSE for training data
    st.write("## Training Data Metrics")
    st.write(f"Training R^2 Score: {train_r2}")
    st.write(f"Training RMSE: {train_rmse}")
    # Display R2 and RMSE for testing data
    st.write("## Test Data Metrics")
    st.write(f"Test R^2 Score: {test_r2}")
    st.write(f"Test RMSE: {test_rmse}")
    
    # Initialize lists to store R2 and RMSE for each station's test data
    test_r2_scores = []
    test_rmse_scores = []

    # Evaluate model on each station's test data
    for station in df['station_code'].unique():
        station_test_data = combined_testing_data[combined_testing_data['station_code'] == station]
        X_test = station_test_data[selected_features]
        y_test = station_test_data['Chlorophyll-a (ug/L)']
        
        # Predictions on test data
        y_test_pred = xgb_regressor.predict(X_test)

        # Evaluation metrics for test data
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Append scores to lists
        test_r2_scores.append(test_r2)
        test_rmse_scores.append(test_rmse)

    # Create DataFrame for test data metrics
    test_metrics_df = pd.DataFrame({
        'Station': df['station_code'].unique(),
        'Test R^2 Score': test_r2_scores,
        'Test RMSE': test_rmse_scores
    })

    # Display test data metrics table
    st.write("## Test Data Metrics")
    st.table(test_metrics_df)
