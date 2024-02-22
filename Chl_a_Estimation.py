import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import streamlit as st

# Load data
df = pd.read_csv('DataFile_ML_All.csv')

# Define selected features
selected_features = ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                     'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                     'ATemp_max_7dlag']

# Create combined training data by randomly selecting 80% data from each station
combined_training_data = pd.DataFrame(columns=df.columns)
for station in df['station_code'].unique():
    station_data = df[df['station_code'] == station]
    train_data, test_data = train_test_split(station_data, test_size=0.2, random_state=42)
    combined_training_data = pd.concat([combined_training_data, train_data])
    combined_testing_data = pd.concat([combined_training_data, test_data])

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


# Function to update map with evaluation results
def update_map_with_evaluation_results():
    # Create combined training data by randomly selecting 80% data from each station
    combined_training_data = pd.DataFrame(columns=df.columns)
    for station in df['station_code'].unique():
        station_data = df[df['station_code'] == station]
        train_data, _ = train_test_split(station_data, test_size=0.2, random_state=42)
        combined_training_data = pd.concat([combined_training_data, train_data])
    
    # Display map with gauged stations
    st.map(df, latitude='Latitude', longitude='Longitude', use_container_width=True)

    # Iterate through each station
    for station in df['station_code'].unique():
        st.write(f"Evaluating Station {station}")
        
        # Filter data for the current station
        station_data = df[df['station_code'] == station]
        
        # Evaluate model for the current station
        train_r2, test_r2, train_rmse, test_rmse = evaluate_model_per_station(station_data, combined_training_data)
        
        # Display evaluation results
        st.write(f"Station {station} - Prediction Accuracy:")
        st.write(f"Training R2: {train_r2}")
        st.write(f"Testing R2: {test_r2}")
        st.write(f"Training RMSE: {train_rmse}")
        st.write(f"Testing RMSE: {test_rmse}")


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
    size='Predicted')
    st.title('Evaluate the Apalachicola Bay Model')
    
    # Button to evaluate the model
    if st.button('Evaluate Model'):
        # Update map with evaluation results
        update_map_with_evaluation_results()

elif selected_page == 'Ungauged Stations':
    st.title('Show Ungauged Stations')
    st.write('This page is currently under development.')

