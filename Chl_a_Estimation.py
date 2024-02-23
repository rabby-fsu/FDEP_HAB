import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk
import folium


# Load data
df = pd.read_csv('DataFile_ML_All.csv')

# Define selected features
selected_features = ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                     'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                     'ATemp_max_7dlag']

X = df[selected_features]
y = df['Chlorophyll-a (ug/L)']


# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Function to create colored markers based on chlorophyll-a concentration
def color_marker(chl_a):
    if chl_a <= 10:
        return 'green'  # No bloom
    else:
        return 'red'  # Bloom

# Function to create Folium map
def create_map(selected_year, selected_month):
    # Filter data for the selected year and month
    filtered_df = df[(df['Date'].dt.year == selected_year) & (df['Date'].dt.month == selected_month)]

    # Create map centered at mean latitude and longitude
    map_data = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=10)

    # Add markers for each data point
    for index, row in filtered_df.iterrows():
        folium.Marker([row['lat'], row['lon']],
                      popup=f"Chlorophyll-a (ug/L): {row['Chlorophyll-a (ug/L)']}",
                      icon=folium.Icon(color=color_marker(row['Chlorophyll-a (ug/L)']))).add_to(map_data)

    # Display the map
    return map_data


# Perform train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
# Initialize and fit the XGBoost Regressor
#xgb_regressor = XGBRegressor(n_estimators=334, max_depth=4, learning_rate=0.07818940902700418, random_state=42)
#xgb_regressor.fit(X_train, y_train)
    

#Introduction Page
st.sidebar.title('Pages')
selected_page = st.sidebar.radio('Go to', ['Introduction', 'Apalachicola Bay-Estuary', 'Pensacola-Perdido Bay-Estuary'])

if selected_page == 'Introduction':
    st.title('Introduction')
    st.write('This is an application to evaluate the Apalachicola Bay Model.')

elif selected_page == 'Apalachicola Bay-Estuary':
    
    st.title('Gauged Stations')
    st.map(df,latitude='Latitude',longitude='Longitude',use_container_width=True)


    # Button to evaluate the model
    if st.button('Evaluate Model'):
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
    
elif selected_page == 'Pensacola-Perdido Bay-Estuary':
    st.title('Spatial Distribution of Chlorophyll-a Concentrations')

    # Set default values for year and month based on the data range
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    default_date = min_date + (max_date - min_date) // 2
    default_year = default_date.year
    default_month = default_date.month

    # Controls for year and month selection
    selected_year = st.slider('Select Year', min_value=min_date.year, max_value=max_date.year, value=default_year)
    selected_month = st.slider('Select Month', min_value=1, max_value=12, value=default_month)

    # Create map based on selected year and month
    map_data = create_map(selected_year, selected_month)

    # Add tile layer to the map
    folium.TileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', attr='OpenStreetMap').add_to(map_data)

    # Display the map
    st.write(map_data._repr_html_(), unsafe_allow_html=True)
