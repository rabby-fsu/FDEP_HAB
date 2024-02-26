import pandas as pd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import streamlit as st
import threading
from queue import Queue
import pydeck as pdk
import folium
import cartopy.crs as ccrs
from collections import defaultdict
import matplotlib.colors as mcolors
import requests
from io import BytesIO
import base64
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error




# Define a list of dictionaries for each case
cases = [
    {
        'name': 'Apalachicola',
        'data_file': 'Apalachicola.csv',
        'threshold': 5,
        'selected_features': ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'ATemp_max',
                              'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                              'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                              'ATemp_max_7dlag'],
        'model': XGBRegressor(n_estimators=335, max_depth=4, learning_rate=0.037818940902700418, random_state=42),
        'test_size': 0.2
    },
    {
        'name': 'Joseph',
        'data_file': 'Joseph.csv',
        'threshold': 5,
        'selected_features': ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                              'Nitrogen, Kjeldahl (mg/L)', 'ATemp_max_1dlag', 'ATemp_max_2dlag',
                              'ATemp_max_3dlag', 'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                              'ATemp_max_7dlag'],
        'model': RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42),
        'test_size': 0.2
    },
    {
        'name': 'Andrew',
        'data_file': 'Andrew.csv',
        'threshold': 10,
        'selected_features': ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                              'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                              'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag'],
        'model': XGBRegressor(n_estimators=11, max_depth=5, random_state=42),
        'test_size' : 0.2
    },
    {
        'name': 'Pensacola-Perdido',
        'data_files': 'pensacola_perdido.csv',
        'threshold': 10,
        'selected_features': ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                              'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                              'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag'],
        'model': RandomForestRegressor(n_estimators=10, random_state=42),
        'test_size' : 0.3
    }
]

# Define a function for model evaluation
def evaluate_model(trained_model, X_train, X_test, y_train, y_test):
    y_train_pred = trained_model.predict(X_train)
    y_test_pred = trained_model.predict(X_test)

    # Calculate evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Plot Actual vs Predicted Chlorophyll-a for train and test datasets
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Train dataset plot
    sns.scatterplot(x=y_train, y=y_train_pred, ax=axes[0])
    axes[0].set_title('Actual vs. Predicted Chlorophyll-a (Train)')
    axes[0].set_xlabel('Actual Chlorophyll-a (ug/L)')
    axes[0].set_ylabel('Predicted Chlorophyll-a (ug/L)')
    axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    axes[0].text(0.1, 0.9, f'R2: {train_r2:.2f}\nRMSE: {train_rmse:.2f}', transform=axes[0].transAxes)

    # Test dataset plot
    sns.scatterplot(x=y_test, y=y_test_pred, ax=axes[1])
    axes[1].set_title('Actual vs. Predicted Chlorophyll-a (Test)')
    axes[1].set_xlabel('Actual Chlorophyll-a (ug/L)')
    axes[1].set_ylabel('Predicted Chlorophyll-a (ug/L)')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    axes[1].text(0.1, 0.9, f'R2: {test_r2:.2f}\nRMSE: {test_rmse:.2f}', transform=axes[1].transAxes)

    # Display the plots
    st.pyplot(fig)

# Function to process each case
def process_case(case):
    # Read data
    if 'data_file' in case:
        df = pd.read_csv(case['data_file'])
    else:
        df = pd.concat([pd.read_csv(file) for file in case['data_files']], ignore_index=True)
    
    # Drop NaNs in selected features
    df.dropna(subset=case['selected_features'], inplace=True)
    
    # Separate features and target
    X = df[case['selected_features']]
    y = df['Chlorophyll-a (ug/L)']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=case['test_size'], random_state=42)
    
    # Model training
    model = case['model']
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Predict for the whole dataset
    df['Predicted Chlorophyll-a'] = model.predict(X)

    
    # Model evaluation
    #evaluate_model(model, X_train, X_test, y_train, y_test)

    # Return the trained model along with other results
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': case['selected_features'],
        'df': df
    }
    
    # Group by latitude and longitude
    location_counts = df.groupby(['Lat', 'Long']).size().reset_index(name='TotalDataPoints')
    hab_counts = df[df['Predicted Chlorophyll-a'] > case['threshold']].groupby(['Lat', 'Long']).size().reset_index(name='HABOccurrences')
    location_counts = location_counts.merge(hab_counts, on=['Lat', 'Long'], how='left')
    location_counts['HABOccurrences'].fillna(0, inplace=True)
    location_counts['NormalizedHABOccurrences'] = location_counts['HABOccurrences'] / location_counts['TotalDataPoints']
    max_total_data_points = location_counts['TotalDataPoints'].max()
    location_counts['NormalizedTotalDataPoints'] = location_counts['TotalDataPoints'] / max_total_data_points
    location_counts['HABRiskQuotient'] = location_counts['NormalizedHABOccurrences'] * location_counts['NormalizedTotalDataPoints']
    

    # Create main plot with specified extent
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Plot coastlines
    ax.coastlines()

    # Plot HAB Risk Quotient
    sc = ax.scatter(location_counts['Long'], location_counts['Lat'], c=location_counts['HABRiskQuotient'], cmap='OrRd', marker='o', s=5000, alpha=0.8)
    plt.colorbar(sc, label='HAB Risk Quotient')
    plt.title(f'HAB Risk Quotient for {case["name"]}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Process each case
#for case in cases:
    #process_case(case)

    

#Introduction Page
st.sidebar.title('Pages')
selected_page = st.sidebar.radio('Go to', ['Introduction', 'Apalachicola Bay-Estuary', 'Pensacola-Perdido Bay-Estuary'])

if selected_page == 'Introduction':
    st.title('Introduction')
    st.write('This is an application to evaluate the Apalachicola Bay Model.')

elif selected_page == 'Apalachicola Bay-Estuary':
    #st.header('Gauged Stations')
    #st.map(df_ap_nut,latitude='lat',longitude='lon',use_container_width=True)
    # Subpage navigation for Apalachicola Bay-Estuary
    subpage_selected = st.sidebar.radio('Go to', ['Historical Observation', 'Prediction', 'Vulnerability'])
    # Content for subpages of Apalachicola Bay-Estuary
    if subpage_selected == 'Historical Observation':
      
        st.header('Historical Observations')

        df_ap_nut = pd.read_csv('combined_AP_nut.csv')
        df_ap_nut['Date'] = pd.to_datetime(df_ap_nut['Date'])

        # Calculate Map extent
        extent = [df_ap_nut['lon'].min()-0.2, df_ap_nut['lon'].max()+0.2, df_ap_nut['lat'].min()-0.2, df_ap_nut['lat'].max()+0.2]

        # Calculate number of ticks
        num_ticks = 5
        lon_ticks = np.linspace(extent[0], extent[1], num_ticks)
        lat_ticks = np.linspace(extent[2], extent[3], num_ticks)

        # Sort station codes based on longitude
        sorted_station_codes = sorted(df_ap_nut['station_code'].unique(), key=lambda x: df_ap_nut[df_ap_nut['station_code'] == x]['lon'].iloc[0])

        # Create a dictionary to store coordinates for each station
        station_coordinates = defaultdict(list)
        for i, station in enumerate(sorted_station_codes):
          station_name = f'{i+1}'
          station_coordinates[station_name] = (df_ap_nut[df_ap_nut['station_code'] == station]['lon'].iloc[0], df_ap_nut[df_ap_nut['station_code'] == station]['lat'].iloc[0])

        # Sort station coordinates by longitude
        sorted_station_coordinates = sorted(station_coordinates.items(), key=lambda x: x[1][0])
    
        # Set default values for year and month based on the data range
        min_date = df_ap_nut['Date'].min().date()
        max_date = df_ap_nut['Date'].max().date()
        default_date = min_date + (max_date - min_date) // 2
        default_year = default_date.year
        default_month = default_date.month
    
        # Slider for selecting year and month
        selected_year = st.slider('Select Year', min_value=min_date.year, max_value=max_date.year, value=default_year)
        selected_month = st.slider('Select Month', min_value=1, max_value=12, value=default_month)
    
        # Create map based on selected year and month
        fig = create_map(selected_year, selected_month)

        # Display the map
        with st.container(height= 500, border=True):
           st.pyplot(fig)

    if subpage_selected == 'Prediction':
        # Display a table summarizing the data size and selected features
        st.write("## Data Summary")
        st.write(f"Number of Samples: {len(X)}")
        st.write(f"Number of Features: {X.shape[1]}")
        st.write(f"Selected Features: {selected_features}")
        st.write(f"Target Variable: Chlorophyll-a (ug/L)")
        # Display the train and test sample sizes
        st.write("Train/Test Sample Sizes: 80% Training and 20% Testing")

        # Process the Apalachicola case
        apalachicola_case = process_case(cases[0]) 

        # Button to evaluate the model
        if st.button('Evaluate Model'):
           # Evaluate the model using the returned model from process_case
           evaluate_model(apalachicola_case['model'], apalachicola_case['X_train'], apalachicola_case['X_test'], apalachicola_case['y_train'], apalachicola_case['y_test'])

        # Remaining code for uploading user's CSV file, making predictions, and downloading results
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)
            # Ask the user to match column names with selected features
            st.write("### Match Column Names")
            selected_columns = {}
            for feature in apalachicola_case['selected_features']:
                selected_columns[feature] = st.selectbox(f"Select column for '{feature}'", user_data.columns)
            # Extract the selected columns from user_data
            user_data_selected = user_data[list(selected_columns.values())]
            
            # Ensure column names match expected features
            if set(user_data_selected.columns) == set(apalachicola_case['selected_features']):
                # Make predictions
                user_data['Predicted Chlorophyll-a (ug/L)'] = apalachicola_case['model'].predict(user_data[apalachicola_case['selected_features']])
                # Display the modified DataFrame
                st.write("## Predicted Data")
                st.write(user_data)

                # Download the results as a CSV file
                st.write("Download the results as a CSV file.")
                csv = user_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="model_b_predictions.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.write("Uploaded CSV file does not contain expected features.")

