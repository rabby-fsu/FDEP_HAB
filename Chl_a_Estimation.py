import pandas as pd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import cartopy.feature as cfeature
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

df = pd.read_csv('PenPerd.csv')


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
        'data_file': 'PenPerd.csv',
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
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': case['selected_features'],
        'df': df,
        'threshold': case['threshold']
    }
    

# Function to generate HAB Risk Quotient map
def generate_hab_quotient_map(df, case, scenario, min_lat=None, max_lat=None,min_lon=None, max_lon=None):
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

    # Add OpenStreetMap basemap
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Plot coastlines
    ax.coastlines()

    # Plot HAB Risk Quotient
    sc = ax.scatter(location_counts['Long'], location_counts['Lat'], c=location_counts['HABRiskQuotient'], cmap='OrRd', marker='o', s=200, alpha=1, edgecolors='green')
    plt.colorbar(sc, label='HAB Risk Quotient')
    # Modify the way to set the title to avoid KeyError
    plt.title(f'HAB Risk Quotient - {scenario}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Set latitude and longitude limits if provided
    if min_lat is not None and max_lat is not None:
        ax.set_ylim(bottom=min_lat, top=max_lat)
    if min_lon is not None and max_lon is not None:
        ax.set_xlim(left=min_lon, right=max_lon)
    
    # Set latitude and longitude as ticks based on min and max values
    if min_lat is not None and max_lat is not None:
        # Calculate the interval between min and max values
        lat_interval = (max_lat - min_lat) / 3
        # Calculate tick positions
        lat_ticks = [min_lat, min_lat + lat_interval, max_lat - lat_interval, max_lat]
        ax.set_yticks(lat_ticks)
        
    if min_lon is not None and max_lon is not None:
        # Calculate the interval between min and max values
        lon_interval = (max_lon - min_lon) / 3
        # Calculate tick positions
        lon_ticks = [min_lon, min_lon + lon_interval, max_lon - lon_interval, max_lon]
        ax.set_xticks(lon_ticks)


    return fig




# Process each case
#for case in cases:
    #process_case(case)

def handle_prediction(subpage_name, case_index):
    # Process the selected case
    selected_case = process_case(cases[case_index])

    # Button to evaluate the model
    if st.button('Evaluate Model'):
        # Evaluate the model using the returned model from process_case
        evaluate_model(selected_case['model'], selected_case['X_train'], selected_case['X_test'], selected_case['y_train'], selected_case['y_test'])

    # Remaining code for uploading user's CSV file, making predictions, and downloading results
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        # Ask the user to match column names with selected features
        st.write("### Match Column Names")
        selected_columns = {}
        for feature in selected_case['selected_features']:
            selected_columns[feature] = st.selectbox(f"Select column for '{feature}'", user_data.columns)
        # Extract the selected columns from user_data
        user_data_selected = user_data[list(selected_columns.values())]

        # Ensure column names match expected features
        if set(user_data_selected.columns) == set(selected_case['selected_features']):
            # Make predictions
            user_data['Predicted Chlorophyll-a (ug/L)'] = selected_case['model'].predict(user_data[selected_case['selected_features']])
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


#Introduction Page
st.sidebar.title('Pages')
selected_page = st.sidebar.radio('Go to', ['Introduction', 'Apalachicola Bay-Estuary', 'Saint Joseph Bay-Estuary', 'Saint Andrew Bay-Estuary', 'Pensacola-Perdido Bay-Estuary'])

if selected_page == 'Introduction':
    st.title('Introduction')
    st.write('This is an application to evaluate the Apalachicola Bay Model.')

elif selected_page == 'Apalachicola Bay-Estuary':
    # Subpage navigation for Apalachicola Bay-Estuary
    subpage_selected = st.sidebar.radio('Go to', ['Prediction', 'Vulnerability'])
    if subpage_selected == 'Prediction':
        handle_prediction('Apalachicola', 0)  # Passing the subpage name and case index
    elif subpage_selected == 'Vulnerability':
        # Your code for vulnerability
        selected_case = process_case(cases[0])

       
        # Sliders for scenarios
        ocean_acidification = st.slider('Ocean Acidification', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        cool_warm_climate = st.slider('Cool-Warm Climate', min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
        salinity_change = st.slider('Salinity Change (%)', min_value=-100, max_value=100, value=0, step=1)

        original_predictions = cases[0]['model'].predict(selected_case['X'])
        selected_case['df']['Predicted Chlorophyll-a'] = original_predictions

        # Generate map for Business-as-Usual
        plot1= generate_hab_quotient_map(selected_case['df'], selected_case, scenario='Business-as-Usual',min_lat=29.5, max_lat=30.5, min_lon=-85, max_lon=-83)
        # Display plots side by side using columns layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("Plot 1")
            st.pyplot(plot1)

        # Generate maps for Business-as-Usual and Hypothetical Scenario
        modified_df = selected_case['df'].copy()  # Corrected copy operation
        modified_df['pH'] += ocean_acidification  # Apply modifications to the copied DataFrame
        modified_df['Salinity(ppt)'] *= ((salinity_change / 100) + 1)
        # Columns to modify for cool-warm climate
        columns_to_modify = ['ATemp_max', 'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']
        # Apply modification for cool-warm climate to selected columns
        modified_df[columns_to_modify] += cool_warm_climate




        # Predict chlorophyll-a for modified scenario
        modified_predictions = cases[0]['model'].predict(modified_df[selected_case['selected_features']])
        modified_df['Predicted Chlorophyll-a'] = modified_predictions
        # Generate map for Hypothetical Scenario
        plot2 = generate_hab_quotient_map(modified_df, cases[0], scenario='Hypothetical Scenario',min_lat=29.5, max_lat=30.5, min_lon=-85, max_lon=-83)  # Pass modified DataFrame



        with col2:
            st.write("Plot 2")
            st.pyplot(plot2)


elif selected_page == 'Saint Joseph Bay-Estuary':
    # Subpage navigation for Joseph Bay-Estuary
    subpage_selected = st.sidebar.radio('Go to', ['Prediction', 'Vulnerability'])
    if subpage_selected == 'Prediction':
        handle_prediction('Joseph', 1)  # Passing the subpage name and case index
    elif subpage_selected == 'Vulnerability':
        # Your code for vulnerability
        selected_case = process_case(cases[1])


        # Sliders for scenarios
        ocean_acidification = st.slider('Ocean Acidification', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        cool_warm_climate = st.slider('Cool-Warm Climate', min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
        salinity_change = st.slider('Salinity Change (%)', min_value=-100, max_value=100, value=0, step=1)


        original_predictions = cases[1]['model'].predict(selected_case['X'])
        selected_case['df']['Predicted Chlorophyll-a'] = original_predictions

        # Generate map for Business-as-Usual
        plot1= generate_hab_quotient_map(selected_case['df'], selected_case, scenario='Business-as-Usual',min_lat=29.65, max_lat=29.9, min_lon=-85.42, max_lon=-85.29)
        # Display plots side by side using columns layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("Plot 1")
            st.pyplot(plot1)

        # Generate maps for Business-as-Usual and Hypothetical Scenario
        modified_df = selected_case['df'].copy()  # Corrected copy operation
        modified_df['pH'] += ocean_acidification  # Apply modifications to the copied DataFrame
        modified_df['Salinity(ppt)'] *= ((salinity_change / 100) + 1)
        # Columns to modify for cool-warm climate
        columns_to_modify = ['ATemp_max', 'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']
        # Apply modification for cool-warm climate to selected columns
        modified_df[columns_to_modify] += cool_warm_climate

        

        # Predict chlorophyll-a for modified scenario
        modified_predictions = cases[1]['model'].predict(modified_df[selected_case['selected_features']])
        modified_df['Predicted Chlorophyll-a'] = modified_predictions
        # Generate map for Hypothetical Scenario
        plot2 = generate_hab_quotient_map(modified_df, cases[1], scenario='Hypothetical Scenario',min_lat=29.65, max_lat=29.9, min_lon=-85.42, max_lon=-85.29)  # Pass modified DataFrame



        with col2:
            st.write("Plot 2")
            st.pyplot(plot2)

elif selected_page == 'Saint Andrew Bay-Estuary':
    # Subpage navigation for Saint Andrew Bay-Estuary
    subpage_selected = st.sidebar.radio('Go to', ['Prediction', 'Vulnerability'])
    if subpage_selected == 'Prediction':
        handle_prediction('Andrew', 2)  # Passing the subpage name and case index
    elif subpage_selected == 'Vulnerability':
        # Your code for vulnerability
        selected_case = process_case(cases[2])

        
        # Sliders for scenarios
        ocean_acidification = st.slider('Ocean Acidification', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        cool_warm_climate = st.slider('Cool-Warm Climate', min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
        salinity_change = st.slider('Salinity Change (%)', min_value=-100, max_value=100, value=0, step=1)

        original_predictions = cases[2]['model'].predict(selected_case['X'])
        selected_case['df']['Predicted Chlorophyll-a'] = original_predictions

        # Generate map for Business-as-Usual
        plot1= generate_hab_quotient_map(selected_case['df'], selected_case, scenario='Business-as-Usual',min_lat=30, max_lat=30.35, min_lon=-85.9, max_lon=-85.35)
        # Display plots side by side using columns layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("Plot 1")
            st.pyplot(plot1)

        # Generate maps for Business-as-Usual and Hypothetical Scenario
        modified_df = selected_case['df'].copy()  # Corrected copy operation
        modified_df['pH'] += ocean_acidification  # Apply modifications to the copied DataFrame
        modified_df['Salinity(ppt)'] *= ((salinity_change / 100) + 1)
        # Columns to modify for cool-warm climate
        columns_to_modify = ['ATemp_max', 'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']
        # Apply modification for cool-warm climate to selected columns
        modified_df[columns_to_modify] += cool_warm_climate

        # Predict chlorophyll-a for modified scenario
        modified_predictions = cases[2]['model'].predict(modified_df[selected_case['selected_features']])
        modified_df['Predicted Chlorophyll-a'] = modified_predictions
        # Generate map for Hypothetical Scenario
        plot2 = generate_hab_quotient_map(modified_df, cases[2], scenario='Hypothetical Scenario',min_lat=30, max_lat=30.35, min_lon=-85.9, max_lon=-85.35)  # Pass modified DataFrame



        with col2:
            st.write("Plot 2")
            st.pyplot(plot2)


elif selected_page == 'Pensacola-Perdido Bay-Estuary':
    # Subpage navigation for Saint Andrew Bay-Estuary
    subpage_selected = st.sidebar.radio('Go to', ['Prediction', 'Vulnerability'])
    if subpage_selected == 'Prediction':
        handle_prediction('Pensacola-Perdido', 3)  # Passing the subpage name and case index
    elif subpage_selected == 'Vulnerability':
        # Your code for vulnerability
        selected_case = process_case(cases[3])

        # Sliders for scenarios
        ocean_acidification = st.slider('Ocean Acidification', min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        cool_warm_climate = st.slider('Cool-Warm Climate', min_value=-10.0, max_value=10.0, value=0.0, step=1.0)
        salinity_change = st.slider('Salinity Change (%)', min_value=-100, max_value=100, value=0, step=1)

        original_predictions = cases[3]['model'].predict(selected_case['X'])
        selected_case['df']['Predicted Chlorophyll-a'] = original_predictions

        # Generate map for Business-as-Usual
        plot1= generate_hab_quotient_map(selected_case['df'], selected_case, scenario='Business-as-Usual',min_lat=30, max_lat=31, min_lon=-87.6, max_lon=-86.8)
        # Display plots side by side using columns layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("Plot 1")
            st.pyplot(plot1)

        # Generate maps for Business-as-Usual and Hypothetical Scenario
        modified_df = selected_case['df'].copy()  # Corrected copy operation
        modified_df['pH'] += ocean_acidification  # Apply modifications to the copied DataFrame
        modified_df['Salinity(ppt)'] *= ((salinity_change / 100) + 1)
        # Columns to modify for cool-warm climate
        columns_to_modify = ['ATemp_max', 'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag', 'ATemp_max_7dlag']
        # Apply modification for cool-warm climate to selected columns
        modified_df[columns_to_modify] += cool_warm_climate

        # Predict chlorophyll-a for modified scenario
        modified_predictions = cases[3]['model'].predict(modified_df[selected_case['selected_features']])
        modified_df['Predicted Chlorophyll-a'] = modified_predictions
        # Generate map for Hypothetical Scenario
        plot2 = generate_hab_quotient_map(modified_df, cases[3], scenario='Hypothetical Scenario',min_lat=30, max_lat=34, min_lon=-89, max_lon=-86)  # Pass modified DataFrame



        with col2:
            st.write("Plot 2")
            st.pyplot(plot2)
