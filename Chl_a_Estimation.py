import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk
import folium
import cartopy.crs as ccrs
from collections import defaultdict
import matplotlib.colors as mcolors


# Load data
df = pd.read_csv('DataFile_ML_All.csv')
df_ap_nut = pd.read_csv('combined_AP_nut.csv')

# Define selected features
selected_features = ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)', 'pH', 'ATemp_max',
                     'ATemp_max_1dlag', 'ATemp_max_2dlag', 'ATemp_max_3dlag',
                     'ATemp_max_4dlag', 'ATemp_max_5dlag', 'ATemp_max_6dlag',
                     'ATemp_max_7dlag']

X = df[selected_features]
y = df['Chlorophyll-a (ug/L)']


# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df_ap_nut['Date'] = pd.to_datetime(df_ap_nut['Date'])

# Function to create colored markers based on chlorophyll-a concentration
def color_marker(chl_a):
    if chl_a <= 10:
        return 'green'  # No bloom
    else:
        return 'red'  # Bloom

# Define color marker function
def color_marker(chl_a):
    if chl_a <= 10:
        return 'green'  # No bloom
    else:
        return 'red'  # Bloom
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
    station_name = f'G{i+1}'
    station_coordinates[station_name] = (df_ap_nut[df_ap_nut['station_code'] == station]['lon'].iloc[0], df_ap_nut[df_ap_nut['station_code'] == station]['lat'].iloc[0])

# Sort station coordinates by longitude
sorted_station_coordinates = sorted(station_coordinates.items(), key=lambda x: x[1][0])



# Function to create map
def create_map(selected_year, selected_month):
    # Filter data for the selected year and month
    filtered_df = df_ap_nut[(df_ap_nut['Date'].dt.year == selected_year) & (df_ap_nut['Date'].dt.month == selected_month)]

    # Create main plot with specified extent
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), extent=extent)

    # Plot coastlines
    ax.coastlines()
        # Annotate station names and handle overlapping
    used_coordinates = set()
    for i, (station, (lon, lat)) in enumerate(sorted_station_coordinates):
        arrow_shift = 0
        while (lon, lat) in used_coordinates:  # Check for overlapping
            lat += 0.015  # Adjust the latitude to avoid overlapping
            arrow_shift += 0.5

        # Annotate station name with arrow
        ax.annotate(station, xy=(lon, lat), xytext=(15, 15), textcoords='offset points', fontsize=10, color='red',
                    arrowprops=dict(facecolor='red', arrowstyle='->'))

        used_coordinates.add((lon, lat))
    # Plot chlorophyll-a concentration using color plot
    sc = ax.scatter(filtered_df['lon'], filtered_df['lat'], s=100, c=filtered_df['Chlorophyll-a (ug/L)'], cmap='BuGn', edgecolor='black',vmin=df_ap_nut['Chlorophyll-a (ug/L)'].min(), vmax=df_ap_nut['Chlorophyll-a (ug/L)'].max())
    #sc = ax.scatter(filtered_df['lon'], filtered_df['lat'], s=100, c=filtered_df['Chlorophyll-a (ug/L)'], cmap='BuGn', edgecolor='black',vmin=df_ap_nut['Chlorophyll-a (ug/L)'].min(), vmax=df_ap_nut['Chlorophyll-a (ug/L)'].max())

    # Handle overlapping points
    for i, (x, y) in enumerate(zip(filtered_df['lon'], filtered_df['lat'])):
        shift_amount = 0.01  # Adjust this value as needed
        if (x, y) in used_coordinates:
            x += shift_amount
            y += shift_amount
        used_coordinates.add((x, y))
        ax.scatter(x, y, s=100, c=filtered_df['Chlorophyll-a (ug/L)'].iloc[i], cmap='BuGn', edgecolor='black', vmin=df_ap_nut['Chlorophyll-a (ug/L)'].min(), vmax=df_ap_nut['Chlorophyll-a (ug/L)'].max())

      
    # Add color bar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Chlorophyll-a (ug/L)')

    # Set x and y ticks as longitude and latitude
    ax.set_xticks(lon_ticks)
    ax.set_yticks(lat_ticks)

    # Set labels for x and y ticks
    ax.set_xticklabels([f"{x:.1f}" for x in lon_ticks])
    ax.set_yticklabels([f"{y:.1f}" for y in lat_ticks])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Map withChlorophyll-a Concentration')

    return fig




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
    st.map(df_ap_nut,latitude='lat',longitude='lon',use_container_width=True)


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
    st.pyplot(fig)
