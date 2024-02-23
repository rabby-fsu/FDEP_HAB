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
    st.title('Gauged Stations')

    # Create the map
    map_data = df[['lat', 'lon']]
    clicked = st.button("Click on the map to get Chlorophyll-a (ug/L)")
    if clicked:
        click_result = st.map(map_data)
        if click_result:
            clicked_point = (click_result["lat"], click_result["lon"])
            selected_location_data = df[(df['lat'] == clicked_point[0]) & (df['lon'] == clicked_point[1])]
            if not selected_location_data.empty:
                st.write("Boxplot of Chlorophyll-a (ug/L) for the selected location:")
                st.write(selected_location_data['Chlorophyll-a (ug/L)'].plot(kind='box'))
            else:
                st.write("No data available for the selected location.")
