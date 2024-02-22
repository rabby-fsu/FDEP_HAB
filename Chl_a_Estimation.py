import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the XGBoost Regressor
xgb_regressor_1 = XGBRegressor(n_estimators=334, max_depth=4, learning_rate=0.07818940902700418, random_state=42)
xgb_regressor_1.fit(X_train, y_train)

# Predict chlorophyll-a concentrations
predicted_chlorophyll = xgb_regressor_1.predict(X)

# Add predicted chlorophyll-a concentrations as a new column in the DataFrame
df['Predicted Chlorophyll-a (ug/L)'] = predicted_chlorophyll

# Extract longitude, latitude, and predicted chlorophyll-a data
longitudes = df['Long']
latitudes = df['Lat']
predicted_chlorophyll_a = df['Predicted Chlorophyll-a (ug/L)']

# Create main plot with specified extent
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Plot coastlines
ax.coastlines()

# Plot chlorophyll-a concentrations on the map
sc = ax.scatter(longitudes, latitudes, s=100, c=predicted_chlorophyll_a, cmap='viridis', transform=ccrs.PlateCarree())
plt.colorbar(sc, ax=ax, label='Predicted Chlorophyll-a (ug/L)')

# Annotate station names with station codes
for i, txt in enumerate(df['station_code']):
    ax.annotate(txt, (longitudes[i], latitudes[i]), color='red')

# Set labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Map with Predicted Chlorophyll-a Concentration')

# Show plot
st.pyplot(fig)

