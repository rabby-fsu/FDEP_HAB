import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import streamlit as st



# Load data
df = pd.read_csv('DataFile_ML_All.csv')

# Define selected features
selected_features = ['Salinity(ppt)', 'Turbidity(NTU)', 'DO(mg/l)','pH','ATemp_max',
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
df['Predicted'] = predicted_chlorophyll


st.map(df,
    latitude='Latitude',
    longitude='Longitude',
    size='Predicted')


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
