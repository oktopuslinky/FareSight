import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Define the model architecture (must match the trained model)
class RidePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(RidePredictionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Function to load and preprocess data (same as original)
def load_kaggle_dataset(date):
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "jeffsinsel/nyc-fhvhv-data",
            f"fhvhv_tripdata_{date}.parquet",
            pandas_kwargs={"columns": ["hvfhs_license_num", "pickup_datetime", "PULocationID"]}
        )
        return df
    except Exception as e:
        print(f"Error loading data for {date}: {e}")
        return None

# Function to predict rides for all zones at a given time
def predict_rides_all_zones(hour, day_of_week, month, year, model_path='ride_prediction_model.pth'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define dates (same as original)
    dates = [
        '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05',
        '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11'
    ]
    
    # Load and concatenate datasets
    print("Loading datasets...")
    all_dfs = []
    for date in dates:
        df = load_kaggle_dataset(date)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data loaded. Exiting.")
        return None
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Filter for Uber rides
    uber_df = df[df["hvfhs_license_num"] == "HV0003"].copy()
    
    # Convert pickup_datetime to datetime
    uber_df['pickup_datetime'] = pd.to_datetime(uber_df['pickup_datetime'])
    
    # Extract time features
    uber_df['hour'] = uber_df['pickup_datetime'].dt.hour
    uber_df['day_of_week'] = uber_df['pickup_datetime'].dt.dayofweek
    uber_df['month'] = uber_df['pickup_datetime'].dt.month
    uber_df['year'] = uber_df['pickup_datetime'].dt.year
    
    # Aggregate rides by hour and PULocationID
    uber_df['hourly_interval'] = uber_df['pickup_datetime'].dt.floor('H')
    counts = uber_df.groupby(['hourly_interval', 'PULocationID']).size().reset_index(name='ride_count')
    
    # Merge with time features
    time_features = uber_df[['hourly_interval', 'PULocationID', 'hour', 'day_of_week', 'month', 'year']].drop_duplicates()
    data = counts.merge(time_features, on=['hourly_interval', 'PULocationID'], how='left')
    
    # Prepare features and target
    X = data[['PULocationID', 'hour', 'day_of_week', 'month', 'year']]
    y = data['ride_count'].values
    
    # Fit preprocessors
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    pulocation_encoded = onehot.fit_transform(X[['PULocationID']])
    
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(X[['hour', 'day_of_week', 'month', 'year']])
    
    X_processed = np.hstack([pulocation_encoded, numerical_features])
    
    y_scaler = StandardScaler()
    y_processed = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Get all possible PULocationIDs
    pulocation_ids = onehot.categories_[0]
    
    # Create input data for all zones
    input_data = pd.DataFrame({
        'PULocationID': pulocation_ids,
        'hour': [hour] * len(pulocation_ids),
        'day_of_week': [day_of_week] * len(pulocation_ids),
        'month': [month] * len(pulocation_ids),
        'year': [year] * len(pulocation_ids)
    })
    
    # Preprocess inputs
    pulocation_enc = onehot.transform(input_data[['PULocationID']])
    numerical = scaler.transform(input_data[['hour', 'day_of_week', 'month', 'year']])
    X = np.hstack([pulocation_enc, numerical])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Load model
    input_size = X.shape[1]
    model = RidePredictionModel(input_size).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("Model file not found. Please ensure ride_prediction_model.pth is in the directory.")
        return None
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    # Inverse transform predictions
    predictions = y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    
    # Create ranked list
    results = pd.DataFrame({
        'PULocationID': pulocation_ids,
        'Predicted_Rides': predictions
    })
    results = results.sort_values(by='Predicted_Rides', ascending=False).reset_index(drop=True)
    
    return results

# Example usage
if __name__ == "__main__":
    # Example: Predict rides for all zones on June 15, 2022 (Wednesday), at 12:00
    results = predict_rides_all_zones(
        hour=12,
        day_of_week=2,  # Wednesday
        month=6,
        year=2022
    )
    
    if results is not None:
        print(f"Ranked list of predicted rides for 2022-06, Wednesday, 12:00:")
        print(results)