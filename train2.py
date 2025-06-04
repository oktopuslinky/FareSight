# saves pickle files

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import pickle

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
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

# List of dates
dates = [
    '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05',
    '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11'
]

# Load and concatenate all datasets
print("Loading datasets...")
all_dfs = []
for date in dates:
    df = load_kaggle_dataset(date)
    if df is not None:
        all_dfs.append(df)

# Concatenate all dataframes
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

# One-hot encode PULocationID
onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
pulocation_encoded = onehot.fit_transform(X[['PULocationID']])

# Print number of PULocationID categories for debugging
print(f"Number of PULocationID categories: {len(onehot.categories_[0])}")

# Save OneHotEncoder
with open('onehot.pkl', 'wb') as f:
    pickle.dump(onehot, f)
print("Saved OneHotEncoder to onehot.pkl")

# Normalize numerical features
scaler = StandardScaler()
numerical_features = scaler.fit_transform(X[['hour', 'day_of_week', 'month', 'year']])

# Save StandardScaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved StandardScaler to scaler.pkl")

# Combine features
X_processed = np.hstack([pulocation_encoded, numerical_features])

# Normalize target
y_scaler = StandardScaler()
y_processed = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Save y_scaler
with open('y_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)
print("Saved y_scaler to y_scaler.pkl")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Custom Dataset
class RideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_dataset = RideDataset(X_train, y_train)
val_dataset = RideDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define PyTorch model
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

# Initialize model, loss, and optimizer
input_size = X_processed.shape[1]
model = RidePredictionModel(input_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load saved model if it exists
model_path = 'ride_prediction_model.pth'
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded saved model from {model_path}")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Starting with a fresh model.")
else:
    print(f"No saved model found at {model_path}. Starting with a fresh model.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1
print("Training model...")
best_val_loss = 1
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Save model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

# Evaluate model (RMSE)
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch).squeeze().cpu().numpy()
        predictions.extend(outputs)
        actuals.extend(y_batch.numpy())

# Inverse transform predictions and actuals
predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = y_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
print(f'Validation RMSE: {rmse:.2f} rides')

# Example prediction function
def predict_rides(pulocation_id, hour, day_of_week, month, year, model, onehot, scaler, y_scaler, device):
    model.eval()
    # Create input dataframe
    input_data = pd.DataFrame({
        'PULocationID': [pulocation_id],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'year': [year]
    })
    # Preprocess
    pulocation_enc = onehot.transform(input_data[['PULocationID']])
    numerical = scaler.transform(input_data[['hour', 'day_of_week', 'month', 'year']])
    X = np.hstack([pulocation_enc, numerical])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    # Predict
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()
    # Inverse transform
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred[0]

# Example usage
example_pred = predict_rides(
    pulocation_id=79,  # Example zone
    hour=12,          # Noon
    day_of_week=2,    # Wednesday
    month=6,          # June
    year=2022,
    model=model,
    onehot=onehot,
    scaler=scaler,
    y_scaler=y_scaler,
    device=device
)
print(f'Predicted rides for PULocationID 79 on 2022-06, Wednesday, 12:00: {example_pred:.2f}')