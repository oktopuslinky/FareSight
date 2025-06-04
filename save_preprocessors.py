import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import kagglehub
from kagglehub import KaggleDatasetAdapter

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

def main():
    dates = [
        '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05',
        '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11'
    ]
    
    print("Loading datasets...")
    all_dfs = []
    for date in dates:
        df = load_kaggle_dataset(date)
        if df is not None:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data loaded. Exiting.")
        return
    
    df = pd.concat(all_dfs, ignore_index=True)
    uber_df = df[df["hvfhs_license_num"] == "HV0003"].copy()
    uber_df['pickup_datetime'] = pd.to_datetime(uber_df['pickup_datetime'])
    uber_df['hour'] = uber_df['pickup_datetime'].dt.hour
    uber_df['day_of_week'] = uber_df['pickup_datetime'].dt.dayofweek
    uber_df['month'] = uber_df['pickup_datetime'].dt.month
    uber_df['year'] = uber_df['pickup_datetime'].dt.year
    uber_df['hourly_interval'] = uber_df['pickup_datetime'].dt.floor('H')
    counts = uber_df.groupby(['hourly_interval', 'PULocationID']).size().reset_index(name='ride_count')
    time_features = uber_df[['hourly_interval', 'PULocationID', 'hour', 'day_of_week', 'month', 'year']].drop_duplicates()
    data = counts.merge(time_features, on=['hourly_interval', 'PULocationID'], how='left')
    
    X = data[['PULocationID', 'hour', 'day_of_week', 'month', 'year']]
    y = data['ride_count'].values
    
    onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    pulocation_encoded = onehot.fit_transform(X[['PULocationID']])
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(X[['hour', 'day_of_week', 'month', 'year']])
    y_scaler = StandardScaler()
    y_processed = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Save preprocessors
    with open('onehot.pkl', 'wb') as f:
        pickle.dump(onehot, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('y_scaler.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)
    with open('pulocation_ids.pkl', 'wb') as f:
        pickle.dump(onehot.categories_[0], f)
    
    print("Preprocessors saved as onehot.pkl, scaler.pkl, y_scaler.pkl, pulocation_ids.pkl")

if __name__ == "__main__":
    main()