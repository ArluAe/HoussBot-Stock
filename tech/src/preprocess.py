# src/preprocess.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    # Load CSV with specified column names
    data = pd.read_csv(file_path, names=["Date", "Close", "Volume", "Open", "High", "Low"], skiprows=1)

    # Convert "Date" column to datetime
    data["Date"] = pd.to_datetime(data["Date"], format="%m/%d/%Y")
    data.set_index("Date", inplace=True)

    # Remove dollar signs and commas, and convert columns to numeric
    data["Close"] = pd.to_numeric(data["Close"].replace('[\$,]', '', regex=True))
    data["Volume"] = pd.to_numeric(data["Volume"].replace('[,]', '', regex=True))
    data["Open"] = pd.to_numeric(data["Open"].replace('[\$,]', '', regex=True))
    data["High"] = pd.to_numeric(data["High"].replace('[\$,]', '', regex=True))
    data["Low"] = pd.to_numeric(data["Low"].replace('[\$,]', '', regex=True))

    return data

def feature_engineering(data):
    # Compute additional features
    data["Return"] = data["Close"].pct_change()  # Daily returns
    data["Direction"] = (data["Close"].shift(-1) > data["Close"]).astype(int)  # Binary target for up/down
    data.dropna(inplace=True)  # Remove any NaNs
    return data

def scale_features(data):
    # Standardize features for stable training
    scaler = StandardScaler()
    features = scaler.fit_transform(data[["Close", "Volume", "Open", "High", "Low", "Return"]])
    return features, scaler

def create_sequences(features, target, sequence_length=60):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

def prepare_data(file_path, sequence_length=60):
    data = load_data(file_path)
    data = feature_engineering(data)
    features, scaler = scale_features(data)
    X, y = create_sequences(features, data["Direction"].values, sequence_length)
    return X, y, scaler
