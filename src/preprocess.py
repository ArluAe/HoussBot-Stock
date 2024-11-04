# preprocess.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def feature_engineering(data):
    data['Return'] = data['Close'].pct_change()
    data['Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    return data

def scale_features(data):
    scaler = StandardScaler()
    features = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']])
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
    X, y = create_sequences(features, data['Direction'].values, sequence_length)
    return X, y, scaler
