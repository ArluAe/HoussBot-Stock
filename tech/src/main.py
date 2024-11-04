# src/main.py

import torch
from model import StockPredictor
from train import train_model, evaluate_model
from utils import save_model, load_model, predict_next_day
from preprocess import load_data, feature_engineering, scale_features, create_sequences
from torch.utils.data import DataLoader, TensorDataset
import os

def main(data_path, model_path):
    # File paths
    #data_path = 'tech/data/google_stock.csv'
    #model_path = 'stock_predictor1.pth'
    
    # Load data and prepare for training
    data = load_data(data_path)
    data = feature_engineering(data)
    features, scaler = scale_features(data)

    # Create sequences for training
    X, y = create_sequences(features, data["Direction"].values, sequence_length=60)
    
    # Split data into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to tensors
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train).view(-1, 1)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test).view(-1, 1)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = X_train.shape[2]
    model = StockPredictor(input_size=input_size, hidden_size=50, num_layers=2)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Check if model exists and load it, else train and save the model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model, model_path)
        
    else:
        print("Training model..." + model_path)
        train_model(model, train_loader, criterion, optimizer, num_epochs=100)
        print("Evaluating model...")
        evaluate_model(model, test_loader)
        print("Saving model..." + model_path)
        save_model(model, model_path)
    
    # Prepare the last 60 days of data for prediction
    recent_data = data.tail(60)  # Get the last 60 days of stock data
    recent_features = recent_data[["Close", "Volume", "Open", "High", "Low", "Return"]].values
    recent_features = scaler.transform(recent_features)  # Scale the recent data
    recent_features = torch.tensor(recent_features, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 60, features)

    # Predict next day's movement
    prediction = predict_next_day(model, recent_features)

    print(f"Prediction for the next day: {prediction}")

if __name__ == '__main__':
    main("tech/data/google_stock.csv", "google.pth")
    main("tech/data/apple_stock.csv", "apple.pth")
    main("tech/data/adobe_stock.csv", "adobe.pth")
    main("tech/data/meta_stock.csv", "meta.pth")
    main("tech/data/uber_stock.csv", "uber.pth")
    main("tech/data/tesla_stock.csv", "tesla.pth")
    main("tech/data/nvidia_stock.csv", "nvidia.pth")
    main("tech/data/miscrosoft_stock.csv", "microsoft.pth")
    main("tech/data/amazon_stock.csv", "amazon.pth")

