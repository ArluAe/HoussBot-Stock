# src/train.py

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from model import StockPredictor
from preprocess import prepare_data

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(predictions.squeeze().tolist())
            all_labels.extend(y_batch.tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.4f}')

def main():
    file_path = 'tech/data/apple_stock.csv'
    X, y, scaler = prepare_data(file_path)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to tensors
    X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train).view(-1, 1)
    X_test, y_test = torch.Tensor(X_test), torch.Tensor(y_test).view(-1, 1)

    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Model, Loss, Optimizer
    input_size = X_train.shape[2]
    model = StockPredictor(input_size=input_size, hidden_size=50, num_layers=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()
