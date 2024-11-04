# src/utils.py

import torch
from model import StockPredictor

def save_model(model, path='stock_predictor.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='stock_predictor.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_next_day(model, recent_data):
    """
    Predict whether the stock will go up or down the next day.
    
    Args:
        model (torch.nn.Module): The trained LSTM model.
        recent_data (torch.Tensor): Tensor containing the last 60 days of scaled stock data, 
                                    with shape (1, 60, num_features).

    Returns:
        str: "Up" if the model predicts the stock will go up, "Down" otherwise.
    """
    # Ensure recent_data has the necessary shape (1, 60, num_features)
    assert recent_data.shape[1] == 60, "recent_data should be a sequence of 60 days."
    
    # Make prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(recent_data)
    
    # Apply sigmoid to get a probability
    probability = torch.sigmoid(output).item()
    
    # Interpret probability as "Up" or "Down"
    return "Up" if probability > 0.5 else "Down"
