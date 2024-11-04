# src/utils.py

import torch

def save_model(model, path='stock_predictor.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='stock_predictor.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
