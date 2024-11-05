# Stock Prediction Model

This repository contains a PyTorch-based LSTM model designed to predict the next-day movement of stock prices based on historical stock data. The model uses a binary classification approach to determine if the stock price will go up or down.

## Project Structure

- `src/main.py`: Main entry point to load data, train, evaluate, and make predictions with the model.
- `src/model.py`: Defines the LSTM-based `StockPredictor` model architecture.
- `src/preprocess.py`: Handles data loading, feature engineering, scaling, and sequence creation.
- `src/utils.py`: Includes utilities for saving/loading models and making predictions.
- `src/train.py`: Contains functions for training and evaluating the model.

## How It Works

### 1. Data Loading & Preprocessing
   - **Data Loading**: Loads CSV files with columns `Date`, `Close`, `Volume`, `Open`, `High`, `Low`.
   - **Feature Engineering**: Adds new features, such as daily returns and the binary target variable for next-day direction (`Direction`).
   - **Scaling**: Standardizes features for stable training using `StandardScaler`.
   - **Sequence Creation**: Generates sequences of 60 days of data to use as input for the LSTM model. Each sequence is paired with a binary label indicating the stock's movement on the 61st day.

### 2. Model Architecture (`StockPredictor`)
   - The model architecture is an LSTM network with:
     - LSTM layers to capture temporal patterns in stock data.
     - A fully connected layer to produce the output, predicting the probability of upward movement.
   - `input_size`, `hidden_size`, and `num_layers` are configurable parameters that define the LSTM's structure.

### 3. Training & Evaluation
   - **Train-Test Split**: Splits data into 80% training and 20% testing.
   - **Loss Function**: Uses `BCEWithLogitsLoss`, suitable for binary classification with raw logits.
   - **Optimizer**: Adam optimizer with a learning rate of 0.001.
   - **Training Process**: Trains over multiple epochs, adjusting weights to minimize prediction error.
   - **Evaluation**: Computes accuracy on the test set by thresholding model outputs at 0.5.

### 4. Prediction
   - After training, the model can predict the stock's next-day movement using the most recent 60 days of data.
   - `predict_next_day` function returns a "Up" or "Down" label along with the probability score.

## Usage

Run `src/main.py` with paths to your CSV stock data files and the desired model output path. Example:

```bash
python src/main.py tech/data/google_stock.csv stock_predictor.pth
