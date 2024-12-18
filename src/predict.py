import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/princemaphupha/Desktop/Visual Studio Code/trading/data/Binance_BTCUSDT_1h.csv")

print("Step 1: Dataset Succesfully Loaded")

df = df.drop(columns=["Unix", "tradecount", "Symbol"])

df = df.sort_values(by='Date')

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract year, month, day, and hour into new columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour

# Short-Term Averages
df['SMA_70'] = df['Close'].rolling(window=70).mean()
df['SMA_140'] = df['Close'].rolling(window=140).mean()
df['EMA_140'] = df['Close'].ewm(span=140).mean()
df['SMA_240'] = df['Close'].rolling(window=240).mean()

# Medium-Term Averages
df['SMA_500'] = df['Close'].rolling(window=500).mean()
df['EMA_500'] = df['Close'].ewm(span=500).mean()
df['SMA_1000'] = df['Close'].rolling(window=1000).mean()
df['EMA_1000'] = df['Close'].ewm(span=1000).mean()

# Long-Term Averages
df['SMA_2000'] = df['Close'].rolling(window=2000).mean()
df['EMA_2000'] = df['Close'].ewm(span=2000).mean()

# Long-Term Averages
df['SMA_3000'] = df['Close'].rolling(window=3000).mean()
df['EMA_3000'] = df['Close'].ewm(span=3000).mean()

# Long-Term Averages
df['SMA_4000'] = df['Close'].rolling(window=4000).mean()
df['EMA_4000'] = df['Close'].ewm(span=4000).mean()

df = df.dropna()

df = df.drop(columns=['Date'])

print("Step 2: Dataset Fully Created")

# Step 3: Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

print("Step 3: Dataset Fully Normalized")

# Step 4: Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])   # Previous 'seq_length' hours
        y.append(data[i+seq_length, 0])  # Predict 'Close' price
    return np.array(X), np.array(y)

print("Step 4: Sequences Created for LSTM")

SEQ_LENGTH = 168  # Use past 168 hours to predict next hour
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Step 5: Split into training and testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Load the model
model = load_model('/Users/princemaphupha/Desktop/Visual Studio Code/trading/models/model.keras')

# Use the model for predictions
predictions = model.predict(X_test)

print(predictions)

scaled_predictions = np.zeros((len(predictions), scaled_data.shape[1]))
scaled_predictions[:, 0] = predictions[:, 0]  # Only replace 'Close' column
predicted_close = scaler.inverse_transform(scaled_predictions)[:, 0]

# Step 10: Plot Actual vs Predicted
'''
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], df['Close'].iloc[-len(y_test):], label="Actual Prices", color='blue')
plt.plot(df.index[-len(y_test):], predicted_close, label="Predicted Prices", color='red')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title("Actual vs Predicted Close Prices")
plt.legend()
plt.show()
'''

# Step 11: Predict the Next 7 Days (168 Hours)
last_sequence = scaled_data[-SEQ_LENGTH:]  # Use the last available sequence
future_predictions = []

for _ in range(168):  # Predict next 168 hours
    next_prediction = model.predict(last_sequence.reshape(1, SEQ_LENGTH, -1))
    future_predictions.append(next_prediction[0, 0])
    
    # Update the sequence with the predicted value
    new_row = np.zeros((1, scaled_data.shape[1]))
    new_row[0, 0] = next_prediction[0, 0]  # Replace 'Close'
    last_sequence = np.vstack([last_sequence[1:], new_row])

# Inverse transform future predictions
scaled_future = np.zeros((len(future_predictions), scaled_data.shape[1]))
scaled_future[:, 0] = future_predictions
future_close = scaler.inverse_transform(scaled_future)[:, 0]

print(future_close)

plt.figure(figsize=(12, 6))
plt.plot(range(168), future_close, label="Predicted Next 7 Days", color='green')
plt.xlabel('Hours Ahead')
plt.ylabel('Close Price')
plt.title("Predicted Close Prices for Next 7 Days")
plt.legend()
plt.show()

