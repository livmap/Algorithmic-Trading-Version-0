import pandas as pd

data = pd.read_csv("/Users/princemaphupha/Desktop/Visual Studio Code/trading/data/Binance_BTCUSDT_1h.csv")

print("Step 1: Dataset Succesfully Loaded")

data = data.drop(columns=["Unix", "tradecount", "Symbol"])

data = data.sort_values(by='Date')

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Extract year, month, day, and hour into new columns
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Hour'] = data['Date'].dt.hour

# Short-Term Averages
data['SMA_70'] = data['Close'].rolling(window=70).mean()
data['SMA_140'] = data['Close'].rolling(window=140).mean()
data['EMA_140'] = data['Close'].ewm(span=140).mean()
data['SMA_240'] = data['Close'].rolling(window=240).mean()

# Medium-Term Averages
data['SMA_500'] = data['Close'].rolling(window=500).mean()
data['EMA_500'] = data['Close'].ewm(span=500).mean()
data['SMA_1000'] = data['Close'].rolling(window=1000).mean()
data['EMA_1000'] = data['Close'].ewm(span=1000).mean()

# Long-Term Averages
data['SMA_2000'] = data['Close'].rolling(window=2000).mean()
data['EMA_2000'] = data['Close'].ewm(span=2000).mean()

# Long-Term Averages
data['SMA_3000'] = data['Close'].rolling(window=3000).mean()
data['EMA_3000'] = data['Close'].ewm(span=3000).mean()

# Long-Term Averages
data['SMA_4000'] = data['Close'].rolling(window=4000).mean()
data['EMA_4000'] = data['Close'].ewm(span=4000).mean()

data = data.dropna()

data = data.drop(columns=['Date'])

# Filter data to start from 2023
data_2023 = data[data['Year'] >= 2024]

initial_cash = 0

# Define a new strategy using volume and price changes
def trading_bot_with_volume(data):
    # Initial parameters

    global initial_cash

    position = 0  # 0 means no position, 1 means holding BTC
    cash = 250  # Starting with $100,000
    btc_holding = 0  # Initial BTC holdings
    initial_cash = cash


    # Calculate additional metrics
    data['Volume_SMA20'] = data['Volume BTC'].rolling(window=20).mean()  # 20-hour moving average of volume
    data['Price_Change'] = data['Close'].pct_change()  # Percentage change in price
    data['Volume_Change'] = data['Volume BTC'].pct_change()  # Percentage change in volume

    # Iterate through the data
    for i in range(len(data)):
        if i < 20:  # Wait until enough data for the volume SMA
            continue

        current_price = data['Close'].iloc[i]

        # Buy signal: Volume spike (volume > SMA20 by a factor) and price increase
        if (
            position == 0
            and data['Volume BTC'].iloc[i] > 1.5 * data['Volume_SMA20'].iloc[i]
            and data['Price_Change'].iloc[i] > 0.01  # 1% price increase
        ):
            # Buy BTC with all available cash
            btc_holding = cash / current_price
            cash = 0
            position = 1  # Now holding BTC

        # Sell signal: Volume drop (volume < SMA20 by a factor) and price drop
        elif (
            position == 1
            and data['Volume BTC'].iloc[i] < 0.7 * data['Volume_SMA20'].iloc[i]
            and data['Price_Change'].iloc[i] < -0.01  # 1% price decrease
        ):
            # Sell all BTC holdings
            cash = btc_holding * current_price
            btc_holding = 0
            position = 0  # No longer holding BTC

    # Final portfolio value
    final_value = cash + (btc_holding * data['Close'].iloc[-1])

    return final_value - initial_cash

# Simulate the trading bot with the new strategy
profit_with_volume = trading_bot_with_volume(data_2023)
print(f"Profit with volume-based strategy: ${profit_with_volume:.2f}")
print(f"Percentage Increase: {((profit_with_volume / initial_cash) * 100):.2f}%")