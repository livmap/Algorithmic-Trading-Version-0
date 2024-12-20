import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/Users/princemaphupha/Desktop/Visual Studio Code/trading/data/Binance_BTCUSDT_1h.csv")

print("Step 1: Dataset Successfully Loaded")

# Drop unnecessary columns and sort
df = df.drop(columns=["Unix", "tradecount", "Symbol"])
df = df.sort_values(by='Date')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract year, month, day, and hour into new columns
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Hour'] = df['Date'].dt.hour

# Filter rows for the year 2024
df = df[df['Year'] == 2023]

# Calculate ATR (Average True Range) for volatility-based thresholds
def calculate_atr(df, window=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
    df['True_Range'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['True_Range'].rolling(window=window).mean()
    return df

# Apply ATR calculation
df = calculate_atr(df, window=24)  # 24-hour ATR for volatility

# Short-Term, Medium-Term, and Long-Term Averages
df['SMA_70'] = df['Close'].rolling(window=70).mean()
df['SMA_140'] = df['Close'].rolling(window=140).mean()
df['EMA_140'] = df['Close'].ewm(span=140).mean()
df['SMA_240'] = df['Close'].rolling(window=240).mean()

df['SMA_500'] = df['Close'].rolling(window=500).mean()
df['EMA_500'] = df['Close'].ewm(span=500).mean()
df['SMA_1000'] = df['Close'].rolling(window=1000).mean()
df['EMA_1000'] = df['Close'].ewm(span=1000).mean()

df['SMA_2000'] = df['Close'].rolling(window=2000).mean()
df['EMA_2000'] = df['Close'].ewm(span=2000).mean()

df['SMA_3000'] = df['Close'].rolling(window=3000).mean()
df['EMA_3000'] = df['Close'].ewm(span=3000).mean()

df['SMA_4000'] = df['Close'].rolling(window=4000).mean()
df['EMA_4000'] = df['Close'].ewm(span=4000).mean()

# Drop NaN values after calculating averages and ATR
df = df.dropna()

# Drop Date column as needed
df = df.drop(columns=['Date'])

# Initialize variables
initial_balance = 100  # Starting balance in USD
balance = initial_balance
position = 0  # Number of BTC held
entry_price = 0  # Entry price for the trade
transaction_log = []

# Mean Reversion Strategy Parameters
transaction_fee = 0.001  # 0.1% trading fee per trade
risk_percentage = 0.02  # Risk 2% of balance per trade

# Define function for position sizing
def calculate_position_size(balance, stop_loss, entry_price):
    risk_amount = balance * risk_percentage
    position_size = risk_amount / abs(entry_price - stop_loss)
    return position_size

# Step through each record to test the strategy
for index, row in df.iterrows():
    close_price = row['Close']
    sma_70 = row['SMA_70']
    ema_140 = row['EMA_140']
    sma_2000 = row['SMA_2000']
    atr = row['ATR']
    
    # Time-based filter (Only trade between 8 AM to 6 PM UTC)
    if 8 <= row['Hour'] <= 18:
        # Calculate dynamic threshold based on ATR
        threshold = atr / close_price
        
        # Define Stop Loss and Take Profit levels
        stop_loss = entry_price * (1 - 0.02)  # 2% stop loss
        take_profit = entry_price * (1 + 0.02)  # 2% take profit

        # Check for BUY signal: Price is below short-term SMA and EMA but above long-term SMA
        if (close_price < sma_70) and (close_price < ema_140) and (close_price > sma_2000) and position == 0:
            position_size = calculate_position_size(balance, stop_loss, close_price)
            position = position_size  # Buy BTC
            entry_price = close_price
            balance -= position * close_price * (1 + transaction_fee)
            transaction_log.append(f"BUY at {close_price:.2f} | Balance: {balance:.2f}")
        
        # Check for SELL signal: Price crosses above the short-term SMA/EMA and below long-term SMA
        elif (close_price > sma_70) and (close_price > ema_140) and position > 0:
            balance += position * close_price * (1 - transaction_fee)  # Sell BTC
            profit = (close_price - entry_price) * position
            transaction_log.append(f"SELL at {close_price:.2f} | Profit: {profit:.2f} | Balance: {balance:.2f}")
            position = 0
            entry_price = 0
        
        # Trailing Stop Logic
        if position > 0:
            if close_price <= stop_loss or close_price >= take_profit:
                balance += position * close_price * (1 - transaction_fee)
                transaction_log.append(f"STOPPED OUT at {close_price:.2f} | Balance: {balance:.2f}")
                position = 0
                entry_price = 0

# Final results
if position > 0:
    balance += position * df.iloc[-1]['Close'] * (1 - transaction_fee)
    transaction_log.append(f"Final SELL at {df.iloc[-1]['Close']:.2f} | Balance: {balance:.2f}")

# Profit/Loss Summary
net_profit = balance - initial_balance
print("\nTransaction Log:")
for log in transaction_log:
    print(log)

print(f"\nStarting Balance: ${initial_balance:.2f}")
print(f"Ending Balance: ${balance:.2f}")
print(f"Net Profit/Loss: ${net_profit:.2f}")