import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

# Function to fetch data in chunks
def fetch_data_in_chunks(ticker, start_date, end_date, interval, chunk_size_days=60):
    data = pd.DataFrame()
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    while current_date < end_date:
        next_date = min(current_date + pd.Timedelta(days=chunk_size_days), end_date)
        chunk_data = yf.download(ticker, start=current_date.strftime('%Y-%m-%d'), end=next_date.strftime('%Y-%m-%d'), interval=interval)
        data = pd.concat([data, chunk_data])
        current_date = next_date

    return data


# Fetch historical 1-hour data
ticker = 'EURUSD=X'
start_date = '2023-06-17'
end_date = '2024-06-18'
interval = '1h'
data = fetch_data_in_chunks(ticker, start_date, end_date, interval)
data = data[['Close']]

#technical indicators
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Function to calculate Shannon entropy over a rolling window
def Shannon_entropy(prices, window=50):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    rolling_entropy = log_returns.rolling(window=window).apply(
        lambda x: entropy(np.histogram(x, bins=10, density=True)[0]), raw=False
    )
    return rolling_entropy

# Calculate entropy for the 'Close' prices
data['Entropy'] = Shannon_entropy(data['Close'])

# Drop rows with NaN values to avoid lookahead bias
data = data.dropna()

# Initialize the signal column
data['Signal'] = 0

# Set thresholds
threshold_entropy = data['Entropy'].mean()
print(data)
print(len)
# Generate signals
for i in range(1, len(data)):
    if data['SMA50'].iloc[i-1] < data['SMA200'].iloc[i-1] and data['SMA50'].iloc[i] > data['SMA200'].iloc[i] and data['Entropy'].iloc[i] < threshold_entropy:
        data.at[data.index[i], 'Signal'] = 1  # Buy signal
    elif data['SMA50'].iloc[i-1] > data['SMA200'].iloc[i-1] and data['SMA50'].iloc[i] < data['SMA200'].iloc[i] and data['Entropy'].iloc[i] >= (threshold_entropy * 1.3):
        data.at[data.index[i], 'Signal'] = -1  # Sell signal

# Simulate trading and track buy/sell pairs with stop loss and take profit
def simulate_trading(data, initial_balance=10000, share_size=10, risk_unit=0.01):
    balance = initial_balance
    position = 0
    trades = []
    buy_sell_pairs = []
    stop_loss = None
    take_profit = None

    for index, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:  # Buy signal
            position += share_size
            balance -= row['Close'] * share_size
            stop_loss = row['Close'] * (1 - risk_unit)
            take_profit = row['Close'] * (1 + 3 * risk_unit)
            trades.append((index, 'Buy', share_size, row['Close'], balance, stop_loss, take_profit))
        elif row['Signal'] == -1 and position > 0:  # Sell signal
            balance += position * row['Close']
            trades.append((index, 'Sell', position, row['Close'], balance))
            buy_sell_pairs.append((trades[-2], trades[-1]))
            position = 0
            stop_loss = None
            take_profit = None
        elif position > 0 and (row['Close'] <= stop_loss or row['Close'] >= take_profit):  # Stop loss or take profit triggered
            balance += position * row['Close']
            trades.append((index, 'Sell', position, row['Close'], balance))
            buy_sell_pairs.append((trades[-2], trades[-1]))
            position = 0
            stop_loss = None
            take_profit = None

    final_balance = balance + position * data.iloc[-1]['Close']
    return final_balance, trades, buy_sell_pairs

final_balance, trades, buy_sell_pairs = simulate_trading(data)

# Calculate average holding time
holding_times = [(sell[0] - buy[0]).total_seconds() / 3600 for buy, sell in buy_sell_pairs]  # Convert to hours
average_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

# Print average holding time
print(f"Average holding time: {average_holding_time:.2f} hours")

# Print number of shares bought per trade
shares_per_trade = [trade[2] for trade in trades if trade[1] == 'Buy']
print(f"Shares bought per trade: {shares_per_trade}")

# Evaluate performance
initial_balance = 10000
print(f"Initial balance: ${initial_balance:.2f}")
print(f"Final balance: ${final_balance:.2f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True)

# Plot market prices
ax1.plot(data['Close'], label='Close Price')
ax1.plot(data['SMA50'], label='50-period SMA')
ax1.plot(data['SMA200'], label='200-period SMA')

# Plot buy/sell pairs with holding periods
buy_signal_added = False
sell_signal_added = False
for buy, sell in buy_sell_pairs:
    buy_date, buy_action, buy_shares, buy_price, buy_balance, buy_stop_loss, buy_take_profit = buy
    sell_date, sell_action, sell_shares, sell_price, sell_balance = sell
    if not buy_signal_added:
        ax1.plot(buy_date, buy_price, '^', markersize=10, color='g', label='Buy Signal')
        buy_signal_added = True
    else:
        ax1.plot(buy_date, buy_price, '^', markersize=10, color='g')
    if not sell_signal_added:
        ax1.plot(sell_date, sell_price, 'v', markersize=10, color='r', label='Sell Signal')
        sell_signal_added = True
    else:
        ax1.plot(sell_date, sell_price, 'v', markersize=10, color='r')
    ax1.plot([buy_date, sell_date], [buy_price, sell_price], color='black', linestyle='--', linewidth=1)
    # Plot stop loss and take profit levels
    ax1.axhline(y=buy_stop_loss, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(y=buy_take_profit, color='green', linestyle='--', alpha=0.7)

ax1.set_title('Trading Strategy Simulation with SMA Crossover and Entropy Filter')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid()

# Plot entropy values
ax2.plot(data['Entropy'], label='Entropy')
ax2.axhline(threshold_entropy, linestyle='--', alpha=0.5, color='red')

for i, (buy, sell) in enumerate(buy_sell_pairs):
    buy_date, buy_action, buy_shares, buy_price, buy_balance, buy_stop_loss, buy_take_profit = buy
    sell_date, sell_action, sell_shares, sell_price, sell_balance = sell
    if i == 0:
        ax2.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8, label='Buy Signal')
        ax2.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8, label='Sell Signal')
    else:
        ax2.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8)
        ax2.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8)

ax2.set_title('Entropy with Buy/Sell Signals')
ax2.set_xlabel('Date')
ax2.set_ylabel('Entropy')
ax2.legend()
ax2.grid()

plt.show()
