import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy, linregress


class TradingStrategy:
    def __init__(self, ticker, start_date, end_date, interval='1h', chunk_size_days=60):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.chunk_size_days = chunk_size_days
        self.data = self.fetch_data_in_chunks()
        self.calculate_indicators()
        self.generate_signals()
        
        
        
    def fetch_data_in_chunks(self):
        data = pd.DataFrame()
        current_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)

        while current_date < end_date:
            next_date = min(current_date + pd.Timedelta(days=self.chunk_size_days), end_date)
            chunk_data = yf.download(self.ticker, start=current_date.strftime('%Y-%m-%d'), end=next_date.strftime('%Y-%m-%d'), interval=self.interval)
            data = pd.concat([data, chunk_data])
            current_date = next_date

        return data[['Open', 'Close']]

    def calculate_indicators(self):
        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['SMA200'] = self.data['Close'].rolling(window=200).mean()
        self.data['Entropy'] = self.Shannon_Entropy()
        self.Normal_Distribution()
        self.Linear_Regression()
        
        self.data.dropna(inplace=True)

    def Normal_Distribution(self, window=100):
        self.data['LogReturn'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        self.data['RollingMean'] = self.data['LogReturn'].rolling(window=window).mean()
        self.data['RollingStd'] = self.data['LogReturn'].rolling(window=window).std()
        
        self.data['UpperBound'] = self.data['RollingMean'] + (self.data['RollingStd'] * 1.2)
        self.data['LowerBound'] = self.data['RollingMean'] - (self.data['RollingStd'] * 1.2)

    def Shannon_Entropy(self, window=50):
        log_returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
        rolling_entropy = log_returns.rolling(window=window).apply(
            lambda x: entropy(np.histogram(x, bins=10, density=True)[0]), raw=False
        )
        return rolling_entropy
    
    def Linear_Regression(self, window=60):
        self.data['Predicted'] = np.nan
        self.data['PMCC'] = np.nan
        
        for i in range(window + 1, len(self.data)):
            y = self.data['Close'].iloc[i-window - 1:i - 1].values
            x = np.arange(window)
            gradient, intercept, r_value, p_value, std_err = linregress(x, y)
            self.data.loc[self.data.index[i], 'Predicted'] = intercept + gradient * window
            self.data.loc[self.data.index[i], 'PMCC'] = r_value
            self.data.loc[self.data.index[i], 'Gradient'] = gradient

    def generate_signals(self):
        self.data['Signal'] = 0
        threshold_entropy = self.data['Entropy'].mean()
        threshold_gradient = 0.0002
        for i in range(1, len(self.data)):
            pmcc = self.data['PMCC'].iloc[i]
            
            #5% significant level
            if abs(pmcc) <= 0.2353:
                continue

            if pmcc >= 0.5 and (self.data['LogReturn'].iloc[i] <= (self.data['LowerBound'].iloc[i-1])) and self.data['Entropy'].iloc[i] <= (threshold_entropy * 1.05):
                if self.data['Gradient'].iloc[i] >= threshold_gradient:
                    self.data.at[self.data.index[i], 'Signal'] = 1  # Buy signal
            elif pmcc <= 0 and self.data['LogReturn'].iloc[i] >= self.data['UpperBound'].iloc[i-1] and self.data['Entropy'].iloc[i] >= (threshold_entropy):
                if self.data['Gradient'].iloc[i] <= 0:
                    self.data.at[self.data.index[i], 'Signal'] = -1  # Sell signal

    def simulate_trading(self, initial_balance=10000, share_size=10):
        balance = initial_balance
        position = 0
        trades = []
        buy_sell_pairs = []

        for index, row in self.data.iterrows():
            if row['Signal'] == 1 and position == 0:  # Buy signal
                position += share_size
                balance -= row['Close'] * share_size
                trades.append((index, 'Buy', share_size, row['Close'], balance))
            elif row['Signal'] == -1 and position > 0:  # Sell signal
                balance += position * row['Close']
                trades.append((index, 'Sell', position, row['Close'], balance))
                buy_sell_pairs.append((trades[-2], trades[-1]))
                position = 0

        final_balance = balance + position * self.data.iloc[-1]['Close']
        return final_balance, trades, buy_sell_pairs

    def plot_results(self, final_balance, trades, buy_sell_pairs):
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(6, 6), sharex=True)

        # Plot market prices
        ax1.plot(self.data['Close'], label='Close Price')
        ax1.plot(self.data['SMA50'], label='50-period SMA')
        ax1.plot(self.data['SMA200'], label='200-period SMA')

        # Plot buy/sell pairs with holding periods
        buy_signal_added = False
        sell_signal_added = False
        for buy, sell in buy_sell_pairs:
            buy_date, buy_action, buy_shares, buy_price, buy_balance = buy
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

        ax1.set_title('Trading Strategy Simulation')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid()

        # Plot entropy values
        threshold_entropy = self.data['Entropy'].mean()
        ax2.plot(self.data['Entropy'], label='Entropy')
        ax2.axhline(threshold_entropy, linestyle='--', alpha=0.5, color='red')

        for i, (buy, sell) in enumerate(buy_sell_pairs):
            buy_date, buy_action, buy_shares, buy_price, buy_balance = buy
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
        
        # Plot rolling mean, bounds, and log returns
        ax3.plot(self.data['LogReturn'], label='Log Return', alpha=0.7)
        ax3.plot(self.data['RollingMean'], label='Rolling Mean')
        ax3.plot(self.data['UpperBound'], label='Upper Bound', linestyle='--')
        ax3.plot(self.data['LowerBound'], label='Lower Bound', linestyle='--')
        for i, (buy, sell) in enumerate(buy_sell_pairs):
            buy_date, buy_action, buy_shares, buy_price, buy_balance = buy
            sell_date, sell_action, sell_shares, sell_price, sell_balance = sell
            if i == 0:
                ax3.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8, label='Buy Signal')
                ax3.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8, label='Sell Signal')
            else:
                ax3.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8)
                ax3.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8)

        ax3.set_title('Rolling Mean, Bounds, and Log Returns')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Log Return')
        ax3.legend()
        ax3.grid()
        
        # Plot predicted values and PMCC
        ax4.plot(self.data['Close'], label='Close Price')
        ax4.plot(self.data['Predicted'], label='Predicted Price', linestyle='--')
        ax4.set_title('Close Price and Predicted Price')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price')
        for i, (buy, sell) in enumerate(buy_sell_pairs):
            buy_date, buy_action, buy_shares, buy_price, buy_balance = buy
            sell_date, sell_action, sell_shares, sell_price, sell_balance = sell
            if i == 0:
                ax4.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8, label='Buy Signal')
                ax4.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8, label='Sell Signal')
            else:
                ax4.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8)
                ax4.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8)

        ax4.legend()
        ax4.grid()
        
        ax5.plot(self.data['PMCC'], label='PMCC')
        ax5.set_title('PMCC Values')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('PMCC')
        for i, (buy, sell) in enumerate(buy_sell_pairs):
            buy_date, buy_action, buy_shares, buy_price, buy_balance = buy
            sell_date, sell_action, sell_shares, sell_price, sell_balance = sell
            if i == 0:
                ax5.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8, label='Buy Signal')
                ax5.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8, label='Sell Signal')
            else:
                ax5.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8)
                ax5.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8)
        ax5.legend()
        ax5.grid()

        ax6.plot(self.data['Gradient'], label='Gradient')
        ax6.set_title('Gradient Values')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Gradient')
        for i, (buy, sell) in enumerate(buy_sell_pairs):
            buy_date, buy_action, buy_shares, buy_price, buy_balance = buy
            sell_date, sell_action, sell_shares, sell_price, sell_balance = sell
            if i == 0:
                ax6.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8, label='Buy Signal')
                ax6.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8, label='Sell Signal')
            else:
                ax6.axvline(x=buy_date, color='g', linestyle='--', linewidth=0.8)
                ax6.axvline(x=sell_date, color='r', linestyle='--', linewidth=0.8)
        ax6.legend()
        ax6.grid()


        plt.show()

if __name__ == "__main__":
    strategy = TradingStrategy(ticker='EURUSD=X', start_date='2023-06-17', end_date='2024-06-18')
    final_balance, trades, buy_sell_pairs = strategy.simulate_trading()

    # Calculate average holding time
    holding_times = [(sell[0] - buy[0]).total_seconds() / 3600 for buy, sell in buy_sell_pairs]  # Convert to hours
    average_holding_time = sum(holding_times) / len(holding_times) if holding_times else 0

    # Print average holding time
    print(f"Average holding time: {average_holding_time:.2f} hours")

    # Print number of shares bought per trade
    shares_per_trade = [trade for trade in trades if trade[1] == 'Buy']
    print(f"Shares bought per trade: {shares_per_trade}")

    # Evaluate performance
    initial_balance = 10000
    print(f"Initial balance: ${initial_balance:.2f}")
    print(f"Final balance: ${final_balance:.2f}")

    # Plot the results
    strategy.plot_results(final_balance, trades, buy_sell_pairs)
