"""
Backtesting Script for TTM Squeeze Strategy with RVOL and Risk Management

This script performs a backtest of an advanced trading strategy based on the
TTM Squeeze indicator for a single stock ('NSE:HINDCOPPER'). It runs the
backtest on multiple intraday timeframes and generates a performance report
and a chart with trade overlays for each.

---
Strategy Logic:
1.  **Indicator Calculation**: The script calculates:
    -   Bollinger Bands (BB): 20-period SMA with 2 standard deviations.
    -   Keltner Channels (KC): 20-period EMA with ATR(14) bands multiplied by 2.0.
    -   Average True Range (ATR): 14-period, calculated using EMA for smoothing.
    -   Relative Volume (RVOL): Current volume divided by its 20-day simple moving average.

2.  **Squeeze Detection**: A "squeeze" is identified when the Bollinger Bands are
    fully contained within the Keltner Channels.

3.  **Trade Signal (Fired Event)**: A trade signal occurs when the squeeze ends,
    but only if the breakout has high volume.
    -   **Entry Condition**: A trade is only considered if the RVOL on the breakout
      day is greater than 1.
    -   **Bullish Breakout**: Close is above the upper BB, which is above the upper KC.
    -   **Bearish Breakout**: Close is below the lower BB, which is below the lower KC.

4.  **Trade Execution and Risk Management**:
    -   **Entry**: A position is entered at the opening price of the day *after* the
      signal day to avoid lookahead bias.
    -   **Risk (R)**: Risk is defined as the distance from the entry price to the
      opposite Bollinger Band on the signal day.
    -   **Stop-Loss**: Set at 1R from the entry price.
    -   **Take-Profit**: Set at 2R from the entry price.
    -   **Exit**: The trade is exited if the daily high/low touches either the
      stop-loss or take-profit level.

5.  **Visualization**:
    -   For each timeframe, a candlestick chart is generated showing the price
      action for the last 15 trades.
    -   Trade entries, exits, stop-loss, and take-profit levels are overlaid
      on the chart for visual analysis.

---
Assumptions:
-   **No Slippage or Commissions**: Trades are executed at the exact price levels.
-   **Full Capital Allocation**: Profit/loss is calculated on a per-share basis.
-   **Exit Priority**: On any given day, the take-profit level is checked before the
    stop-loss level, assuming a more favorable exit.
-   **Data Accuracy**: The historical data is assumed to be accurate.
"""

import pandas as pd
import mplfinance as mpf
import numpy as np
import os

def plot_trades(df, trades, symbol, timeframe):
    """
    Generates and saves a candlestick chart with trade overlays for the last 15 trades.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data.
        trades (pd.DataFrame): DataFrame of trades.
        symbol (str): The symbol that was backtested.
        timeframe (str): The timeframe of the backtest.
    """
    if trades.empty:
        print(f"No trades to plot for {symbol} on {timeframe}.")
        return

    # --- Prepare data for plotting ---
    df_plot = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

    trades_to_plot = trades.tail(15)

    if not trades_to_plot.empty:
        first_trade_date = trades_to_plot['entry_date'].min()
        last_trade_date = trades_to_plot['exit_date'].max()
        plot_start_date = first_trade_date - pd.Timedelta(days=3)
        plot_end_date = last_trade_date + pd.Timedelta(days=3)
        df_plot = df_plot.loc[plot_start_date:plot_end_date]

    if df_plot.empty:
        print(f"No data in the plotting window for {timeframe}. Skipping plot.")
        return

    # --- Create markers and lines for plotting ---
    addplots = []

    long_entries = pd.Series(np.nan, index=df_plot.index)
    short_entries = pd.Series(np.nan, index=df_plot.index)
    exits = pd.Series(np.nan, index=df_plot.index)

    for trade in trades_to_plot.itertuples():
        if trade.entry_date in df_plot.index:
            if trade.direction == 'long':
                long_entries.loc[trade.entry_date] = df_plot.loc[trade.entry_date]['Low'] * 0.99
            else:
                short_entries.loc[trade.entry_date] = df_plot.loc[trade.entry_date]['High'] * 1.01

        if trade.exit_date in df_plot.index:
            exits.loc[trade.exit_date] = df_plot.loc[trade.exit_date]['Close']

        sl_series = pd.Series(np.nan, index=df_plot.index)
        tp_series = pd.Series(np.nan, index=df_plot.index)

        trade_duration_idx = df_plot.loc[trade.entry_date:trade.exit_date].index

        sl_series.loc[trade_duration_idx] = trade.stop_loss
        tp_series.loc[trade_duration_idx] = trade.take_profit

        addplots.append(mpf.make_addplot(sl_series, color='red', linestyle='--', width=0.8))
        addplots.append(mpf.make_addplot(tp_series, color='green', linestyle='--', width=0.8))

    addplots.append(mpf.make_addplot(long_entries, type='scatter', marker='^', color='green', markersize=150))
    addplots.append(mpf.make_addplot(short_entries, type='scatter', marker='v', color='red', markersize=150))
    addplots.append(mpf.make_addplot(exits, type='scatter', marker='x', color='blue', markersize=100))

    # --- Generate and Save Plot ---
    charts_dir = "charts"
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    filename = os.path.join(charts_dir, f"{symbol}_{timeframe}_trades.png")

    try:
        mpf.plot(df_plot, type='candle', style='yahoo',
                 title=f'Trades for {symbol} ({timeframe})',
                 addplot=addplots,
                 figsize=(20, 10),
                 savefig=filename)
        print(f"Chart saved to {filename}")
    except Exception as e:
        print(f"Could not generate plot for {symbol} on {timeframe}. Error: {e}")

def load_and_resample_data(symbol, timeframe_minutes):
    """
    Loads 1-minute data for a symbol and resamples it to the specified timeframe.

    Args:
        symbol (str): The stock symbol (e.g., 'TCS').
        timeframe_minutes (int): The target timeframe in minutes (e.g., 5, 15).

    Returns:
        pd.DataFrame: Resampled OHLCV data, or None if file is not found.
    """
    filename = f"{symbol}_minute.csv"
    if not os.path.exists(filename):
        print(f"Data file not found: {filename}")
        return None

    print(f"Loading data from {filename} and resampling to {timeframe_minutes} minutes...")
    df = pd.read_csv(filename, index_col='date', parse_dates=True)

    resample_period = f'{timeframe_minutes}min'

    resampled_df = df.resample(resample_period).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled_df

def calculate_indicators(df, atr_period=14, bb_period=20, kc_period=20, kc_multiplier=2.0):
    """
    Calculates ATR, Bollinger Bands, Keltner Channels, and RVOL for the given DataFrame.
    """
    df['high_low'] = df['high'] - df['low']
    df['high_prev_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_prev_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/atr_period, adjust=False).mean()

    df['bb_sma'] = df['close'].rolling(window=bb_period).mean()
    df['bb_std'] = df['close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)

    df['kc_sma'] = df['close'].rolling(window=kc_period).mean()
    df['kc_upper'] = df['kc_sma'] + (df['atr'] * kc_multiplier)
    df['kc_lower'] = df['kc_sma'] - (df['atr'] * kc_multiplier)

    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['rvol'] = df['volume'] / df['avg_volume']

    df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'tr', 'avg_volume'], axis=1, inplace=True)

    return df

def run_backtest(df):
    """
    Runs the backtest based on the TTM Squeeze "fired" signal, incorporating RVOL
    and a dynamic 1R/2R stop-loss/take-profit exit strategy.
    """
    trades = []
    in_position = False
    trade_details = {}

    df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

    for i in range(1, len(df)):
        if in_position:
            current_low = df['low'].iloc[i]
            current_high = df['high'].iloc[i]
            exit_price = None

            if trade_details['direction'] == 'long':
                if current_high >= trade_details['take_profit']:
                    exit_price = trade_details['take_profit']
                elif current_low <= trade_details['stop_loss']:
                    exit_price = trade_details['stop_loss']

            elif trade_details['direction'] == 'short':
                if current_low <= trade_details['take_profit']:
                    exit_price = trade_details['take_profit']
                elif current_high >= trade_details['stop_loss']:
                    exit_price = trade_details['stop_loss']

            if exit_price is not None:
                if trade_details['direction'] == 'long':
                    profit_loss = exit_price - trade_details['entry_price']
                else:
                    profit_loss = trade_details['entry_price'] - exit_price

                trade_details.update({
                    'exit_date': df.index[i],
                    'exit_price': exit_price,
                    'profit_loss': profit_loss
                })
                trades.append(trade_details.copy())

                in_position = False
                trade_details = {}

        if not in_position:
            squeeze_fired = df['squeeze_on'].iloc[i-1] and not df['squeeze_on'].iloc[i]

            if squeeze_fired and df['rvol'].iloc[i] > 1:
                is_bullish = df['close'].iloc[i] > df['bb_upper'].iloc[i] and df['bb_upper'].iloc[i] > df['kc_upper'].iloc[i]
                is_bearish = df['close'].iloc[i] < df['bb_lower'].iloc[i] and df['bb_lower'].iloc[i] < df['kc_lower'].iloc[i]

                direction = 'long' if is_bullish else 'short' if is_bearish else None

                if direction and i + 1 < len(df):
                    entry_price = df['open'].iloc[i+1]

                    if direction == 'long':
                        risk = entry_price - df['bb_lower'].iloc[i]
                        stop_loss = entry_price - risk
                        take_profit = entry_price + (2 * risk)
                    else:
                        risk = df['bb_upper'].iloc[i] - entry_price
                        stop_loss = entry_price + risk
                        take_profit = entry_price - (2 * risk)

                    if risk <= 0: continue

                    in_position = True
                    trade_details = {
                        'entry_date': df.index[i+1],
                        'entry_price': entry_price,
                        'direction': direction,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'risk': risk,
                        'exit_date': None, 'exit_price': None, 'profit_loss': None
                    }

    return pd.DataFrame([t for t in trades if t.get('exit_date') is not None])

def calculate_performance_metrics(trades, initial_capital=100000.0):
    """
    Calculates key performance metrics for all trades, and for long and short trades separately.
    """
    if trades.empty or trades['profit_loss'].isnull().any():
        return {'Overall': {}, 'Long': {}, 'Short': {}}

    results = {}

    def _calculate_subset_metrics(trade_subset):
        if trade_subset.empty: return {'Total Trades': 0, 'Win Rate (%)': 0, 'Total Profit/Loss ($)': 0}
        total = len(trade_subset)
        wins = trade_subset[trade_subset['profit_loss'] > 0]
        return {
            'Total Trades': total,
            'Win Rate (%)': (len(wins) / total) * 100,
            'Total Profit/Loss ($)': trade_subset['profit_loss'].sum()
        }

    results['Overall'] = _calculate_subset_metrics(trades)
    results['Long'] = _calculate_subset_metrics(trades[trades['direction'] == 'long'])
    results['Short'] = _calculate_subset_metrics(trades[trades['direction'] == 'short'])

    results['Overall']['Average Profit per Trade'] = trades['profit_loss'].mean()

    trades['equity_curve'] = initial_capital + trades['profit_loss'].cumsum()
    trades['running_max'] = trades['equity_curve'].cummax()
    trades['drawdown'] = trades['running_max'] - trades['equity_curve']
    max_drawdown_value = trades['drawdown'].max()
    results['Overall']['Maximum Drawdown (%)'] = (max_drawdown_value / trades['running_max'].max()) * 100 if trades['running_max'].max() > 0 else 0

    return results

def run_backtest_for_timeframe(symbol, timeframe_minutes):
    """
    Encapsulates the entire backtesting process for a single symbol and timeframe.
    """
    df = load_and_resample_data(symbol, timeframe_minutes)
    if df is None:
        return None, None

    df = calculate_indicators(df)
    df.dropna(inplace=True)

    trades = run_backtest(df)
    timeframe_str = f"{timeframe_minutes}min"

    if not trades.empty:
        performance_metrics = calculate_performance_metrics(trades)
        plot_trades(df, trades, symbol, timeframe_str)
        return performance_metrics, trades
    else:
        print(f"\nNo trades were executed for {symbol} on {timeframe_str}.")
        return None, None

def generate_consolidated_report(all_metrics, symbol):
    """
    Generates a consolidated report from all backtest runs and saves it to a file.
    """
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    filename = os.path.join(reports_dir, f"{symbol}_consolidated_report.txt")

    with open(filename, 'w') as f:
        f.write(f"--- Consolidated Backtest Performance Report for {symbol} ---\n")
        f.write("=" * 60 + "\n")

        for timeframe, metrics in all_metrics.items():
            f.write(f"\n--- Timeframe: {timeframe} ---\n")
            if not metrics:
                f.write("No trades to analyze.\n")
                continue

            def _write_subset(name, data):
                f.write(f"\n{name} Performance:\n")
                if data and data['Total Trades'] > 0:
                    f.write(f"  Total Trades: {data['Total Trades']}\n")
                    f.write(f"  Win Rate: {data['Win Rate (%)']:.2f}%\n")
                    f.write(f"  Total Profit/Loss: ${data['Total Profit/Loss ($)']:.2f}\n")
                    if 'Average Profit per Trade' in data:
                        f.write(f"  Average Profit per Trade: ${data['Average Profit per Trade']:.2f}\n")
                    if 'Maximum Drawdown (%)' in data:
                        f.write(f"  Maximum Drawdown: {data['Maximum Drawdown (%)']:.2f}%\n")
                else:
                    f.write("  No trades to analyze for this subset.\n")

            _write_subset("Overall", metrics.get('Overall', {}))
            _write_subset("Long", metrics.get('Long', {}))
            _write_subset("Short", metrics.get('Short', {}))
            f.write("-" * 40 + "\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("--- End of Report ---\n")

    print(f"Consolidated report saved to {filename}")


if __name__ == "__main__":
    # --- Parameters ---
    SYMBOL = "TCS"  # The symbol for which to run the backtest.
                    # Requires a corresponding 'TCS_minute.csv' file.

    # Define the list of timeframes (in minutes) to test.
    timeframes_to_test = [5, 15, 30]

    all_metrics = {}
    print(f"--- Starting Consolidated Backtest for {SYMBOL} ---")

    # Loop through each timeframe and run the full backtest process.
    for timeframe in timeframes_to_test:
        print(f"\nRunning backtest on {timeframe}-minute timeframe...")
        metrics, trades = run_backtest_for_timeframe(SYMBOL, timeframe)
        if metrics:
            all_metrics[f"{timeframe}min"] = metrics

    # After testing all timeframes, generate one consolidated report.
    if all_metrics:
        generate_consolidated_report(all_metrics, SYMBOL)
    else:
        print(f"\nNo trades were executed for {SYMBOL} across any timeframe. No report to generate.")

    print(f"\n--- Consolidated Backtest for {SYMBOL} Complete ---")