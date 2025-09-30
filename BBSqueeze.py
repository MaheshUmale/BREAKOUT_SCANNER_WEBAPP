import os
import traceback
import urllib.parse
import json
from time import sleep
from datetime import datetime, timedelta
import numpy as np
import sqlite3
from tradingview_screener import Query, col, And, Or
import pandas as pd

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()  # Or obj.strftime('%Y-%m-%d %H:%M:%S')
        return json.JSONEncoder.default(self, obj)


# --- SQLite Timestamp Handling ---
def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 format."""
    return val.isoformat()

def convert_timestamp(val):
    """Convert ISO 8601 string to datetime.datetime object."""
    return datetime.fromisoformat(val.decode())

# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp)
sqlite3.register_converter("Timestamp", convert_timestamp)


pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


def append_df_to_csv(df, csv_path):
    """
    Appends a DataFrame to a CSV file. Creates the file with a header if it doesn't
    exist, otherwise appends without the header.
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


def generate_heatmap_json(df, output_path):
    """
    Generates a simple, flat JSON array of stock data for the D3 heatmap.
    """
    # Ensure required columns exist for JSON generation
    base_required_cols = ['ticker', 'HeatmapScore', 'SqueezeCount', 'rvol', 'URL', 'logo', 'momentum', 'highest_tf', 'squeeze_strength']
    for c in base_required_cols:
        if c not in df.columns:
            # Provide a default value if a column is missing
            if c == 'momentum':
                df[c] = 'Neutral'
            elif c == 'highest_tf' or c == 'squeeze_strength':
                df[c] = 'N/A'
            else:
                df[c] = 0

    # Create a flat list of stock data
    heatmap_data = []
    for _, row in df.iterrows():
        stock_data = {
            "name": row['ticker'],
            "value": row['HeatmapScore'],
            "count": row['SqueezeCount'],
            "rvol": row['rvol'],
            "url": row['URL'],
            "logo": row['logo'],
            "momentum": row['momentum'],
            "highest_tf": row['highest_tf'],
            "squeeze_strength": row['squeeze_strength']
        }
        # Add optional volatility data if present in the DataFrame (for fired squeezes)
        if 'fired_timeframe' in df.columns:
            stock_data['fired_timeframe'] = row['fired_timeframe']
        if 'fired_timestamp' in df.columns:
             stock_data['fired_timestamp'] = row['fired_timestamp']
        if 'previous_volatility' in df.columns:
            stock_data['previous_volatility'] = row['previous_volatility']
        if 'current_volatility' in df.columns:
            stock_data['current_volatility'] = row['current_volatility']
        if 'volatility_increased' in df.columns:
            stock_data['volatility_increased'] = row['volatility_increased']

        heatmap_data.append(stock_data)

    # Write the JSON file
    with open(output_path, 'w') as f:
        json.dump(heatmap_data, f, indent=4, cls=CustomJSONEncoder)
    print(f"✅ Flat JSON successfully generated at '{output_path}'.")


# --- Configuration ---
DB_FILE = 'squeeze_history.db'
OUTPUT_JSON_FIRED = 'treemap_data_fired.json'
OUTPUT_JSON_FORMED = 'treemap_data_formed.json'
OUTPUT_JSON_IN_SQUEEZE = 'treemap_data_in_squeeze.json'
TIME_INTERVAL_SECONDS = 120

# --- Timeframe Configuration ---
timeframes = ['', '|1', '|5', '|15', '|30', '|60', '|120', '|240', '|1W', '|1M']
tf_order_map = {
    '|1M': 10, '|1W': 9, '|240': 8, '|120': 7, '|60': 6,
    '|30': 5, '|15': 4, '|5': 3, '|1': 2, '': 1
}
tf_display_map = {
    '': 'Daily', '|1': '1m', '|5': '5m', '|15': '15m', '|30': '30m',
    '|60': '1H', '|120': '2H', '|240': '4H', '|1W': 'Weekly', '|1M': 'Monthly'
}
tf_suffix_map = {v: k for k, v in tf_display_map.items()}

# Construct select columns for all timeframes
select_cols = ['name', 'logoid', 'close', 'MACD.hist']
for tf in timeframes:
    select_cols.extend([
        f'KltChnl.lower{tf}', f'KltChnl.upper{tf}',
        f'BB.lower{tf}', f'BB.upper{tf}',
        f'ATR{tf}', f'SMA20{tf}',
        f'volume{tf}', f'average_volume_10d_calc{tf}',
        f'Value.Traded{tf}'
    ])


def get_highest_squeeze_tf(row):
    for tf_suffix in sorted(tf_order_map, key=tf_order_map.get, reverse=True):
        if row.get(f'InSqueeze{tf_suffix}', False):
            return tf_display_map[tf_suffix]
    return 'Unknown'

def get_dynamic_rvol(row, timeframe_name, tf_suffix_map):
    """Calculates RVOL for a specific timeframe."""
    tf_suffix = tf_suffix_map.get(timeframe_name)
    if tf_suffix is None:
        return 0

    vol_col = f'volume{tf_suffix}'
    avg_vol_col = f'average_volume_10d_calc{tf_suffix}'

    volume = row.get(vol_col)
    avg_volume = row.get(avg_vol_col)

    if pd.isna(volume) or pd.isna(avg_volume) or avg_volume == 0:
        return 0

    return volume / avg_volume


def get_squeeze_strength(row):
    """Calculates and categorizes the strength of the squeeze."""
    highest_tf_name = row['highest_tf']
    tf_suffix = tf_suffix_map.get(highest_tf_name)

    if tf_suffix is None:
        return "N/A"

    bb_upper = row.get(f'BB.upper{tf_suffix}')
    bb_lower = row.get(f'BB.lower{tf_suffix}')
    kc_upper = row.get(f'KltChnl.upper{tf_suffix}')
    kc_lower = row.get(f'KltChnl.lower{tf_suffix}')

    if any(pd.isna(val) for val in [bb_upper, bb_lower, kc_upper, kc_lower]):
        return "N/A"

    bb_width = bb_upper - bb_lower
    kc_width = kc_upper - kc_lower

    if bb_width == 0:
        return "N/A" # Avoid division by zero

    sqz_strength = kc_width / bb_width

    if sqz_strength >= 2:
        return "VERY STRONG"
    elif sqz_strength >= 1.5:
        return "STRONG"
    elif sqz_strength > 1:
        return "Regular"
    else:
        return "N/A"


def get_fired_breakout_direction(row, fired_tf_name, tf_suffix_map):
    """
    Determines the breakout direction for a fired squeeze based on price action
    relative to Bollinger Bands and Keltner Channels.
    """
    tf_suffix = tf_suffix_map.get(fired_tf_name)
    if not tf_suffix:
        return 'Neutral'

    close = row.get('close')
    bb_upper = row.get(f'BB.upper{tf_suffix}')
    kc_upper = row.get(f'KltChnl.upper{tf_suffix}')
    bb_lower = row.get(f'BB.lower{tf_suffix}')
    kc_lower = row.get(f'KltChnl.lower{tf_suffix}')

    if any(pd.isna(val) for val in [close, bb_upper, kc_upper, bb_lower, kc_lower]):
        return 'Neutral'

    # Positive breakout: Close is above the upper Bollinger Band, which is above the upper Keltner Channel.
    if close > bb_upper and bb_upper > kc_upper:
        return 'Bullish'
    # Negative breakout: Close is below the lower Bollinger Band, which is below the lower Keltner Channel.
    elif close < bb_lower and bb_lower < kc_lower:
        return 'Bearish'
    else:
        return 'Neutral'


def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS squeeze_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_timestamp TIMESTAMP NOT NULL,
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            volatility REAL,
            rvol REAL,
            SqueezeCount INTEGER,
            squeeze_strength TEXT,
            HeatmapScore REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fired_squeeze_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fired_timestamp TIMESTAMP NOT NULL,
            ticker TEXT NOT NULL,
            fired_timeframe TEXT NOT NULL,
            momentum TEXT,
            previous_volatility REAL,
            current_volatility REAL,
            rvol REAL,
            HeatmapScore REAL,
            URL TEXT,
            logo TEXT
        )
    ''')
    # Create indexes for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON squeeze_history (scan_timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fired_timestamp ON fired_squeeze_events (fired_timestamp)')
    conn.commit()
    conn.close()


def load_previous_squeeze_list_from_db():
    """Loads the list of (ticker, timeframe, volatility) tuples from the most recent scan."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT MAX(scan_timestamp) FROM squeeze_history')
        last_timestamp = cursor.fetchone()[0]
        if last_timestamp is None: return []
        cursor.execute('SELECT ticker, timeframe, volatility FROM squeeze_history WHERE scan_timestamp = ?', (last_timestamp,))
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    finally:
        conn.close()


def save_current_squeeze_list_to_db(squeeze_records):
    """Saves the current list of squeeze data to the database."""
    if not squeeze_records: return
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    now = datetime.now()
    data_to_insert = [
        (
            now,
            r['ticker'],
            r['timeframe'],
            r['volatility'],
            r.get('rvol'),
            r.get('SqueezeCount'),
            r.get('squeeze_strength'),
            r.get('HeatmapScore')
        )
        for r in squeeze_records
    ]
    cursor.executemany(
        'INSERT INTO squeeze_history (scan_timestamp, ticker, timeframe, volatility, rvol, SqueezeCount, squeeze_strength, HeatmapScore) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        data_to_insert
    )
    conn.commit()
    conn.close()

def save_fired_events_to_db(fired_events_df):
    """Saves the processed fired squeeze events to the database."""
    if fired_events_df.empty: return
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    now = datetime.now()
    data_to_insert = [
        (now, row['ticker'], row['fired_timeframe'], row.get('momentum'), row.get('previous_volatility'),
         row.get('current_volatility'), row.get('rvol'), row.get('HeatmapScore'), row.get('URL'), row.get('logo'))
        for _, row in fired_events_df.iterrows()
    ]
    cursor.executemany('''
        INSERT INTO fired_squeeze_events (
            fired_timestamp, ticker, fired_timeframe, momentum, previous_volatility,
            current_volatility, rvol, HeatmapScore, URL, logo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    conn.commit()
    conn.close()
    print(f"✅ Saved {len(data_to_insert)} newly fired events to the database.")

def cleanup_old_fired_events():
    """Removes fired squeeze events older than 15 minutes from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    fifteen_minutes_ago = datetime.now() - timedelta(minutes=15)
    cursor.execute("DELETE FROM fired_squeeze_events WHERE fired_timestamp < ?", (fifteen_minutes_ago,))
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted_rows > 0:
        print(f"🧹 Cleaned up {deleted_rows} old fired events from the database.")

def load_recent_fired_events_from_db():
    """Loads all fired squeeze events from the last 15 minutes."""
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    fifteen_minutes_ago = datetime.now() - timedelta(minutes=15)
    query = "SELECT * FROM fired_squeeze_events WHERE fired_timestamp >= ?"
    df = pd.read_sql_query(query, conn, params=(fifteen_minutes_ago,))
    conn.close()
    print(f"Loaded {len(df)} recent fired events from the database.")
    return df


# --- Main Loop ---
init_db()
while True:
    try:
        # 1. Load previous squeeze state
        prev_squeeze_pairs = load_previous_squeeze_list_from_db()
        print(f"Loaded {len(prev_squeeze_pairs)} (ticker, timeframe) pairs from the previous scan.")

        # 2. Find all stocks currently in a squeeze
        squeeze_conditions = [And(col(f'BB.upper{tf}') < col(f'KltChnl.upper{tf}'), col(f'BB.lower{tf}') > col(f'KltChnl.lower{tf}')) for tf in timeframes]

        # --- Scanner Filters ---
        # Define the set of rules to find stocks in a squeeze.
        filters = [
            col('is_primary') == True,  # Only primary listings
            col('typespecs').has('common'),  # Only common stocks
            col('type') == 'stock',  # Exclude ETFs, etc.
            col('exchange') == 'NSE',  # Only stocks on the National Stock Exchange
            col('close').between(20, 10000),  # Price filter
            col('active_symbol') == True,  # Only actively traded symbols
            col('average_volume_10d_calc|5') > 50000,  # Minimum average volume
            col('Value.Traded|5') > 10000000,  # Minimum traded value
            Or(*squeeze_conditions)  # The core squeeze condition across all timeframes
        ]
        query_in_squeeze = Query().select(*select_cols).where2(And(*filters)).set_markets('india')
        _, df_in_squeeze = query_in_squeeze.get_scanner_data()

        current_squeeze_pairs = []
        if df_in_squeeze is not None and not df_in_squeeze.empty:
            print(f"Found {len(df_in_squeeze)} tickers currently in a squeeze.")
            # Process and save "In Squeeze" data
            for _, row in df_in_squeeze.iterrows():
                for tf_suffix, tf_name in tf_display_map.items():
                    if (row.get(f'BB.upper{tf_suffix}') < row.get(f'KltChnl.upper{tf_suffix}')) and (row.get(f'BB.lower{tf_suffix}') > row.get(f'KltChnl.lower{tf_suffix}')):
                        atr = row.get(f'ATR{tf_suffix}')
                        sma20 = row.get(f'SMA20{tf_suffix}')
                        bb_upper = row.get(f'BB.upper{tf_suffix}')

                        # New volatility calculation
                        if pd.notna(atr) and atr != 0 and pd.notna(sma20) and pd.notna(bb_upper):
                            std = bb_upper - sma20
                            volatility = std / atr
                        else:
                            volatility = 0
                        current_squeeze_pairs.append((row['ticker'], tf_name, volatility))
            df_in_squeeze['encodedTicker'] = df_in_squeeze['ticker'].apply(urllib.parse.quote)
            df_in_squeeze['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + df_in_squeeze['encodedTicker']
            df_in_squeeze['logo'] = df_in_squeeze['logoid'].apply(lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')
            squeeze_count_cols = [f'InSqueeze{tf}' for tf in timeframes]
            for tf in timeframes:
                df_in_squeeze[f'InSqueeze{tf}'] = (df_in_squeeze[f'BB.upper{tf}'] < df_in_squeeze[f'KltChnl.upper{tf}']) & (df_in_squeeze[f'BB.lower{tf}'] > df_in_squeeze[f'KltChnl.lower{tf}'])
            df_in_squeeze['SqueezeCount'] = df_in_squeeze[squeeze_count_cols].sum(axis=1)
            df_in_squeeze['highest_tf'] = df_in_squeeze.apply(get_highest_squeeze_tf, axis=1)
            # --- "In Squeeze" Data Enrichment ---
            # Calculate the strength of the squeeze (Regular, Strong, Very Strong)
            df_in_squeeze['squeeze_strength'] = df_in_squeeze.apply(get_squeeze_strength, axis=1)
            # Filter for only STRONG and VERY STRONG squeezes, as requested
            df_in_squeeze = df_in_squeeze[df_in_squeeze['squeeze_strength'].isin(['STRONG', 'VERY STRONG'])]
            # Calculate RVOL using the volume data for the highest timeframe the stock is in a squeeze
            df_in_squeeze['rvol'] = df_in_squeeze.apply(lambda row: get_dynamic_rvol(row, row['highest_tf'], tf_suffix_map), axis=1)
            # Determine momentum for "In Squeeze" stocks directly from MACD histogram
            df_in_squeeze['momentum'] = df_in_squeeze['MACD.hist'].apply(lambda x: 'Bullish' if x > 0 else 'Bearish' if x < 0 else 'Neutral')

            # Create a map for quick volatility lookup from the pairs calculated earlier
            volatility_map = {(ticker, tf): vol for ticker, tf, vol in current_squeeze_pairs}
            # Get the volatility for the highest timeframe the stock is in a squeeze
            df_in_squeeze['volatility'] = df_in_squeeze.apply(
                lambda row: volatility_map.get((row['ticker'], row['highest_tf']), 0),
                axis=1
            )
            # Calculate a heatmap score using the new formula: RVOL * Momentum * Volatility
            df_in_squeeze['HeatmapScore'] = df_in_squeeze['rvol'] * df_in_squeeze['momentum'].map({'Bullish': 1, 'Neutral': 0.5, 'Bearish': -1}) * df_in_squeeze['volatility']

            # --- Prepare Records for Database ---
            # Create a dictionary of the enriched ticker data for quick lookup
            ticker_data_map = {row['ticker']: row.to_dict() for _, row in df_in_squeeze.iterrows()}
            current_squeeze_records = []
            for ticker, timeframe, volatility in current_squeeze_pairs:
                ticker_data = ticker_data_map.get(ticker, {})
                current_squeeze_records.append({
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "volatility": volatility,
                    "rvol": ticker_data.get('rvol'),
                    "SqueezeCount": ticker_data.get('SqueezeCount'),
                    "squeeze_strength": ticker_data.get('squeeze_strength'),
                    "HeatmapScore": ticker_data.get('HeatmapScore')
                })

            generate_heatmap_json(df_in_squeeze, OUTPUT_JSON_IN_SQUEEZE)
        else:
            print("No tickers found currently in a squeeze.")
            generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_IN_SQUEEZE)

        # 3. Process event-based squeezes
        prev_squeeze_set = {(ticker, tf) for ticker, tf, vol in prev_squeeze_pairs}
        current_squeeze_set = {(r['ticker'], r['timeframe']) for r in current_squeeze_records}

        # Newly Formed Squeezes
        formed_pairs = current_squeeze_set - prev_squeeze_set
        if formed_pairs:
            formed_tickers = list(set(ticker for ticker, tf in formed_pairs))
            print(f"Found {len(formed_pairs)} newly formed squeezes.")
            df_formed_processed = df_in_squeeze[df_in_squeeze['ticker'].isin(formed_tickers)].copy()
            generate_heatmap_json(df_formed_processed, OUTPUT_JSON_FORMED)
        else:
            generate_heatmap_json(pd.DataFrame(), OUTPUT_JSON_FORMED)

        # Newly Fired Squeezes
        fired_pairs = prev_squeeze_set - current_squeeze_set
        if fired_pairs:
            fired_tickers = list(set(ticker for ticker, tf in fired_pairs))
            previous_volatility_map = {(ticker, tf): vol for ticker, tf, vol in prev_squeeze_pairs}
            query_fired = Query().select(*select_cols).set_tickers(*fired_tickers)
            _, df_fired = query_fired.get_scanner_data()

            if df_fired is not None and not df_fired.empty:
                newly_fired_events = []
                df_fired_map = {row['ticker']: row for _, row in df_fired.iterrows()}
                for ticker, fired_tf_name in fired_pairs:
                    if ticker in df_fired_map:
                        row_data = df_fired_map[ticker]
                        tf_suffix = tf_suffix_map.get(fired_tf_name)
                        if tf_suffix:
                            previous_volatility = previous_volatility_map.get((ticker, fired_tf_name), 0.0) or 0
                            atr = row_data.get(f'ATR{tf_suffix}')
                            sma20 = row_data.get(f'SMA20{tf_suffix}')
                            bb_upper = row_data.get(f'BB.upper{tf_suffix}')

                            # New volatility calculation
                            if pd.notna(atr) and atr != 0 and pd.notna(sma20) and pd.notna(bb_upper):
                                std = bb_upper - sma20
                                current_volatility = std / atr
                            else:
                                current_volatility = 0

                            if current_volatility > previous_volatility:
                                fired_event = row_data.to_dict()
                                fired_event.update({
                                    'fired_timeframe': fired_tf_name, 'previous_volatility': previous_volatility,
                                    'current_volatility': current_volatility, 'volatility_increased': True
                                })
                                newly_fired_events.append(fired_event)
                if newly_fired_events:
                    df_newly_fired = pd.DataFrame(newly_fired_events)
                    # --- "Fired Squeeze" Data Enrichment ---
                    df_newly_fired['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + df_newly_fired['ticker'].apply(urllib.parse.quote)
                    df_newly_fired['logo'] = df_newly_fired['logoid'].apply(lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')
                    # Calculate RVOL using the volume data for the timeframe the squeeze fired on
                    df_newly_fired['rvol'] = df_newly_fired.apply(lambda row: get_dynamic_rvol(row, row['fired_timeframe'], tf_suffix_map), axis=1)
                    # Determine breakout direction (momentum) using the new price-based logic
                    df_newly_fired['momentum'] = df_newly_fired.apply(lambda row: get_fired_breakout_direction(row, row['fired_timeframe'], tf_suffix_map), axis=1)
                    df_newly_fired['SqueezeCount'] = 1  # Squeeze count is always 1 for a fired event
                    df_newly_fired['highest_tf'] = df_newly_fired['fired_timeframe']
                    df_newly_fired['squeeze_strength'] = 'FIRED'
                    # Calculate a heatmap score using the new formula: RVOL * Momentum * Volatility
                    df_newly_fired['HeatmapScore'] = df_newly_fired['rvol'] * df_newly_fired['momentum'].map({'Bullish': 1, 'Neutral': 0.5, 'Bearish': -1}) * df_newly_fired['current_volatility']
                    save_fired_events_to_db(df_newly_fired)
                    append_df_to_csv(df_newly_fired, 'BBSCAN_FIRED_' + datetime.now().strftime('%d_%m_%y') + '.csv')

        # 4. Consolidate and display all recent fired events
        cleanup_old_fired_events()
        df_recent_fired = load_recent_fired_events_from_db()
        generate_heatmap_json(df_recent_fired, OUTPUT_JSON_FIRED)

        # 5. Save current squeeze state for next cycle
        save_current_squeeze_list_to_db(current_squeeze_records)
        print(f"Saved {len(current_squeeze_records)} (ticker, timeframe) pairs for the next scan.")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc() 

    print(f"\n--- Waiting for {TIME_INTERVAL_SECONDS} seconds until the next scan ---\n")
    sleep(TIME_INTERVAL_SECONDS)
