import os
import urllib.parse
import json
from time import sleep
import threading
from datetime import datetime, timedelta
import numpy as np
import sqlite3
# Assuming tradingview_screener version that supports query objects directly
from tradingview_screener import get_scanner_data, query as tv_query, Query, col, And, Or
import pandas as pd
from flask import Flask, render_template, jsonify, request
import logging # Import logging

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app_log.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)


# --- Flask App Initialization ---
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False # Preserve order in JSON responses

# --- Global state for auto-scanning ---
auto_scan_enabled = True
latest_scan_dfs = {
    "in_squeeze": pd.DataFrame(),
    "formed": pd.DataFrame(),
    "fired": pd.DataFrame(),
    "fired_confluence": pd.DataFrame() # NEW: For MTF confluence fired events
}
data_lock = threading.Lock()


import rookiepy
cookies = None
try:
    # Use brave by default, if other browsers are preferred, change this line
    cookies_list = rookiepy.brave(['.tradingview.com'])
    if cookies_list:
        cookies = {c['name']: c['value'] for c in cookies_list if 'name' in c and 'value' in c}
        logger.info("Successfully loaded TradingView cookies.")
    else:
        logger.warning("No TradingView cookies found. Scanning will be disabled.")
except Exception as e:
    logger.error(f"Error loading TradingView cookies: {e}. Scanning will be disabled.")

# --- SQLite Timestamp Handling ---
def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 format."""
    return val.isoformat()

def convert_timestamp(val):
    """Convert ISO 8601 string to datetime.datetime object."""
    # Ensure timezone-naive comparison if DB is storing naive timestamps
    return datetime.fromisoformat(val.decode())

# Register the adapter and converter
sqlite3.register_adapter(datetime, adapt_datetime_iso)
sqlite3.register_converter("timestamp", convert_timestamp)

# --- Pandas Configuration ---
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)

# --- Configuration for Timeframes and MTF Mapping ---
# All timeframes we care about, in ascending order of granularity
# We explicitly list common intraday and higher timeframes
TIME_FRAMES = ['1', '3', '5', '15', '30', '45', '60', '120', '240', '1D', '1W', '1M']

# Map shorter timeframes to potentially relevant higher timeframes for confluence checks
# Customize this based on your specific MTF strategy (e.g., 5m breakout, check 15m/60m)
HIGHER_TF_MAP = {
    '1': ['5', '15'],
    '3': ['15', '30'],
    '5': ['15', '30', '60'],
    '15': ['30', '60', '120'],
    '30': ['60', '120', '240', '1D'],
    '45': ['120', '240', '1D'],
    '60': ['240', '1D'],
    '120': ['1D'],
    '240': ['1D'],
}

# Mapping for display (keep existing logic for consistency)
tf_display_map = {
    '1': '1m', '3': '3m', '5': '5m', '15': '15m', '30': '30m', '45': '45m',
    '60': '1H', '120': '2H', '240': '4H', '1D': 'Daily', '1W': 'Weekly', '1M': 'Monthly'
}
# Reverse map for convenience
tf_suffix_map = {v: k for k, v in tf_display_map.items()} # e.g., '1m': '1'

# The original `timeframes` list from your code had empty string for Daily, and `|` prefixes
# Let's align this with `TIME_FRAMES` for clarity in Screener fields.
# For TradingView Screener, `Daily` often has no suffix or 'D'. We'll map '1D' to no suffix.
# Rebuilding `select_cols` to use `TIME_FRAMES` directly with correct suffix for screener.
# Screener fields use `|1`, `|5`, etc. For daily, it's often just `BB.upper` or `BB.upper|D`
# The `tradingview_screener` library often handles `1D` as an empty string suffix internally
# or uses 'D'. Let's stick to the `|{tf}` format and explicitly handle '1D' as no suffix
# for actual field names if that's how TradingView sends it.

select_cols = ['name', 'logoid', 'close', 'MACD.hist']
for tf in TIME_FRAMES:
    tf_suffix_for_screener = f'|{tf}' if tf not in ['1D'] else '' # TradingView often uses no suffix for Daily
    select_cols.extend([
        f'KltChnl.lower{tf_suffix_for_screener}', f'KltChnl.upper{tf_suffix_for_screener}',
        f'BB.lower{tf_suffix_for_screener}', f'BB.upper{tf_suffix_for_screener}',
        f'ATR{tf_suffix_for_screener}', f'SMA20{tf_suffix_for_screener}',
        f'volume{tf_suffix_for_screener}', f'average_volume_10d_calc{tf_suffix_for_screener}',
        f'Value.Traded{tf_suffix_for_screener}',
        f'RSI{tf_suffix_for_screener}' # Added RSI for potential future use
    ])

# Helper to get the correct suffix for column names in the fetched DataFrame
def get_column_suffix(timeframe_key):
    return f'|{timeframe_key}' if timeframe_key not in ['1D'] else ''


# --- Helper Functions ---
def append_df_to_csv(df, csv_path):
    """
    Appends a DataFrame to a CSV file. Creates the file with a header if it doesn't
    exist, otherwise appends without the header.
    """
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def generate_heatmap_data(df):
    """
    Generates a simple, flat list of dictionaries from the dataframe for the D3 heatmap.
    This replaces the JSON file generation.
    """
    # Ensure all expected columns are present, even if empty, for frontend consistency
    base_required_cols = [
        'ticker', 'HeatmapScore', 'SqueezeCount', 'rvol', 'URL', 'logo', 'momentum',
        'highest_tf', 'squeeze_strength', 'description' # Added description
    ]
    for c in base_required_cols:
        if c not in df.columns:
            if c == 'momentum': df[c] = 'Neutral'
            elif c in ['highest_tf', 'squeeze_strength', 'description']: df[c] = 'N/A'
            elif c == 'HeatmapScore': df[c] = 0.0
            elif c == 'SqueezeCount': df[c] = 0
            else: df[c] = '' # For URL, logo, ticker

    heatmap_data = []
    for _, row in df.iterrows():
        stock_data = {
            "name": row['ticker'],
            "description": row['description'], # Include description
            "value": row['HeatmapScore'],
            "count": row.get('SqueezeCount', 0),
            "rvol": row['rvol'],
            "url": row['URL'],
            "logo": row['logo'],
            "momentum": row['momentum'],
            "highest_tf": row['highest_tf'],
            "squeeze_strength": row['squeeze_strength']
        }
        # Add additional columns if they exist in the dataframe (for fired events)
        if 'fired_timeframe' in df.columns: stock_data['fired_timeframe'] = row['fired_timeframe']
        if 'fired_timestamp' in df.columns and pd.notna(row['fired_timestamp']):
            stock_data['fired_timestamp'] = row['fired_timestamp'].isoformat()
        if 'previous_volatility' in df.columns: stock_data['previous_volatility'] = row['previous_volatility']
        if 'current_volatility' in df.columns: stock_data['current_volatility'] = row['current_volatility']
        if 'volatility_increased' in df.columns: stock_data['volatility_increased'] = row['volatility_increased']
        if 'higher_timeframe_squeeze' in df.columns: stock_data['higher_timeframe_squeeze'] = row['higher_timeframe_squeeze']
        if 'breakout_direction' in df.columns: stock_data['breakout_direction'] = row['breakout_direction']

        heatmap_data.append(stock_data)
    return heatmap_data

# --- Data Processing Functions ---
def calculate_squeeze_status_for_row(row, tf_key):
    """Calculates if a single stock (row) is in a squeeze for a given timeframe."""
    tf_suffix = get_column_suffix(tf_key)
    bb_upper_col = f"BB.upper{tf_suffix}"
    bb_lower_col = f"BB.lower{tf_suffix}"
    kc_upper_col = f"KltChnl.upper{tf_suffix}"
    kc_lower_col = f"KltChnl.lower{tf_suffix}"

    if not all(col in row.index for col in [bb_upper_col, bb_lower_col, kc_upper_col, kc_lower_col]):
        return False

    bb_upper = row.get(bb_upper_col)
    bb_lower = row.get(bb_lower_col)
    kc_upper = row.get(kc_upper_col)
    kc_lower = row.get(kc_lower_col)

    if any(pd.isna(val) for val in [bb_upper, bb_lower, kc_upper, kc_lower]):
        return False

    return (bb_upper < kc_upper) and (bb_lower > kc_lower)

def get_highest_squeeze_tf(row):
    # Sort timeframes from longest to shortest for 'highest_tf'
    sorted_timeframes = sorted(TIME_FRAMES, key=lambda tf: TIME_FRAMES.index(tf), reverse=True)
    for tf_key in sorted_timeframes:
        if row.get(f'InSqueeze_{tf_key}', False): # Check the dynamically created boolean column
            return tf_display_map.get(tf_key, tf_key)
    return 'N/A'

def get_dynamic_rvol(row, timeframe_key):
    tf_suffix = get_column_suffix(timeframe_key)
    vol_col, avg_vol_col = f'volume{tf_suffix}', f'average_volume_10d_calc{tf_suffix}'
    volume, avg_volume = row.get(vol_col), row.get(avg_vol_col)
    if pd.isna(volume) or pd.isna(avg_volume) or avg_volume == 0: return 0
    return volume / avg_volume

def get_volatility(row, timeframe_key):
    tf_suffix = get_column_suffix(timeframe_key)
    atr, sma20, bb_upper = row.get(f'ATR{tf_suffix}'), row.get(f'SMA20{tf_suffix}'), row.get(f'BB.upper{tf_suffix}')
    if pd.isna(atr) or atr == 0 or pd.isna(sma20) or pd.isna(bb_upper): return 0
    return (bb_upper - sma20) / atr

def get_squeeze_strength(row):
    highest_tf_name = row['highest_tf']
    if highest_tf_name == 'N/A': return "N/A"
    tf_key = tf_suffix_map.get(highest_tf_name) # Convert display name back to key
    if not tf_key: return "N/A" # Should not happen if mapping is correct

    tf_suffix = get_column_suffix(tf_key)
    bb_upper, bb_lower = row.get(f'BB.upper{tf_suffix}'), row.get(f'BB.lower{tf_suffix}')
    kc_upper, kc_lower = row.get(f'KltChnl.upper{tf_suffix}'), row.get(f'KltChnl.lower{tf_suffix}')

    if any(pd.isna(val) for val in [bb_upper, bb_lower, kc_upper, kc_lower]): return "N/A"
    bb_width, kc_width = bb_upper - bb_lower, kc_upper - kc_lower
    if bb_width == 0: return "N/A" # Avoid division by zero

    sqz_strength = kc_width / bb_width
    if sqz_strength >= 2: return "VERY STRONG"
    elif sqz_strength >= 1.5: return "STRONG"
    elif sqz_strength > 1: return "Regular"
    else: return "N/A" # Should be in squeeze, but maybe a weak one not strong enough to rate

def get_breakout_direction(row, timeframe_key):
    tf_suffix = get_column_suffix(timeframe_key)
    close, bb_upper, kc_upper, bb_lower, kc_lower = \
        row.get('close'), row.get(f'BB.upper{tf_suffix}'), row.get(f'KltChnl.upper{tf_suffix}'), \
        row.get(f'BB.lower{tf_suffix}'), row.get(f'KltChnl.lower{tf_suffix}')

    if any(pd.isna(val) for val in [close, bb_upper, kc_upper, bb_lower, kc_lower]): return 'Neutral'

    # Bullish condition: close breaks above upper BB, and upper BB is outside upper KC
    if close > bb_upper and bb_upper > kc_upper: return 'Bullish'
    # Bearish condition: close breaks below lower BB, and lower BB is outside lower KC
    elif close < bb_lower and bb_lower < kc_lower: return 'Bearish'
    else: return 'Neutral'


# --- Database Functions ---
DB_FILE = 'squeeze_history.db' # Define DB_FILE globally

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Table to store which (ticker, timeframe) pairs were in squeeze in the LAST scan
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prev_squeeze_state (
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            volatility REAL,
            PRIMARY KEY (ticker, timeframe)
        );
    """)
    # Table to store detected fired events (including MTF confluence)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fired_squeeze_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fired_timestamp TIMESTAMP NOT NULL,
            ticker TEXT NOT NULL,
            fired_timeframe TEXT NOT NULL,
            higher_timeframe_squeeze BOOLEAN, -- NEW: Indicates if a higher TF was in squeeze
            breakout_direction TEXT,
            rvol REAL,
            squeeze_strength TEXT,
            momentum TEXT,
            previous_volatility REAL,
            current_volatility REAL,
            HeatmapScore REAL,
            URL TEXT,
            logo TEXT,
            SqueezeCount INTEGER, -- Total count for all TFs at fire time (if applicable)
            highest_tf TEXT, -- Highest TF in squeeze at fire time (if applicable)
            description TEXT
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized with updated schema.")


def load_previous_squeeze_list_from_db():
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    try:
        # We need all previous squeeze states to compare with current
        cursor.execute('SELECT ticker, timeframe, volatility FROM prev_squeeze_state')
        return [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    finally: conn.close()

def save_current_squeeze_list_to_db(current_squeeze_records):
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prev_squeeze_state") # Clear previous state
    if current_squeeze_records:
        data_to_insert = [(r['ticker'], r['timeframe'], r['volatility']) for r in current_squeeze_records]
        cursor.executemany('INSERT INTO prev_squeeze_state (ticker, timeframe, volatility) VALUES (?, ?, ?)', data_to_insert)
    conn.commit()
    conn.close()
    logger.debug(f"Saved {len(current_squeeze_records)} current squeeze states to DB.")

def save_fired_events_to_db(fired_events_df):
    if fired_events_df.empty: return
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    data_to_insert = []
    for _, row in fired_events_df.iterrows():
        data_to_insert.append((
            row['fired_timestamp'], row['ticker'], row['fired_timeframe'],
            row['higher_timeframe_squeeze'], row['breakout_direction'], row['rvol'],
            row['squeeze_strength'], row['momentum'], row['previous_volatility'],
            row['current_volatility'], row['HeatmapScore'], row['URL'],
            row['logo'], row.get('SqueezeCount', 0), row.get('highest_tf', 'N/A'),
            row.get('description', '')
        ))
    cursor.executemany("""
        INSERT INTO fired_squeeze_events (fired_timestamp, ticker, fired_timeframe,
        higher_timeframe_squeeze, breakout_direction, rvol, squeeze_strength, momentum,
        previous_volatility, current_volatility, HeatmapScore, URL, logo, SqueezeCount, highest_tf, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    conn.commit()
    conn.close()
    logger.info(f"Saved {len(fired_events_df)} fired events to DB.")


def load_recent_fired_events_from_db(is_confluence=False):
    conn = sqlite3.connect(DB_FILE, detect_types=sqlite3.PARSE_DECLTYPES)
    # Load events from the last 24 hours to show recent activity
    twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
    query = "SELECT * FROM fired_squeeze_events WHERE fired_timestamp >= ?"
    params = (twenty_four_hours_ago,)
    if is_confluence:
        query += " AND higher_timeframe_squeeze = 1"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def cleanup_old_fired_events():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Keep fired events for a longer duration, e.g., 7 days, for historical analysis
    seven_days_ago = datetime.now() - timedelta(days=7)
    cursor.execute("DELETE FROM fired_squeeze_events WHERE fired_timestamp < ?", (seven_days_ago,))
    conn.commit()
    conn.close()
    logger.debug("Cleaned up old fired events from DB.")


# --- Main Scanning Logic ---
def run_scan():
    """
    Runs a full squeeze scan, processes the data, saves it to the database,
    and returns the processed dataframes.
    """
    global cookies # Use the global cookies variable
    if cookies is None:
        logger.warning("Skipping scan because TradingView cookies are not loaded.")
        return {
            "in_squeeze": pd.DataFrame(),
            "formed": pd.DataFrame(),
            "fired": pd.DataFrame(),
            "fired_confluence": pd.DataFrame()
        }
    try:
        logger.info("Starting TTM Squeeze scan...")
        scan_start_time = datetime.now()

        # 1. Load previous squeeze state from DB
        prev_squeeze_records = load_previous_squeeze_list_from_db()
        # Convert to set for efficient comparison: {(ticker, timeframe): volatility}
        prev_squeeze_state = {(r[0], r[1]): r[2] for r in prev_squeeze_records}

        # 2. Query TradingView for all relevant data fields across all timeframes
        # We fetch all necessary data and then determine squeeze status in Python
        filters = [
            col('is_primary') == True, col('typespecs').has('common'), col('type') == 'stock',
            col('exchange') == 'NSE', col('close').between(20, 10000), col('active_symbol') == True,
            # Ensure liquidity for 5m, as we might look for 5m fires
            col(f'average_volume_10d_calc|5') > 50000, col(f'Value.Traded|5') > 10000000,
            # No need for Or(*squeeze_conditions) in main query if we calculate in Python
        ]
        query_all_indicators = tv_query().select(*select_cols).where2(And(*filters)).set_markets('india').set_property('preset', 'all_stocks')

        _, df_raw_data = get_scanner_data(query_all_indicators, cookies=cookies)

        if df_raw_data is None or df_raw_data.empty:
            logger.warning("No data returned from TradingView screener.")
            return {
                "in_squeeze": pd.DataFrame(),
                "formed": pd.DataFrame(),
                "fired": pd.DataFrame(),
                "fired_confluence": pd.DataFrame()
            }

        # Rename 'name' to 'ticker' for consistency
        df_raw_data.rename(columns={'name': 'ticker', 'MACD.hist': 'MACD_hist'}, inplace=True)
        # Add URL and logo early for all stocks
        df_raw_data['encodedTicker'] = df_raw_data['ticker'].apply(urllib.parse.quote)
        df_raw_data['URL'] = "https://in.tradingview.com/chart/N8zfIJVK/?symbol=" + df_raw_data['encodedTicker']
        df_raw_data['logo'] = df_raw_data['logoid'].apply(lambda x: f"https://s3-symbol-logo.tradingview.com/{x}.svg" if pd.notna(x) and x.strip() else '')
        df_raw_data['description'] = df_raw_data.get('description', '') # Ensure description column exists

        # 3. Determine current squeeze states for all stocks and timeframes
        current_squeeze_records = [] # For saving to prev_squeeze_state in DB
        current_squeeze_status_map = {} # {(ticker, timeframe): True/False}
        ticker_details_map = {row['ticker']: row.to_dict() for _, row in df_raw_data.iterrows()} # For quick lookup

        for _, row in df_raw_data.iterrows():
            ticker = row['ticker']
            for tf_key in TIME_FRAMES:
                is_in_squeeze_now = calculate_squeeze_status_for_row(row, tf_key)
                current_squeeze_status_map[(ticker, tf_display_map[tf_key])] = is_in_squeeze_now

                if is_in_squeeze_now:
                    volatility = get_volatility(row, tf_key)
                    current_squeeze_records.append({
                        'ticker': ticker,
                        'timeframe': tf_display_map[tf_key],
                        'volatility': volatility,
                        **row.to_dict() # Include all raw data for enrichment later
                    })

        # Process df_in_squeeze_current (for heatmap)
        df_in_squeeze_current = pd.DataFrame([rec for rec in current_squeeze_records if 'ticker' in rec])
        if not df_in_squeeze_current.empty:
            df_in_squeeze_current['timeframe_key'] = df_in_squeeze_current['timeframe'].map(tf_suffix_map)
            df_in_squeeze_current['highest_tf'] = df_in_squeeze_current.apply(lambda r: get_highest_squeeze_tf(r), axis=1) # Need to calculate this based on all TFs
            df_in_squeeze_current['squeeze_strength'] = df_in_squeeze_current.apply(lambda r: get_squeeze_strength(r), axis=1)
            df_in_squeeze_current['rvol'] = df_in_squeeze_current.apply(lambda r: get_dynamic_rvol(r, r['highest_tf']), axis=1)
            df_in_squeeze_current['momentum'] = df_in_squeeze_current['MACD_hist'].apply(lambda x: 'Bullish' if x > 0 else 'Bearish' if x < 0 else 'Neutral')
            df_in_squeeze_current['HeatmapScore'] = df_in_squeeze_current['rvol'] * df_in_squeeze_current['momentum'].map({'Bullish': 1, 'Neutral': 0.5, 'Bearish': -1}) * df_in_squeeze_current['volatility']
            df_in_squeeze_current.drop(columns=['timeframe_key'], inplace=True, errors='ignore')
            # Aggregate if a ticker is in squeeze on multiple timeframes for the heatmap display
            df_in_squeeze_processed = df_in_squeeze_current.groupby('ticker').apply(lambda x: x.iloc[0].copy()).reset_index(drop=True)
            df_in_squeeze_processed['SqueezeCount'] = df_in_squeeze_current.groupby('ticker')['timeframe'].nunique().values
        else:
            df_in_squeeze_processed = pd.DataFrame()


        # 4. Detect Formed and Fired Events
        df_formed_processed = pd.DataFrame()
        df_fired_processed = pd.DataFrame()
        df_fired_confluence_processed = pd.DataFrame() # NEW DF

        formed_events = []
        fired_events = []
        fired_confluence_events = [] # NEW LIST

        current_squeeze_pairs_set = set(current_squeeze_status_map.keys())
        prev_squeeze_pairs_set = set(prev_squeeze_state.keys())

        # Formed
        for (ticker, tf_display_name) in (current_squeeze_pairs_set - prev_squeeze_pairs_set):
            # Check if this stock has data in our raw_df
            if ticker in ticker_details_map:
                row_data = ticker_details_map[ticker]
                tf_key = tf_suffix_map[tf_display_name]
                volatility = get_volatility(row_data, tf_key)
                formed_events.append({
                    'ticker': ticker,
                    'fired_timeframe': tf_display_name, # Reusing this for formed_timeframe
                    'fired_timestamp': datetime.now(),
                    'current_volatility': volatility,
                    'rvol': get_dynamic_rvol(row_data, tf_key),
                    'momentum': get_breakout_direction(row_data, tf_key), # Using breakout for momentum
                    'description': row_data.get('description', ''),
                    # Add other fields as needed for display
                    **row_data # Include all raw data for enrichment
                })
        if formed_events:
            df_formed_processed = pd.DataFrame(formed_events)


        # Fired
        for (ticker, fired_tf_display_name) in (prev_squeeze_pairs_set - current_squeeze_pairs_set):
            # Ensure we have current data for this ticker
            if ticker in ticker_details_map:
                row_data = ticker_details_map[ticker]
                fired_tf_key = tf_suffix_map[fired_tf_display_name]

                previous_volatility = prev_squeeze_state.get((ticker, fired_tf_display_name), 0.0)
                current_volatility = get_volatility(row_data, fired_tf_key)
                
                # Basic Fired Event Data
                fired_event_data = {
                    'ticker': ticker,
                    'fired_timeframe': fired_tf_display_name,
                    'fired_timestamp': datetime.now(),
                    'previous_volatility': previous_volatility,
                    'current_volatility': current_volatility,
                    'volatility_increased': current_volatility > previous_volatility,
                    'breakout_direction': get_breakout_direction(row_data, fired_tf_key),
                    'rvol': get_dynamic_rvol(row_data, fired_tf_key),
                    'momentum': row_data.get('MACD_hist'), # Use MACD hist for momentum at fire
                    'squeeze_strength': 'FIRED', # Placeholder, actual strength was pre-fire
                    'description': row_data.get('description', ''),
                    'logo': row_data.get('logo', ''),
                    'URL': row_data.get('URL', ''),
                    # Add all raw data for comprehensive storage
                    **row_data
                }

                # --- NEW: Check for Higher Timeframe Squeeze Confluence ---
                higher_timeframe_active_squeeze = False
                # Get relevant higher timeframes for the fired_tf_key
                higher_tfs_to_check = HIGHER_TF_MAP.get(fired_tf_key, [])

                for h_tf_key in higher_tfs_to_check:
                    h_tf_display_name = tf_display_map[h_tf_key]
                    # Check if the higher TF was in a squeeze just BEFORE this current fire
                    # This means it should be in prev_squeeze_state
                    if (ticker, h_tf_display_name) in prev_squeeze_state:
                        higher_timeframe_active_squeeze = True
                        logger.info(f"!!! MTF Squeeze Confluence Fired: {ticker} {fired_tf_display_name} breakout while {h_tf_display_name} was in squeeze !!!")
                        break # Found one, no need to check others

                fired_event_data['higher_timeframe_squeeze'] = higher_timeframe_active_squeeze
                fired_events.append(fired_event_data)

                if higher_timeframe_active_squeeze:
                    fired_confluence_events.append(fired_event_data)


        if fired_events:
            df_fired_all = pd.DataFrame(fired_events)
            
            # Enrich fired events with SqueezeCount, highest_tf, etc.
            if not df_fired_all.empty:
                # Calculate SqueezeCount and highest_tf *at the time of fire*
                # This requires recalculating squeeze status based on the current row data for all TFs
                for tf_key in TIME_FRAMES:
                    df_fired_all[f'InSqueeze_{tf_key}'] = df_fired_all.apply(lambda r: calculate_squeeze_status_for_row(r, tf_key), axis=1)

                df_fired_all['SqueezeCount'] = df_fired_all[[f'InSqueeze_{tf}' for tf in TIME_FRAMES]].sum(axis=1)
                df_fired_all['highest_tf'] = df_fired_all.apply(lambda r: get_highest_squeeze_tf(r), axis=1)
                
                # Calculate HeatmapScore for fired events (can use volatility at fire and momentum)
                df_fired_all['HeatmapScore'] = df_fired_all['r