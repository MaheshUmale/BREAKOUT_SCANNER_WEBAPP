import os
import urllib.parse
import json
from time import sleep
import threading
from datetime import datetime, timedelta
import numpy as np
import sqlite3
from tradingview_screener import Query, col, And, Or
import pandas as pd
from flask import Flask, render_template, jsonify, request

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global state for auto-scanning ---
auto_scan_enabled = True
latest_scan_dfs = {
    "in_squeeze": pd.DataFrame(),
    "formed": pd.DataFrame(),
    "fired": pd.DataFrame()
}
data_lock = threading.Lock()


import rookiepy
cookies = None
try:
    cookies = rookiepy.to_cookiejar(rookiepy.brave(['.tradingview.com']))
    print("Successfully loaded TradingView cookies.")
    _, df = Query().select('exchange', 'update_mode').limit(1_000_000).get_scanner_data(cookies=cookies)
    df = df.groupby('exchange')['update_mode'].value_counts()
    print(df)

except Exception as e:
    print(f"Warning: Could not load TradingView cookies. Scanning will be disabled. Error: {e}")

 

# --- Main Scanning Logic ---
def run_scan():
    _, df = (Query().select(    "volume",		"relative_volume_10d_calc",		"relative_volume_intraday|5","relative_volume_10d_calc|5",)
        
        
    .set_markets('america', 'italy', 'vietnam')
    .set_tickers('NASDAQ:GLXG','NASDAQ:NOEM','NASDAQ:HCTI','NASDAQ:PALI','NASDAQ:FACT','NASDAQ:ESLA','NASDAQ:MAMK','NASDAQ:VCIG','NASDAQ:ASNS')#,'NASDAQ:RDAG','NASDAQ:CAPS','NASDAQ:VNME','NASDAQ:RANG','NASDAQ:INAC','NASDAQ:GPAT','NASDAQ:SDM','NASDAQ:CEPF','NASDAQ:BTM','NASDAQ:EPSM','NASDAQ:UTSI','NASDAQ:TORO','NASDAQ:JFU','NYSE:AES','NASDAQ:FBIO','NASDAQ:CJET','NASDAQ:FONR','NASDAQ:HSPT','NASDAQ:PCAP','NASDAQ:CLOV','NASDAQ:FIEE','NASDAQ:DTSQ','NASDAQ:VTYX','NASDAQ:TCRX','NASDAQ:FSEA','NASDAQ:RZLV','NASDAQ:VECO',
    .set_tickers('NASDAQ:RITR','NASDAQ:FPAY','NASDAQ:INTG','NASDAQ:QH','NASDAQ:MIMI','NASDAQ:SHIM','NASDAQ:VTSI',)
    .get_scanner_data())
    print(df)
     
        

KAGGLE_USERNAME="mumale"
KAGGLE_KEY="c17479b2fe686386bf3bb5f1c5d3d256"
#{"username":"mumale","key":"c17479b2fe686386bf3bb5f1c5d3d256"}

def testDwonload():
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    # import kagglehub

    # kagglehub.login()

    # Set the path to the file you'd like to load
    file_path = "."

    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "debashis74017/algo-trading-data-nifty-100-data-with-indicators",
    file_path,
    # Provide any additional arguments like 
    # sql_query or pandas_kwargs. See the 
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )

    print("First 5 records:", df.head())


 
if __name__ == '__main__':
    # run_scan()
    # Start the background scanner thread
    testDwonload()
   