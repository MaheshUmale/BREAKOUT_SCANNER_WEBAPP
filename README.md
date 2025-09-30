# Real-Time Multi-Timeframe Squeeze Scanner Web Application

This project is an interactive web application that identifies stocks in a TTM Squeeze across multiple timeframes. It's designed to help traders spot potential volatility breakouts by allowing them to run on-demand scans using their own TradingView session for authenticated requests.

The results are visualized through a dynamic, single-page dashboard.

## Key Features

-   **Interactive Web App**: A Flask-based web application replaces the previous static file generation, providing a modern and interactive user experience.
-   **On-Demand Scanning**: Run scans whenever you want directly from the web interface.
-   **Session Cookie Authentication**: Uses your TradingView `sessionid` and `sessionid_sign` cookies to run authenticated scans, ensuring access to the latest data.
-   **Multi-Timeframe Scanning**: Monitors stocks for squeezes on timeframes from 1 minute to 1 month.
-   **Unified Dashboard**: All data—"In Squeeze," "Newly Formed," and "Recently Fired"—is now presented in a single, consolidated view.
-   **Intelligent Fired Squeezes**: Identifies squeezes that have not just fired, but have done so with a verifiable *increase in volatility*.
-   **Detailed Information**: Tooltips provide key data points like momentum, relative volume (RVOL), and volatility changes.

## How It Works

The application operates on a client-server model, providing a seamless and interactive experience.

### 1. Frontend Interface
The main interface is a single-page application built with HTML, TailwindCSS, and D3.js.
-   It provides input fields for you to enter your TradingView session cookies.
-   A "Run Scan" button triggers the backend scanning process on-demand.
-   The dashboard dynamically renders the results returned from the backend without needing to reload the page.

### 2. Flask Backend
The backend is a Python Flask application (`app.py`) that exposes a `/scan` API endpoint.
-   When you click "Run Scan," the frontend sends your session cookies to this endpoint.
-   The backend then executes the core scanning logic.

### 3. Core Scanning Logic
The backend process remains as sophisticated as before:
-   **The Squeeze Condition**: It identifies stocks where the **Bollinger Bands (BB)** are inside the **Keltner Channels (KC)**.
-   **Filtering**: It applies baseline filters for price, volume, and traded value.
-   **Squeeze Strength**: It calculates the squeeze strength and filters for only **"STRONG"** and **"VERY STRONG"** squeezes.
-   **Event Detection**: It compares the current scan against the previous scan (stored in a local SQLite database) to identify "Newly Formed" and "Recently Fired" events.
-   **Fired Squeeze Analysis**: It validates fired squeezes by confirming an increase in volatility and determining the breakout direction.
-   **Dynamic RVOL**: Relative Volume is calculated dynamically based on the most relevant timeframe for each event.

### 4. Data Visualization
Instead of generating static files, the backend returns a single JSON object containing all the processed data (`in_squeeze`, `formed`, `fired`). The frontend then uses D3.js to parse this data and render the interactive heatmaps and lists.

## Setup and Usage

### 1. Install Dependencies

First, ensure you have Python 3 installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

Open a terminal in the project directory and run the Flask app:

```bash
python3 app.py
```

The application will start, and you can access it by opening your web browser and navigating to:

**http://localhost:5001**

### 3. How to Use the Dashboard

1.  **Get Your Session Cookies**:
    *   Open your web browser and log in to your TradingView account.
    *   Open the browser's Developer Tools (usually by pressing F12).
    *   Go to the "Application" or "Storage" tab.
    *   Find the "Cookies" section and select `https://www.tradingview.com`.
    *   Find the `sessionid` and `sessionid_sign` cookies and copy their values.

2.  **Run a Scan**:
    *   Paste the copied `sessionid` and `sessionid_sign` values into the corresponding input fields on the dashboard.
    *   Click the **"Run Scan"** button.

The application will perform a live scan and display the results on the page. You can re-run the scan at any time by clicking the button again.