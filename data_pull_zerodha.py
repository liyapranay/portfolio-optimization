import requests
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv
from kiteconnect import KiteConnect
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

BASEPATH = "."
# Comment out previous version
# instrument_master = pd.read_csv(f"{BASEPATH}/masters/instrument.csv", low_memory=False)

def get_historical_data(kite, instrument_token, interval, start_time, end_time):
  """
  Fetches historical data from Zerodha Kite API.

  Args:
    kite: KiteConnect instance.
    instrument_token: The instrument token.
    interval: The interval (e.g., '5minute').
    start_time: The start date in 'YYYY-MM-DD' format.
    end_time: The end date in 'YYYY-MM-DD' format.

  Returns:
    A list of historical data, or None if an error occurred.
  """
  try:
    data = kite.historical_data(instrument_token=instrument_token, interval=interval, from_date=start_time, to_date=end_time)
    return data
  except Exception as e:
    print(f"Error fetching data: {e}")
    return None

def convert_to_df(data):
  if data:
    df = pd.DataFrame(data)
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['date'].dt.strftime('%H:%M:%S')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
    return df
  else:
    raise ValueError("Invalid data format")

def generate_date_pairs(start_date_str, end_date_str):
    """
    Generates 13-day date pairs based on start and end dates.

    Args:
        start_date_str: The start date in 'YYYY-MM-DD' format.
        end_date_str: The end date in 'YYYY-MM-DD' format.

    Returns:
        A list of lists, where each inner list contains a start and end date (strings in 'YYYY-MM-DD' format).
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    date_pairs = []

    current_start_date = start_date
    while current_start_date <= end_date:
        current_end_date = current_start_date + timedelta(days=13)
        if current_end_date > end_date:
            current_end_date = end_date

        date_pairs.append([current_start_date.strftime('%Y-%m-%d'), current_end_date.strftime('%Y-%m-%d')])

        current_start_date = current_end_date + timedelta(days=1)

    return date_pairs

if __name__ == "__main__":
    # Initialize KiteConnect with saved access token
    kite_api_key = os.getenv('KITE_API_KEY')
    kite_access_token = os.getenv('KITE_ACCESS_TOKEN')
    if not kite_api_key or not kite_access_token:
        print("KITE_API_KEY or KITE_ACCESS_TOKEN not found in .env. Please run login_zerodha.py first.")
        sys.exit(1)
    kite = KiteConnect(api_key=kite_api_key)
    kite.reqsession = requests.Session()
    kite.reqsession.verify = False  # Disable SSL verification
    kite.set_access_token(kite_access_token)

    # Step 3: Get instruments list
    instruments_list = kite.instruments('NSE')
    instruments_df = pd.DataFrame(instruments_list)
    print("Instruments DataFrame head:")
    print(instruments_df.head())

    parser = argparse.ArgumentParser(description="Pull historical stock data from Zerodha Kite API")
    parser.add_argument('--exchange', default='NSE', help='Exchange (default: NSE)')
    parser.add_argument('--stock_symbol', required=True, help='Stock symbol (required)')
    parser.add_argument('--start_batch_date_str', default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), help='Start batch date (default: yesterday)')
    parser.add_argument('--end_batch_date_str', default=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), help='End batch date (default: yesterday)')
    parser.add_argument('--interval', required=True, choices=['minute', 'day', '3minute', '5minute', '10minute', '15minute', '30minute', '60minute'], help='Interval (required)')

    args = parser.parse_args()

    exchange = args.exchange
    stock_symbol = args.stock_symbol
    start_batch_date_str = args.start_batch_date_str
    end_batch_date_str = args.end_batch_date_str
    interval = args.interval

    if not stock_symbol:
        raise ValueError("stock_symbol cannot be blank")

    # Get instrument token for symbol
    instrument_token = str(instruments_df[instruments_df['tradingsymbol'] == stock_symbol]['instrument_token'].values[0])
    print(f"Instrument token for {stock_symbol}: {instrument_token}")

    date_pairs = generate_date_pairs(start_batch_date_str, end_batch_date_str)

    global_df = pd.DataFrame()
    for pairs in date_pairs:
      start_time = pairs[0]
      end_time = pairs[1]
      try:
        data = get_historical_data(kite, instrument_token, interval, start_time, end_time)
        if data is None:
          print(f"Failed to fetch data for {pairs[0]} -> {pairs[1]}, skipping")
          continue
        data = convert_to_df(data)
        print(f"Extracted for date range {pairs[0]} -> {pairs[1]} with length {len(data)}")
        global_df = pd.concat([global_df, data], ignore_index=True)
      except Exception as e:
        print(f"Error processing data for {pairs[0]} -> {pairs[1]}: {e}, skipping")
        continue

    # Create directory if not exists
    output_dir = f"{BASEPATH}/data/{interval}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/{instrument_token}_{exchange}_{stock_symbol}.csv"

    if global_df.empty:
        print("No data fetched, exiting")
        sys.exit(1)

    # If file exists, read and append
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        global_df = pd.concat([existing_df, global_df], ignore_index=True)
        # Remove duplicates based on date and time
        global_df = global_df.drop_duplicates(subset=['date', 'time'], keep='first')

    global_df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")