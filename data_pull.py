import requests
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
import sys
from dotenv import load_dotenv

load_dotenv()

BASEPATH = "."
instrument_master = pd.read_csv(f"{BASEPATH}/masters/instrument.csv", low_memory=False)
def get_historical_data(exchange, instrument_token, interval, start_time, end_time, api_key, access_token):
  """
  Fetches historical data from the mstock API with specified headers.

  Args:
    exchange: The exchange (e.g., NSE, BSE).
    instrument_token: The instrument token.
    interval: The interval (e.g., 1minute, 5minute).
    start_time: The start time in 'YYYY-MM-DD HH:MM:SS' format.
    end_time: The end time in 'YYYY-MM-DD HH:MM:SS' format.
    api_key: Your API key.
    access_token: Your access token.

  Returns:
    A dictionary containing the API response, or None if an error occurred.
  """
  base_url = "https://api.mstock.trade/openapi/typea/instruments/historical/"
  url = f"{base_url}{exchange}/{instrument_token}/{interval}?from={start_time}&to={end_time}"

  headers = {
      "Authorization": f"token {api_key}:{access_token}",
      "X-Mirae-Version": "1"
  }

  try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()
  except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
    return None

def convert_to_df(data):
  if data and 'data' in data and 'candles' in data['data']:
    candles_data = data['data']['candles']
    df = pd.DataFrame(candles_data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])

    # Split the datetime column into date and time
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['time'] = pd.to_datetime(df['datetime']).dt.time

    # Reorder columns to have date and time first
    df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
    return df
  else:
    raise ValueError("Invalid data format")
    # print("No data found in the expected format.")

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
    parser = argparse.ArgumentParser(description="Pull historical stock data from mstock API")
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

    # Lookup instrument_token
    try:
        instrument_token = str(instrument_master['instrument_token']\
         [(instrument_master['segment']=='EQ')\
          &(instrument_master['exchange']==exchange)\
          &(instrument_master['tradingsymbol']==stock_symbol)].values[0])
    except IndexError:
        print(f"Stock symbol {stock_symbol} not found in instrument master for exchange {exchange}")
        sys.exit(1)

    date_pairs = generate_date_pairs(start_batch_date_str, end_batch_date_str)

    api_key = os.getenv('API_KEY')
    access_token = os.getenv('ACCESS_TOKEN')
    token_timestamp = os.getenv('TOKEN_TIMESTAMP')

    if not api_key or not access_token:
        print("API_KEY or ACCESS_TOKEN not found in .env. Please run login.py")
        sys.exit(1)

    if token_timestamp:
        ts = datetime.fromisoformat(token_timestamp)
        if datetime.now() - ts > timedelta(hours=8):
            print("Access token is older than 8 hours. Please run login.py to refresh.")
            sys.exit(1)
    global_df = pd.DataFrame()
    for pairs in date_pairs:
      start_time = pairs[0]
      end_time = pairs[1]
      try:
        data = get_historical_data(exchange, instrument_token, interval, start_time, end_time, api_key, access_token)
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