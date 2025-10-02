import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
import pymongo
from pymongo import MongoClient
import argparse

# Load environment variables from .env file
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Initialize or reset MongoDB collections for quant trading system.')
parser.add_argument('--reset', action='store_true', help='Drop all collections before recreating them.')
args = parser.parse_args()

# Get MongoDB URI from environment
mongo_uri = os.getenv('MONGO_URI')
if not mongo_uri:
    print("Error: MONGO_URI environment variable is not set.")
    exit(1)

# URL-encode username and password in the URI if present
from urllib.parse import urlparse
parsed = urlparse(mongo_uri)
if '@' in parsed.netloc:
    user_pass, host = parsed.netloc.rsplit('@', 1)
    if ':' in user_pass:
        user, pas = user_pass.split(':', 1)
        user = quote_plus(user)
        pas = quote_plus(pas)
        new_netloc = f"{user}:{pas}@{host}"
        mongo_uri = mongo_uri.replace(parsed.netloc, new_netloc)

# Connect to MongoDB
try:
    client = MongoClient(mongo_uri, uuidRepresentation='standard')
    db = client['quant_trading']
    print("Connected to MongoDB successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Define collections and their indexes
collections = [
    'strategies',
    'experiments',
    'optimization_runs',
    'iterations',
    'backtests',
    'trades',
    'equity_curves',
    'balance_curves',
    'rolling_metrics',
    'windows'
]

indexes = {
    'strategies': [('_id', 1)],
    'experiments': [('experiment_name', 1), ('timestamp', -1)],
    'optimization_runs': [('experiment_id', 1), ('stock_symbol', 1), ('timeframe', 1), ('best_value', -1)],
    'iterations': [('window_id', 1), ('phase', 1), ('iteration', 1)],
    'backtests': [('window_id', 1)],
    'trades': [('backtest_id', 1), ('Datetime', 1)],
    'equity_curves': [('backtest_id', 1), ('datetime', 1), ('scope', 1)],
    'balance_curves': [('backtest_id', 1), ('datetime', 1)],
    'rolling_metrics': [('backtest_id', 1), ('datetime', 1)],
    'windows': [('optimization_run_id', 1)]
}

# Reset if requested
if args.reset:
    print("Reset mode: Dropping all collections...")
    for coll_name in collections:
        db[coll_name].drop()
        print(f"Dropped collection: {coll_name}")

# Create collections and indexes
for coll_name in collections:
    # Create collection if it doesn't exist
    if coll_name not in db.list_collection_names():
        db.create_collection(coll_name)
        print(f"Created collection: {coll_name}")
    else:
        print(f"Collection {coll_name} already exists.")

    # Create indexes
    for index_spec in indexes[coll_name]:
        try:
            db[coll_name].create_index(index_spec)
            print(f"Created index on {coll_name}.{index_spec[0]}")
        except Exception as e:
            print(f"Error creating index on {coll_name}")

print("MongoDB setup complete.")