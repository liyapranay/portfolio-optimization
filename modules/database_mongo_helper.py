import os
import pandas as pd
import uuid
from dotenv import load_dotenv
from urllib.parse import quote_plus, urlparse
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Load environment variables
load_dotenv()

# Get and encode MongoDB URI
mongo_uri = os.getenv('MONGO_URI')
if mongo_uri:
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
client = MongoClient(mongo_uri, uuidRepresentation='standard')
db = client['quant_trading']

# Insert Functions
def save_strategy(strategy_doc):
    try:
        if '_id' in strategy_doc:
            result = db.strategies.replace_one({'_id': strategy_doc['_id']}, strategy_doc, upsert=True)
            print(f"Upserted strategy with _id: {strategy_doc['_id']}")
        else:
            result = db.strategies.insert_one(strategy_doc)
            print(f"Inserted strategy with _id: {result.inserted_id}")
    except PyMongoError as e:
        print(f"Error saving strategy: {e}")

def save_experiment(experiment_doc):
    try:
        if 'experiment_name' not in experiment_doc:
            # Auto-generate
            import random
            import string
            animals = ["Tiger","Wolf","Falcon","Shark","Eagle","Panther"]
            rand = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
            experiment_doc['experiment_name'] = f"{random.choice(animals)}-{rand}"
        if '_id' in experiment_doc:
            result = db.experiments.replace_one({'_id': experiment_doc['_id']}, experiment_doc, upsert=True)
            print(f"Upserted experiment '{experiment_doc['experiment_name']}' with _id: {experiment_doc['_id']}")
        else:
            result = db.experiments.insert_one(experiment_doc)
            print(f"Inserted experiment '{experiment_doc['experiment_name']}' with _id: {result.inserted_id}")
    except PyMongoError as e:
        print(f"Error saving experiment: {e}")

def save_optimization_run(opt_run_doc):
    try:
        # Ensure experiment_id and experiment_name are present
        if 'experiment_id' not in opt_run_doc or 'experiment_name' not in opt_run_doc:
            raise ValueError("experiment_id and experiment_name are required in opt_run_doc")
        if '_id' in opt_run_doc:
            result = db.optimization_runs.replace_one({'_id': opt_run_doc['_id']}, opt_run_doc, upsert=True)
            print(f"Upserted optimization_run with _id: {opt_run_doc['_id']}")
        else:
            result = db.optimization_runs.insert_one(opt_run_doc)
            print(f"Inserted optimization_run with _id: {result.inserted_id}")
    except PyMongoError as e:
        print(f"Error saving optimization_run: {e}")

def save_iteration(iter_doc):
    try:
        if '_id' in iter_doc:
            result = db.iterations.replace_one({'_id': iter_doc['_id']}, iter_doc, upsert=True)
            print(f"Upserted iteration with _id: {iter_doc['_id']}")
        else:
            result = db.iterations.insert_one(iter_doc)
            print(f"Inserted iteration with _id: {result.inserted_id}")
    except PyMongoError as e:
        print(f"Error saving iteration: {e}")

def save_backtest(backtest_doc, metrics=None, composite=None):
    try:
        # FIX: Explicitly persist train and test metrics if present
        if 'train_metrics' in backtest_doc:
            db.backtests.update_one(
                {'_id': backtest_doc['_id']},
                {'$set': {'train_metrics': backtest_doc['train_metrics']}},
                upsert=True
            )
        if 'test_metrics' in backtest_doc:
            db.backtests.update_one(
                {'_id': backtest_doc['_id']},
                {'$set': {'test_metrics': backtest_doc['test_metrics']}},
                upsert=True
            )

        # Legacy fallback: if caller only provided a single metrics dict
        if metrics and 'train_metrics' not in backtest_doc and 'test_metrics' not in backtest_doc:
            backtest_doc['metrics'] = metrics  # FIX: keep backwards compatibility

        if composite:
            backtest_doc['composite'] = composite

        if '_id' in backtest_doc:
            doc_to_set = {k: v for k, v in backtest_doc.items() if k != '_id'}
            db.backtests.update_one({'_id': backtest_doc['_id']}, {'$set': doc_to_set}, upsert=True)
        else:
            db.backtests.insert_one(backtest_doc)

    except PyMongoError as e:
        print(f"Error saving backtest: {e}")

def get_backtest_by_id(backtest_id):
    try:
        backtest = db.backtests.find_one({'_id': backtest_id})
        if backtest:
            print(f"Retrieved backtest with _id: {backtest_id}")
        else:
            print(f"No backtest found with _id: {backtest_id}")
        return backtest
    except PyMongoError as e:
        print(f"Error retrieving backtest: {e}")
        return None

def save_trade(trade_doc):
    try:
        if '_id' in trade_doc:
            result = db.trades.replace_one({'_id': trade_doc['_id']}, trade_doc, upsert=True)
            print(f"Upserted trade with _id: {trade_doc['_id']}")
        else:
            result = db.trades.insert_one(trade_doc)
            print(f"Inserted trade with _id: {result.inserted_id}")
    except PyMongoError as e:
        print(f"Error saving trade: {e}")

def save_trades(backtest_id, trades_list):
    try:
        if trades_list:
            for trade in trades_list:
                trade['backtest_id'] = backtest_id
            result = db.trades.insert_many(trades_list)
            print(f"Inserted {len(result.inserted_ids)} trades")
        else:
            print("No trades to insert")
    except PyMongoError as e:
        print(f"Error saving trades: {e}")

def save_equity_curve(backtest_id, eq_list, scope="daily"):
    try:
        if eq_list:
            for item in eq_list:
                item['backtest_id'] = backtest_id
                item['scope'] = scope
            result = db.equity_curves.insert_many(eq_list)
            print(f"Inserted {len(result.inserted_ids)} equity curve points with scope {scope}")
        else:
            print("No equity curve points to insert")
    except PyMongoError as e:
        print(f"Error saving equity curve: {e}")

def save_rolling_metrics(backtest_id, metrics_list):
    try:
        if metrics_list:
            for item in metrics_list:
                item['backtest_id'] = backtest_id
            result = db.rolling_metrics.insert_many(metrics_list)
            print(f"Inserted {len(result.inserted_ids)} rolling metrics")
        else:
            print("No rolling metrics to insert")
    except PyMongoError as e:
        print(f"Error saving rolling metrics: {e}")

def save_balance_curve(backtest_id, balance_list):
    try:
        if balance_list:
            for item in balance_list:
                item['backtest_id'] = backtest_id
            result = db.balance_curves.insert_many(balance_list)
            print(f"Inserted {len(result.inserted_ids)} balance curve points")
        else:
            print("No balance curve points to insert")
    except PyMongoError as e:
        print(f"Error saving balance curve: {e}")

# Query Functions
def get_strategies():
    try:
        strategies = list(db.strategies.find())
        print(f"Retrieved {len(strategies)} strategies")
        return strategies
    except PyMongoError as e:
        print(f"Error retrieving strategies: {e}")
        return []

def get_experiments():
    try:
        experiments = list(db.experiments.find())
        result = []
        for exp in experiments:
            name = exp.get('experiment_name', f"Legacy-{str(exp['_id'])[:6]}")
            result.append({
                "id": exp["_id"],
                "name": name,
                "timestamp": exp.get("timestamp")
            })
        print(f"Retrieved {len(result)} experiments")
        return result
    except PyMongoError as e:
        print(f"Error retrieving experiments: {e}")
        return []

def get_best_optimization_run(stock, timeframe, strategy_id, experiment_id=None):
    try:
        query = {'stock_symbol': stock, 'timeframe': timeframe, 'strategy_id': strategy_id}
        if experiment_id:
            query['experiment_id'] = experiment_id
        opt_run = db.optimization_runs.find_one(query, sort=[('best_value', -1)])
        if opt_run:
            print(f"Retrieved best optimization run for {stock}, {timeframe}, {strategy_id}" + (f", experiment {experiment_id}" if experiment_id else ""))
        else:
            print(f"No optimization run found for {stock}, {timeframe}, {strategy_id}" + (f", experiment {experiment_id}" if experiment_id else ""))
        return opt_run
    except PyMongoError as e:
        print(f"Error retrieving best optimization run: {e}")
        return None

def get_iterations(opt_run_id=None, window_id=None, phase=None):
    try:
        query = {}
        if opt_run_id:
            query['optimization_run_id'] = uuid.UUID(opt_run_id)
        if window_id:
            query['window_id'] = window_id
        if phase:
            query['phase'] = phase
        iterations = list(db.iterations.find(query))
        filter_desc = []
        if opt_run_id:
            filter_desc.append(f"opt_run_id {opt_run_id}")
        if window_id:
            filter_desc.append(f"window_id {window_id}")
        if phase:
            filter_desc.append(f"phase {phase}")
        print(f"Retrieved {len(iterations)} iterations for " + ", ".join(filter_desc))
        return iterations
    except PyMongoError as e:
        print(f"Error retrieving iterations: {e}")
        return []

def get_optimization_runs(experiment_id=None, strategy_id=None):
    try:
        query = {}
        if experiment_id:
            query['experiment_id'] = experiment_id
        if strategy_id:
            query['strategy_id'] = strategy_id
        opt_runs = list(db.optimization_runs.find(query))
        print(f"Retrieved {len(opt_runs)} optimization runs for experiment_id {experiment_id}")
        return opt_runs
    except PyMongoError as e:
        print(f"Error retrieving optimization runs: {e}")
        return []

def save_window(window_doc):
    try:
        db.windows.insert_one(window_doc)
        print(f"Inserted window with _id: {window_doc['_id']}")
    except PyMongoError as e:
        print(f"Error saving window: {e}")

def get_windows(opt_run_id):
    try:
        windows = list(db.windows.find({'optimization_run_id': uuid.UUID(opt_run_id)}))
        print(f"Retrieved {len(windows)} windows for opt_run_id {opt_run_id}")
        return windows
    except PyMongoError as e:
        print(f"Error retrieving windows: {e}")
        return []

def get_backtests(opt_run_id, window_number=None):
    try:
        query = {'optimization_run_id': uuid.UUID(opt_run_id)}
        if window_number:
            query['window_number'] = window_number
        backtests = list(db.backtests.find(query))
        print(f"Retrieved {len(backtests)} backtests for opt_run_id {opt_run_id}" + (f", window {window_number}" if window_number else ""))
        return backtests
    except PyMongoError as e:
        print(f"Error retrieving backtests: {e}")
        return []

def get_trades(backtest_id):
    try:
        trades = list(db.trades.find({'backtest_id': backtest_id}))
        print(f"Retrieved {len(trades)} trades for backtest_id {backtest_id}")
        return trades
    except PyMongoError as e:
        print(f"Error retrieving trades: {e}")
        return []

def get_equity_curve(backtest_id, scope="daily"):
    try:
        eq_curve = list(db.equity_curves.find({'backtest_id': backtest_id, 'scope': scope}))
        print(f"Retrieved {len(eq_curve)} equity curve points for backtest_id {backtest_id}, scope {scope}")
        return eq_curve
    except PyMongoError as e:
        print(f"Error retrieving equity curve: {e}")
        return []

def get_rolling_metrics(backtest_id):
    try:
        metrics = list(db.rolling_metrics.find({'backtest_id': backtest_id}))
        print(f"Retrieved {len(metrics)} rolling metrics for backtest_id {backtest_id}")
        return metrics
    except PyMongoError as e:
        print(f"Error retrieving rolling metrics: {e}")
        return []


def get_balance_curve(backtest_id):
    try:
        balance_curve = list(db.balance_curves.find({'backtest_id': backtest_id}))
        print(f"Retrieved {len(balance_curve)} balance curve points for backtest_id {backtest_id}")
        return balance_curve
    except PyMongoError as e:
        print(f"Error retrieving balance curve: {e}")
        return []