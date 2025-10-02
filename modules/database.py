import sqlite3
import pandas as pd
import json
import os

DB_PATH = '../backtest.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_runs (
        backtest_id TEXT PRIMARY KEY,
        timestamp TEXT,
        strategy_name TEXT,
        params TEXT,
        start_date TEXT,
        end_date TEXT,
        initial_cash REAL,
        final_value REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
        id INTEGER PRIMARY KEY,
        backtest_id TEXT,
        datetime TEXT,
        transaction_type TEXT,
        price REAL,
        qty REAL,
        pnl REAL,
        commission REAL,
        actual_pnl REAL,
        cumulative_pnl REAL,
        strategy_type TEXT,
        exit_reason TEXT,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY,
        backtest_id TEXT,
        metric_name TEXT,
        value REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS optimization_runs (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        strategy_name TEXT,
        best_params TEXT,
        optimize_metric TEXT,
        best_value REAL,
        backtest_id TEXT,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS optimizer (
        id TEXT,
        datetime TEXT,
        iteration INTEGER,
        fixed_parameter TEXT,
        parameter TEXT,
        cmd_parameter TEXT,
        optimize_metric TEXT,
        target REAL,
        backtest_id TEXT,
        FOREIGN KEY (id) REFERENCES optimization_runs(id),
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')
    # Add column if not exists (for existing tables)
    try:
        c.execute('ALTER TABLE optimizer ADD COLUMN cmd_parameter TEXT')
    except sqlite3.OperationalError:
        pass  # Column already exists
    c.execute('''CREATE TABLE IF NOT EXISTS equity_curve (
        id INTEGER PRIMARY KEY,
        backtest_id TEXT,
        datetime TEXT,
        portfolio_value REAL,
        cumulative_return REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')
    conn.commit()
    conn.close()

def save_backtest_run(backtest_id, timestamp, strategy_name, params, start_date, end_date, initial_cash, final_value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
              (backtest_id, timestamp, strategy_name, json.dumps(params), start_date, end_date, initial_cash, final_value))
    conn.commit()
    conn.close()

def save_trade_logs(backtest_id, trades):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for trade in trades:
        c.execute('INSERT INTO trade_logs (backtest_id, datetime, transaction_type, price, qty, pnl, commission, actual_pnl, cumulative_pnl, strategy_type, exit_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                  (backtest_id, trade['Datetime'], trade['Transaction type'], trade['Price'], trade['Qty'], trade['Profit/Loss'], trade['Commission'], trade['Actual Profit/Loss'], trade['Cumulative run'], trade['Strategy Type'], trade['Exit Reason']))
    conn.commit()
    conn.close()

def save_performance_metrics(backtest_id, metrics):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for metric, value in metrics.items():
        c.execute('INSERT INTO performance_metrics (backtest_id, metric_name, value) VALUES (?, ?, ?)',
                  (backtest_id, metric, value))
    conn.commit()
    conn.close()

def save_equity_curve(backtest_id, equity_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for dt, val, ret in equity_data:
        c.execute('INSERT INTO equity_curve (backtest_id, datetime, portfolio_value, cumulative_return) VALUES (?, ?, ?, ?)',
                  (backtest_id, dt, val, ret))
    conn.commit()
    conn.close()

def get_performance_metrics(backtest_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT metric_name, value FROM performance_metrics WHERE backtest_id = ?', (backtest_id,))
    metrics = dict(c.fetchall())
    conn.close()
    return metrics

def get_trade_stats(backtest_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM trade_logs WHERE backtest_id = ?', (backtest_id,))
    total_trades = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM trade_logs WHERE backtest_id = ? AND pnl > 0', (backtest_id,))
    wins = c.fetchone()[0]
    losses = total_trades - wins
    c.execute('SELECT SUM(commission) FROM trade_logs WHERE backtest_id = ?', (backtest_id,))
    total_commission = c.fetchone()[0] or 0
    # For streaks, assuming we store them in metrics, or calculate
    # For simplicity, return what we have
    conn.close()
    return total_trades, wins, losses, total_commission

def get_sell_trades(backtest_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM trade_logs WHERE backtest_id = ? AND transaction_type = "SELL"', (backtest_id,))
    columns = [desc[0] for desc in c.description]
    rows = c.fetchall()
    sell_df = pd.DataFrame(rows, columns=columns)
    conn.close()
    return sell_df

def get_equity_curve(backtest_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT datetime, portfolio_value, cumulative_return FROM equity_curve WHERE backtest_id = ? ORDER BY datetime', (backtest_id,))
    equity_data = c.fetchall()
    conn.close()
    return equity_data

def get_backtest_run(backtest_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM backtest_runs WHERE backtest_id = ?', (backtest_id,))
    row = c.fetchone()
    conn.close()
    return row

def save_optimization_run(opt_id, timestamp, strategy_name, best_params, optimize_metric, best_value, backtest_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO optimization_runs (id, timestamp, strategy_name, best_params, optimize_metric, best_value, backtest_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (opt_id, timestamp, strategy_name, json.dumps(best_params), optimize_metric, best_value, backtest_id))
    conn.commit()
    conn.close()

def save_optimizer_iteration(opt_id, datetime, iteration, fixed_parameter, parameter, cmd_parameter, optimize_metric, target, backtest_id=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO optimizer (id, datetime, iteration, fixed_parameter, parameter, cmd_parameter, optimize_metric, target, backtest_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
              (opt_id, datetime, iteration, json.dumps(fixed_parameter), json.dumps(parameter), json.dumps(cmd_parameter), optimize_metric, target, backtest_id))
    conn.commit()
    conn.close()

# Initialize DB on import
init_db()