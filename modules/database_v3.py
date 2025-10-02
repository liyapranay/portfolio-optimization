import sqlite3
import json
import uuid

DB_PATH = '../backtest.db'

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn = get_connection()
    c = conn.cursor()

    # strategies
    c.execute('''CREATE TABLE IF NOT EXISTS strategies (
        strategy_id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        asset_class TEXT,
        timeframe TEXT
    )''')

    # experiments
    c.execute('''CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        description TEXT,
        created_at TEXT
    )''')

    # backtest_runs
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_runs (
        backtest_id TEXT PRIMARY KEY,
        timestamp TEXT,
        strategy_id TEXT,
        params TEXT,
        stock_symbol TEXT,
        timeframe TEXT,
        start_date TEXT,
        end_date TEXT,
        initial_cash REAL,
        final_value REAL,
        window_id TEXT,
        experiment_id TEXT,
        FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id),
        FOREIGN KEY (window_id) REFERENCES windows(id),
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    )''')

    # optimization_runs
    c.execute('''CREATE TABLE IF NOT EXISTS optimization_runs (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        strategy_id TEXT,
        optimize_metric TEXT,
        best_params TEXT,
        best_value REAL,
        backtest_id TEXT,
        experiment_id TEXT,
        scope TEXT,
        FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id),
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id),
        FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
    )''')

    # optimization_iterations
    c.execute('''CREATE TABLE IF NOT EXISTS optimization_iterations (
        id TEXT PRIMARY KEY,
        datetime TEXT,
        iteration INTEGER,
        optimize_metric TEXT,
        target REAL,
        best_so_far REAL,
        parameters TEXT,
        phase TEXT,
        backtest_id TEXT,
        window_id TEXT,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id),
        FOREIGN KEY (window_id) REFERENCES windows(id)
    )''')

    # windows
    c.execute('''CREATE TABLE IF NOT EXISTS windows (
        id TEXT PRIMARY KEY,
        optimization_run_id TEXT,
        window_number INTEGER,
        train_start TEXT,
        train_end TEXT,
        test_start TEXT,
        test_end TEXT,
        FOREIGN KEY (optimization_run_id) REFERENCES optimization_runs(id)
    )''')

    # performance_metrics
    c.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id TEXT,
        metric_name TEXT,
        value REAL,
        scope TEXT,
        phase TEXT,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')

    # composite_metrics
    c.execute('''CREATE TABLE IF NOT EXISTS composite_metrics (
        id TEXT PRIMARY KEY,
        backtest_id TEXT,
        sharpe REAL,
        sortino REAL,
        calmar REAL,
        profit_factor REAL,
        composite_score REAL,
        scope TEXT,
        phase TEXT,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')

    # trade_logs
    c.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
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

    # equity_curve
    c.execute('''CREATE TABLE IF NOT EXISTS equity_curve (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id TEXT,
        datetime TEXT,
        equity REAL,
        return REAL,
        drawdown REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')

    # rolling_metrics
    c.execute('''CREATE TABLE IF NOT EXISTS rolling_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        backtest_id TEXT,
        datetime TEXT,
        sharpe REAL,
        drawdown REAL,
        sortino REAL,
        FOREIGN KEY (backtest_id) REFERENCES backtest_runs(backtest_id)
    )''')

    # Indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_backtest_stock_timeframe ON backtest_runs(stock_symbol, timeframe);')
    c.execute('CREATE INDEX IF NOT EXISTS idx_iterations_phase ON optimization_iterations(phase);')
    c.execute('CREATE INDEX IF NOT EXISTS idx_perf_scope_phase ON performance_metrics(scope, phase);')

    conn.commit()
    conn.close()

def save_strategy(strategy_id, name, description, asset_class, timeframe):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT OR IGNORE INTO strategies VALUES (?, ?, ?, ?, ?)', (strategy_id, name, description, asset_class, timeframe))
        conn.commit()
        print(f"Saved strategy: {strategy_id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving strategy {strategy_id}: {e}")
        raise
    finally:
        conn.close()

def save_experiment(experiment_id, description, created_at):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO experiments VALUES (?, ?, ?)', (experiment_id, description, created_at))
        conn.commit()
        print(f"Saved experiment: {experiment_id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving experiment {experiment_id}: {e}")
        raise
    finally:
        conn.close()

def save_backtest_run(backtest_id, timestamp, strategy_id, params, stock_symbol, timeframe, start_date, end_date, initial_cash, final_value, window_id, experiment_id):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO backtest_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (backtest_id, timestamp, strategy_id, json.dumps(params), stock_symbol, timeframe, start_date, end_date, initial_cash, final_value, window_id, experiment_id))
        conn.commit()
        print(f"Saved backtest run: {backtest_id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving backtest run {backtest_id}: {e}")
        raise
    finally:
        conn.close()

def save_optimization_run(id, timestamp, strategy_id, optimize_metric, best_params, best_value, backtest_id, experiment_id, scope):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT OR REPLACE INTO optimization_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (id, timestamp, strategy_id, optimize_metric, json.dumps(best_params), best_value, backtest_id, experiment_id, scope))
        conn.commit()
        print(f"Saved optimization run: {id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving optimization run {id}: {e}")
        raise
    finally:
        conn.close()

def save_iteration(iter_id, datetime, iteration, metric, target, best_so_far, params_json, phase, backtest_id, window_id):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO optimization_iterations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (iter_id, datetime, iteration, metric, target, best_so_far, params_json, phase, backtest_id, window_id))
        conn.commit()
        print(f"Saved iteration: {iter_id}, phase: {phase}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving iteration {iter_id}: {e}")
        raise
    finally:
        conn.close()

def save_window(id, optimization_run_id, window_number, train_start, train_end, test_start, test_end):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('INSERT INTO windows VALUES (?, ?, ?, ?, ?, ?, ?)', (id, optimization_run_id, window_number, train_start, train_end, test_start, test_end))
        conn.commit()
        print(f"Saved window: {id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving window {id}: {e}")
        raise
    finally:
        conn.close()

def save_performance_metrics(backtest_id, metrics_dict, scope, phase):
    conn = get_connection()
    c = conn.cursor()
    try:
        for name, value in metrics_dict.items():
            c.execute('INSERT INTO performance_metrics (backtest_id, metric_name, value, scope, phase) VALUES (?, ?, ?, ?, ?)', (backtest_id, name, value, scope, phase))
        conn.commit()
        print(f"Saved performance metrics for backtest: {backtest_id}, scope: {scope}, phase: {phase}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving performance metrics for {backtest_id}: {e}")
        raise
    finally:
        conn.close()

def save_composite_metrics(backtest_id, sharpe, sortino, calmar, profit_factor, composite_score, scope, phase):
    conn = get_connection()
    c = conn.cursor()
    try:
        id = str(uuid.uuid4())
        c.execute('INSERT INTO composite_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (id, backtest_id, sharpe, sortino, calmar, profit_factor, composite_score, scope, phase))
        conn.commit()
        print(f"Saved composite metrics for backtest: {backtest_id}, scope: {scope}, phase: {phase}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving composite metrics for {backtest_id}: {e}")
        raise
    finally:
        conn.close()

def save_trade_logs(backtest_id, trade_list):
    conn = get_connection()
    c = conn.cursor()
    try:
        for trade in trade_list:
            c.execute('INSERT INTO trade_logs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', (backtest_id, trade['Datetime'], trade['Transaction type'], trade['Price'], trade['Qty'], trade['Profit/Loss'], trade['Commission'], trade['Actual Profit/Loss'], trade['Cumulative run'], trade['Strategy Type'], trade['Exit Reason']))
        conn.commit()
        print(f"Saved trade logs for backtest: {backtest_id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving trade logs for {backtest_id}: {e}")
        raise
    finally:
        conn.close()

def save_equity_curve(backtest_id, equity_data):
    conn = get_connection()
    c = conn.cursor()
    try:
        for dt, eq, ret, drawdown in equity_data:
            c.execute('INSERT INTO equity_curve (backtest_id, datetime, equity, return, drawdown) VALUES (?, ?, ?, ?, ?)', (backtest_id, dt, eq, ret, drawdown))
        conn.commit()
        print(f"Saved equity curve for backtest: {backtest_id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving equity curve for {backtest_id}: {e}")
        raise
    finally:
        conn.close()

def save_rolling_metrics(backtest_id, metrics_list):
    conn = get_connection()
    c = conn.cursor()
    try:
        for dt, sharpe, drawdown, sortino in metrics_list:
            c.execute('INSERT INTO rolling_metrics (backtest_id, datetime, sharpe, drawdown, sortino) VALUES (?, ?, ?, ?, ?)', (backtest_id, dt, sharpe, drawdown, sortino))
        conn.commit()
        print(f"Saved rolling metrics for backtest: {backtest_id}")
    except sqlite3.IntegrityError as e:
        print(f"Error saving rolling metrics for {backtest_id}: {e}")
        raise
    finally:
        conn.close()

# Create tables on import
create_tables()

if __name__ == '__main__':
    pass