import subprocess
import json
import argparse
import sys
sys.path.append('..')
import modules.database as db
from bayes_opt import BayesianOptimization
import numpy as np
import uuid
import pandas as pd

def run_backtest(params, script_path, fixed_args):
    # Convert params to appropriate types
    int_params = ['ema_fast', 'ema_slow', 'adx_threshold', 'mfi_ma_threshold', 'profit_target','momentum_threshold','natr_multiplier']
    for p in int_params:
        if p in params:
            params[p] = int(round(params[p]))
    # Float params remain as float

    # Convert params to command line args
    cmd = ['python3', script_path]
    for key, value in fixed_args.items():
        cmd.extend(['--' + key, str(value)])
    for key, value in params.items():
        cmd.extend(['--' + key, str(value)])
    # Run the backtest
    # print('\n'.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    if result.returncode != 0:
        print(f"Error running backtest: {result.stderr}")
        return -np.inf, None  # Penalize failed runs
    # Parse the output for the optimized metric and backtest_id
    lines = result.stdout.split('\n')
    metric_value = None
    backtest_id = None
    for line in lines:
        if line.startswith('Optimized Metric'):
            metric_value = float(line.split(':')[1].strip())
        elif line.startswith('Backtest UUID:'):
            backtest_id = line.split(':')[1].strip()
    if metric_value is None:
        print("Optimized Metric not found in output")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return -np.inf, None
    return metric_value, backtest_id

def optimize_strategy(script_path, param_bounds, fixed_args, optimize_metric, n_iter=50):
    opt_run_id = str(uuid.uuid4())
    print(f"Optimization UUID: {opt_run_id}")
    iteration_counter = 0

    def objective(**params):
        nonlocal iteration_counter
        iteration_counter += 1
        # Create cmd for logging
        cmd = ['python3', script_path]
        for key, value in fixed_args.items():
            cmd.extend(['--' + key, str(value)])
        for key, value in params.items():
            cmd.extend(['--' + key, str(value)])
        cmd_str = ' '.join(cmd)
        metric, backtest_id = run_backtest(params, script_path, fixed_args)
        db.save_optimizer_iteration(opt_run_id, str(pd.Timestamp.now()), iteration_counter, fixed_args, params, cmd_str, optimize_metric, metric, backtest_id)
        return metric

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=42,
    )

    optimizer.maximize(init_points=5, n_iter=n_iter)

    best_params = optimizer.max['params']
    best_value = optimizer.max['target']

    # Run the best params again to get the backtest_id
    cmd = ['python3', script_path]
    for k, v in {**fixed_args, **best_params}.items():
        cmd.extend(['--' + k, str(v)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.split('\n')
    backtest_id = None
    for line in lines:
        if line.startswith('Backtest UUID:'):
            backtest_id = line.split(':')[1].strip()
            break

    # Save to DB
    db.save_optimization_run(opt_run_id, str(pd.Timestamp.now()), 'ai_v3_streak', best_params, optimize_metric, best_value, backtest_id)

    # If backtest_id is None, populate from optimizer iterations
    if backtest_id is None:
        import sqlite3
        conn = sqlite3.connect('../backtest.db')
        c = conn.cursor()
        c.execute('SELECT backtest_id FROM optimizer WHERE id = ? ORDER BY target DESC LIMIT 1', (opt_run_id,))
        row = c.fetchone()
        if row and row[0]:
            backtest_id = row[0]
            c.execute('UPDATE optimization_runs SET backtest_id = ? WHERE id = ?', (backtest_id, opt_run_id))
        conn.commit()
        conn.close()

    print(f"Best Backtest UUID: {backtest_id}")
    print(f"Best Params: {best_params}")
    print(f"Best {optimize_metric}: {best_value}")

if __name__ == '__main__':
    # Initialize parameters
    script_path = 'ai_v3_streak_backtesting.py'
    param_bounds = {
        'ema_fast': (5, 15),
        'ema_slow': (15, 30),
        'adx_threshold': (15, 30),
        'momentum_threshold': (-5, 5),
        'mfi_ma_threshold': (60, 80),
        'natr_multiplier': (0.5, 2.0),
        'profit_target': (200, 300)
    }
    fixed_args = {
        'data_file': '../data/5minute/910_NSE_EICHERMOT.csv',
        'start_date': '2025-01-01',
        'end_date': '2025-09-25',
        'initial_cash': 15000,
        'intraday': 'False',
        'stoploss_pct': 0.07,
        'takeprofit_pct': 0.5,
        'strategy_name': 'ai_v3_streak',
        'optimize_metric': 'Net Profit'
    }
    optimize_metric = 'Net Profit'
    n_iter = 1

    optimize_strategy(script_path, param_bounds, fixed_args, optimize_metric, n_iter)