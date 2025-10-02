import subprocess
import json
import argparse
import sys
import random
import string
import logging
import os
sys.path.append('..')
from modules.database_mongo_helper import *
from bayes_opt import BayesianOptimization
import numpy as np
import uuid
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import Counter

# Strategy file being optimized
STRATEGY_FILE = 'ai_v3_streak_backtesting.py'

# Parameter bounds
param_bounds = {
    'ema_fast': (5, 15),          # int
    'ema_slow': (15, 30),         # int
    'adx_threshold': (15, 30),    # int
    'momentum_threshold': (-5, 5),# float
    'mfi_ma_threshold': (60, 80), # int
    'natr_multiplier': (0.5, 2.0),# float
    'profit_target': (200, 300)   # int
}

# Parameter types
int_params = ['ema_fast', 'ema_slow', 'adx_threshold', 'mfi_ma_threshold', 'profit_target']
float_params = ['momentum_threshold', 'natr_multiplier']

def generate_random_description():
    animals = ['Lion', 'Tiger', 'Elephant', 'Giraffe', 'Zebra', 'Panda', 'Koala', 'Kangaroo', 'Penguin', 'Dolphin']
    animal = random.choice(animals)
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{animal}-{suffix}"

def run_backtest(params, script_path, fixed_args):
    # Convert params to appropriate types
    for p in int_params:
        if p in params:
            params[p] = int(round(params[p]))
    for p in float_params:
        if p in params:
            params[p] = float(params[p])

    # Convert params to command line args
    cmd = ['python3', script_path]
    for key, value in fixed_args.items():
        cmd.extend(['--' + key, str(value)])
    for key, value in params.items():
        cmd.extend(['--' + key, str(value)])
    # Run the backtest
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    if result.returncode != 0:
        print(f"Error running backtest: {result.stderr}")
        return {'error': True}, None  # Return dict with error

    # Parse metrics from output
    metrics = {}
    backtest_id = None
    lines = result.stdout.split('\n')
    for line in lines:
        if line.startswith('Backtest UUID:'):
            backtest_id = line.split('Backtest UUID:')[1].strip()
        elif ': ' in line and not line.startswith('[') and not line.startswith('Backtest') and not line.startswith('Initial') and not line.startswith('Starting') and not line.startswith('Final') and not line.startswith('Optimized'):
            parts = line.split(': ')
            if len(parts) == 2:
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    metrics[key] = value
                except ValueError:
                    pass
    return metrics, backtest_id

def compute_composite_score(metrics):
    sharpe = metrics.get('Sharpe Ratio', 0)
    sortino = metrics.get('Sortino Ratio', 0)
    calmar = metrics.get('Calmar Ratio', 0)
    pf = metrics.get('Profit Factor', 0)
    return 0.3 * sharpe + 0.3 * sortino + 0.3 * calmar + 0.1 * pf

def random_search(script_path, param_bounds, fixed_args, optimize_metric, window_id, N, top_k, logger):
    samples = []
    best_so_far = -1e10
    logger.info("Running Random Search with %d samples", N)
    for i in range(N):
        params = {}
        for p, (low, high) in param_bounds.items():
            if p in int_params:
                params[p] = np.random.randint(low, high + 1)
            else:
                params[p] = np.random.uniform(low, high)
        metrics, backtest_id = run_backtest(params, script_path, fixed_args)
        if 'error' in metrics:
            value = -1e10
        elif optimize_metric == 'Composite Score':
            value = compute_composite_score(metrics)
        else:
            value = metrics.get(optimize_metric, -1e10)
        if value > best_so_far:
            best_so_far = value
        samples.append((params, value))
        logger.debug("Random Sample %d: Params %s, %s: %f, Best so far: %f", i+1, params, optimize_metric, value, best_so_far)
        # Save to DB
        save_iteration({'_id': str(uuid.uuid4()), 'timestamp': str(pd.Timestamp.now()), 'iteration': i + 1, 'target_metric': optimize_metric, 'target': value, 'best_so_far': best_so_far, 'parameters': params, 'phase': 'random', 'backtest_id': backtest_id, 'window_id': window_id})
    # Sort by value descending
    samples.sort(key=lambda x: x[1], reverse=True)
    top_k_params = samples[:top_k]
    logger.info("Top %d params from Random Search", top_k)
    return top_k_params

def bayesian_optimize(script_path, top_k_params, fixed_args, optimize_metric, window_id, iterations=30, logger=None):
    # Narrow bounds to min/max of top K
    narrowed_bounds = {}
    for p in param_bounds:
        values = [d[p] for d, _ in top_k_params]
        narrowed_bounds[p] = (min(values), max(values))
    logger.info("Narrowed bounds for Bayesian: %s", narrowed_bounds)
    iter_count = 0
    best_so_far = -1e10
    def objective(**params):
        nonlocal iter_count, best_so_far
        iter_count += 1
        metrics, backtest_id = run_backtest(params, script_path, fixed_args)
        if 'error' in metrics:
            value = -1e10
        elif optimize_metric == 'Composite Score':
            value = compute_composite_score(metrics)
        else:
            value = metrics.get(optimize_metric, -1e10)
        if value > best_so_far:
            best_so_far = value
        logger.debug("Bayesian Iteration %d: Params %s, %s: %f, Best so far: %f", iter_count, params, optimize_metric, value, best_so_far)
        # Save to DB
        save_iteration({'_id': str(uuid.uuid4()), 'timestamp': str(pd.Timestamp.now()), 'iteration': iter_count, 'target_metric': optimize_metric, 'target': value, 'best_so_far': best_so_far, 'parameters': params, 'phase': 'bayesian', 'backtest_id': backtest_id, 'window_id': window_id})
        return value

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=narrowed_bounds,
        random_state=42,
    )
    optimizer.maximize(init_points=0, n_iter=iterations)
    best_params = optimizer.max['params']
    best_value = optimizer.max['target']
    print(f"Bayesian Best: Params {best_params}, {optimize_metric}: {best_value}")
    return best_params, best_value

def hybrid_optimize_strategy(script_path, param_bounds, fixed_args, optimize_metric, train_years, test_months, start_date, end_date, random_samples, top_k, bayes_iter):
    opt_run_id = str(uuid.uuid4())
    experiment_id = opt_run_id
    print(f"Hybrid Optimization UUID: {opt_run_id}")

    strategy_id = fixed_args.get('strategy_name', 'unknown')
    experiment_description = generate_random_description()

    ticker = fixed_args['data_file'].split("_")[-1].split(".")[0]

    # Setup logging
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"../logs/{experiment_description}_{timestamp_str}.log"
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Optimization Run started for strategy %s on stock %s", strategy_id, ticker)

    save_experiment({'_id': experiment_id, 'description': experiment_description, 'timestamp': str(pd.Timestamp.now())})
    save_optimization_run({'_id': opt_run_id, 'timestamp': str(pd.Timestamp.now()), 'strategy_id': strategy_id, 'optimize_metric': optimize_metric, 'best_params': {}, 'best_value': 0, 'best_backtest_id': None, 'experiment_id': experiment_id, 'phase': 'out-of-sample', 'stock_symbol': ticker, 'timeframe': '5minute'})
    print(f"Experiment Name: {experiment_description}")

    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    window_results = []
    param_sets = []

    current_train_start = start
    iteration = 0

    while current_train_start < end:
        iteration += 1
        train_end = current_train_start + relativedelta(years=train_years)
        if train_end >= end:
            break
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            test_end = end

        print(f"\n--- Window {iteration} ---")
        print(f"Training: {current_train_start.date()} to {train_end.date()}")
        print(f"Testing: {train_end.date()} to {test_end.date()}")

        window_id = str(uuid.uuid4())
        save_window({'_id': window_id, 'optimization_run_id': opt_run_id, 'window_number': iteration, 'train_start': current_train_start.strftime('%Y-%m-%d'), 'train_end': train_end.strftime('%Y-%m-%d'), 'test_start': train_end.strftime('%Y-%m-%d'), 'test_end': test_end.strftime('%Y-%m-%d')})

        # Training phase
        train_fixed_args = fixed_args.copy()
        train_fixed_args['start_date'] = current_train_start.strftime('%Y-%m-%d')
        train_fixed_args['end_date'] = train_end.strftime('%Y-%m-%d')
        train_fixed_args['experiment_id'] = experiment_id
        train_fixed_args['window_id'] = window_id

        # Random Search
        top_k_params = random_search(script_path, param_bounds, train_fixed_args, optimize_metric, window_id, random_samples, top_k, logger)

        # Bayesian Optimization
        best_params, best_value = bayesian_optimize(script_path, top_k_params, train_fixed_args, optimize_metric, window_id, bayes_iter, logger)

        # Testing phase
        test_fixed_args = fixed_args.copy()
        test_fixed_args['start_date'] = train_end.strftime('%Y-%m-%d')
        test_fixed_args['end_date'] = test_end.strftime('%Y-%m-%d')
        test_fixed_args['experiment_id'] = experiment_id
        test_fixed_args['window_id'] = window_id

        test_metrics, test_backtest_id = run_backtest(best_params, script_path, test_fixed_args)
        if 'error' in test_metrics:
            test_value = -1e10
        elif optimize_metric == 'Composite Score':
            test_value = compute_composite_score(test_metrics)
        else:
            test_value = test_metrics.get(optimize_metric, -1e10)

        # Metrics saved in backtest by ai_v3_streak_backtesting.py

        print(f"Out-of-sample {optimize_metric}: {test_value}")

        window_results.append({
            'iteration': iteration,
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'best_params': best_params,
            'in_sample_value': best_value,
            'out_sample_value': test_value,
            'test_backtest_id': test_backtest_id
        })

        param_sets.append(tuple(sorted(best_params.items())))

        # Move window
        current_train_start = train_end - relativedelta(months=test_months)  # Overlap

    # Determine most robust params (majority vote)
    if param_sets:
        most_common = Counter(param_sets).most_common(1)[0][0]
        best_params_final = dict(most_common)
    else:
        best_params_final = {}

    print(f"\nFinal Best Params (majority vote): {best_params_final}")

    # Find best out-of-sample value
    if window_results:
        best_res = max(window_results, key=lambda x: x['out_sample_value'])
        best_value_final = best_res['out_sample_value']
        best_backtest_id = best_res['test_backtest_id']
    else:
        best_value_final = 0
        best_backtest_id = None

    # Save summary
    save_optimization_run({'_id': opt_run_id, 'timestamp': str(pd.Timestamp.now()), 'strategy_id': strategy_id, 'optimize_metric': optimize_metric, 'best_params': best_params_final, 'best_value': best_value_final, 'best_backtest_id': best_backtest_id, 'experiment_id': experiment_id, 'phase': 'out-of-sample', 'stock_symbol': ticker, 'timeframe': '5minute'})

    return window_results, best_params_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid Walk-Forward Optimization for Trading Strategies')
    parser.add_argument('--data_file', type=str, default='../data/5minute/910_NSE_EICHERMOT.csv', help='Path to the data CSV file')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Overall start date for optimization')
    parser.add_argument('--end_date', type=str, default='2025-09-25', help='Overall end date for optimization')
    parser.add_argument('--initial_cash', type=int, default=15000, help='Initial cash')
    parser.add_argument('--optimize_metric', type=str, default='Net Profit', choices=['Net Profit', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Profit Factor', 'Composite Score'], help='Metric to optimize')
    parser.add_argument('--train_years', type=int, default=2, help='Training window in years')
    parser.add_argument('--test_months', type=int, default=6, help='Testing window in months')
    parser.add_argument('--random_samples', type=int, default=20, help='Number of random samples')
    parser.add_argument('--top_k', type=int, default=5, help='Top K from random search')
    parser.add_argument('--bayes_iter', type=int, default=30, help='Bayesian optimization iterations')
    parser.add_argument('--intraday', type=bool, default=False, help='Intraday mode (True for MIS, False for CNC)')

    args = parser.parse_args()

    fixed_args = {
        'data_file': args.data_file,
        'initial_cash': args.initial_cash,
        'strategy_name': 'ai_v3_streak',
        'optimize_metric': args.optimize_metric,
        'intraday': args.intraday
    }

    results, best_params = hybrid_optimize_strategy(STRATEGY_FILE, param_bounds, fixed_args, args.optimize_metric, args.train_years, args.test_months, args.start_date, args.end_date, args.random_samples, args.top_k, args.bayes_iter)

    print("\nWindow Results:")
    for res in results:
        print(res)
    print(f"\nOverall Best Params: {best_params}")