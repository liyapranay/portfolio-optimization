import subprocess
import json
import argparse
import sys
import random
import string
import logging
import os
import time
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

def run_backtest(params, script_path, fixed_args, logger):
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
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
    exec_time = time.time() - start_time
    if result.returncode != 0:
        logger.error(f"Error running backtest: {result.stderr}")
        return {'error': True}, None, exec_time  # Return dict with error

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
    return metrics, backtest_id, exec_time

def compute_composite_score(metrics):
    sharpe = metrics.get('Sharpe Ratio', 0)
    sortino = metrics.get('Sortino Ratio', 0)
    calmar = metrics.get('Calmar Ratio', 0)
    pf = metrics.get('Profit Factor', 0)
    return 0.3 * sharpe + 0.3 * sortino + 0.3 * calmar + 0.1 * pf

def random_search(script_path, param_bounds, fixed_args, optimize_metric, window_id, N, top_k, logger):
    samples = []
    best_so_far = -1e10
    logger.info("Random Search started")
    for i in range(N):
        params = {}
        for p, (low, high) in param_bounds.items():
            if p in int_params:
                params[p] = np.random.randint(low, high + 1)
            else:
                params[p] = np.random.uniform(low, high)
        metrics, backtest_id, exec_time = run_backtest(params, script_path, fixed_args, logger)
        if 'error' in metrics:
            value = -1e10
        elif optimize_metric == 'Composite Score':
            value = compute_composite_score(metrics)
        else:
            value = metrics.get(optimize_metric, -1e10)
        if value > best_so_far:
            best_so_far = value
        samples.append((params, value))
        logger.info("Random Iteration %d completed in %.2f seconds", i+1, exec_time)
        # Save to DB
        save_iteration({'_id': str(uuid.uuid4()), 'timestamp': str(pd.Timestamp.now()), 'iteration': i + 1, 'target_metric': optimize_metric, 'target': value, 'best_so_far': best_so_far, 'parameters': params, 'phase': 'random', 'backtest_id': backtest_id, 'window_id': window_id, 'exec_time': exec_time})
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
    logger.info("Bayesian Optimization started")
    iter_count = 0
    best_so_far = -1e10
    def objective(**params):
        nonlocal iter_count, best_so_far
        iter_count += 1
        metrics, backtest_id, exec_time = run_backtest(params, script_path, fixed_args, logger)
        if 'error' in metrics:
            value = -1e10
        elif optimize_metric == 'Composite Score':
            value = compute_composite_score(metrics)
        else:
            value = metrics.get(optimize_metric, -1e10)
        if value > best_so_far:
            best_so_far = value
        logger.info("Bayesian Iteration %d completed in %.2f seconds", iter_count, exec_time)
        # Save to DB
        save_iteration({'_id': str(uuid.uuid4()), 'timestamp': str(pd.Timestamp.now()), 'iteration': iter_count, 'target_metric': optimize_metric, 'target': value, 'best_so_far': best_so_far, 'parameters': params, 'phase': 'bayesian', 'backtest_id': backtest_id, 'window_id': window_id, 'exec_time': exec_time})
        return value

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=narrowed_bounds,
        random_state=42,
    )
    optimizer.maximize(init_points=0, n_iter=iterations)
    best_params = optimizer.max['params']
    best_value = optimizer.max['target']
    logger.info("Bayesian Optimization completed")
    return best_params, best_value

def select_global_best_params(window_results, method='majority'):
    if method == 'majority':
        # Majority vote per parameter
        param_votes = {p: [] for p in param_bounds}
        for res in window_results:
            for p, v in res['best_params'].items():
                param_votes[p].append(v)
        global_params = {}
        for p, votes in param_votes.items():
            if votes:
                most_common = Counter(votes).most_common(1)[0][0]
                global_params[p] = most_common
        return global_params
    elif method == 'best_avg':
        # Best average composite score
        param_sets = {}
        for res in window_results:
            key = tuple(sorted(res['best_params'].items()))
            if key not in param_sets:
                param_sets[key] = []
            param_sets[key].append(res['out_sample_value'])
        best_key = max(param_sets, key=lambda k: np.mean(param_sets[k]))
        return dict(best_key)
    else:
        raise ValueError("Invalid method")

def hybrid_optimize_strategy(script_path, param_bounds, fixed_args, optimize_metric, train_years, test_months, start_date, end_date, random_samples, top_k, bayes_iter, experiment_name, logger):
    experiment_start = time.time()
    opt_run_id = str(uuid.uuid4())
    experiment_id = opt_run_id
    experiment_description = experiment_name or generate_random_description()

    ticker = fixed_args['data_file'].split("_")[-1].split(".")[0]

    # Setup logging (unified)
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"../logs/{experiment_description}_{timestamp_str}.log"
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Experiment %s started", experiment_description)

    save_experiment({'_id': experiment_id, 'description': experiment_description, 'timestamp': str(pd.Timestamp.now())})
    save_optimization_run({'_id': opt_run_id, 'timestamp': str(pd.Timestamp.now()), 'strategy_id': fixed_args.get('strategy_name', 'unknown'), 'optimize_metric': optimize_metric, 'best_params': {}, 'best_value': 0, 'best_backtest_id': None, 'experiment_id': experiment_id, 'phase': 'out-of-sample', 'stock_symbol': ticker, 'timeframe': '5minute'})
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

        print(f"Running Random Search + Bayesian Optimization for window {iteration}...")

        window_id = str(uuid.uuid4())
        save_window({'_id': window_id, 'optimization_run_id': opt_run_id, 'window_number': iteration, 'train_start': current_train_start.strftime('%Y-%m-%d'), 'train_end': train_end.strftime('%Y-%m-%d'), 'test_start': train_end.strftime('%Y-%m-%d'), 'test_end': test_end.strftime('%Y-%m-%d')})

        # Training phase
        train_fixed_args = fixed_args.copy()
        train_fixed_args['start_date'] = current_train_start.strftime('%Y-%m-%d')
        train_fixed_args['end_date'] = train_end.strftime('%Y-%m-%d')
        train_fixed_args['experiment_id'] = experiment_id
        train_fixed_args['window_id'] = window_id

        # Random Search
        random_start = time.time()
        top_k_params = random_search(script_path, param_bounds, train_fixed_args, optimize_metric, window_id, random_samples, top_k, logger)
        random_time = time.time() - random_start
        logger.info("Random Search completed in %.2f seconds", random_time)

        # Bayesian Optimization
        bayes_start = time.time()
        best_params, best_value = bayesian_optimize(script_path, top_k_params, train_fixed_args, optimize_metric, window_id, bayes_iter, logger)
        bayes_time = time.time() - bayes_start
        logger.info("Bayesian Optimization completed in %.2f seconds", bayes_time)

        # Testing phase
        test_fixed_args = fixed_args.copy()
        test_fixed_args['start_date'] = train_end.strftime('%Y-%m-%d')
        test_fixed_args['end_date'] = test_end.strftime('%Y-%m-%d')
        test_fixed_args['experiment_id'] = experiment_id
        test_fixed_args['window_id'] = window_id

        test_metrics, test_backtest_id, _ = run_backtest(best_params, script_path, test_fixed_args, logger)
        if 'error' in test_metrics:
            test_value = -1e10
        elif optimize_metric == 'Composite Score':
            test_value = compute_composite_score(test_metrics)
        else:
            test_value = test_metrics.get(optimize_metric, -1e10)

        print(f"Completed window {iteration}, best params saved.")

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

    # Global best parameter selection
    global_selection_method = 'majority'
    global_params = select_global_best_params(window_results, global_selection_method)
    logger.info("Global selection method = %s", global_selection_method)
    logger.info("Global best params: %s", global_params)

    # Global backtest run
    print("Global backtest run started...")
    logger.info("Global backtest run started with params: %s", global_params)
    global_fixed_args = fixed_args.copy()
    global_fixed_args['start_date'] = start_date
    global_fixed_args['end_date'] = end_date
    global_fixed_args['experiment_id'] = experiment_id
    global_fixed_args['window_id'] = None  # No window for global

    global_metrics, global_backtest_id, global_exec_time = run_backtest(global_params, script_path, global_fixed_args, logger)
    if 'error' not in global_metrics:
        global_composite = compute_composite_score(global_metrics)
        logger.info("Global backtest completed in %.2f seconds, metrics: %s", global_exec_time, global_metrics)
        print("Global backtest completed, results saved.")
    else:
        global_composite = 0
        logger.error("Global backtest failed")

    # Save global backtest
    if global_backtest_id:
        global_backtest_doc = {
            '_id': global_backtest_id,
            'timestamp': str(pd.Timestamp.now()),
            'strategy_id': fixed_args.get('strategy_name', 'unknown'),
            'params': global_params,
            'stock_symbol': ticker,
            'timeframe': '5minute',
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': fixed_args['initial_cash'],
            'final_value': global_metrics.get('Final Portfolio Value', fixed_args['initial_cash']),
            'window_id': None,
            'experiment_id': experiment_id,
            'scope': 'global',
            'optimization_run_id': opt_run_id,
            'exec_time': global_exec_time
        }
        save_backtest(global_backtest_doc, global_metrics, {'sharpe': global_metrics.get('Sharpe Ratio', 0), 'sortino': global_metrics.get('Sortino Ratio', 0), 'calmar': global_metrics.get('Calmar Ratio', 0), 'profit_factor': global_metrics.get('Profit Factor', 0), 'composite_score': global_composite})

    # Update optimization run with global best
    total_time = time.time() - experiment_start
    minutes = int(total_time // 60)
    seconds = total_time % 60
    logger.info("Experiment completed in %d minutes %.2f seconds", minutes, seconds)
    save_experiment({'_id': experiment_id, 'experiment_total_time': total_time})
    save_optimization_run({'_id': opt_run_id, 'random_total_time': random_time, 'bayesian_total_time': bayes_time})

    return window_results, global_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid Walk-Forward Optimization for Trading Strategies v3.4')
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
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for logging and MongoDB')

    args = parser.parse_args()

    fixed_args = {
        'data_file': args.data_file,
        'initial_cash': args.initial_cash,
        'strategy_name': 'ai_v3_streak',
        'optimize_metric': args.optimize_metric,
        'intraday': False
    }

    logger = logging.getLogger()  # Placeholder, will be set in function
    results, global_params = hybrid_optimize_strategy(STRATEGY_FILE, param_bounds, fixed_args, args.optimize_metric, args.train_years, args.test_months, args.start_date, args.end_date, args.random_samples, args.top_k, args.bayes_iter, args.experiment_name, logger)

    print("\nWindow Results:")
    for res in results:
        print(res)
    print(f"\nGlobal Best Params: {global_params}")