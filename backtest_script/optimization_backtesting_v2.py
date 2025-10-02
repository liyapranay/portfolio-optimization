import subprocess
import json
import argparse
import sys
sys.path.append('..')
# import database_v2 as db
from modules import database_v2 as db
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

def hybrid_optimize_strategy(script_path, param_bounds, fixed_args, optimize_metric, train_years, test_months, start_date, end_date):
    opt_run_id = str(uuid.uuid4())
    experiment_id = opt_run_id
    print(f"Hybrid Optimization UUID: {opt_run_id}")

    db.save_experiment(experiment_id, 'Hybrid optimization experiment', str(pd.Timestamp.now()))
    db.save_optimization_run(opt_run_id, str(pd.Timestamp.now()), 'hybrid_' + fixed_args.get('strategy_name', 'unknown'), optimize_metric, {}, 0, None, experiment_id)

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
        db.save_window(window_id, opt_run_id, iteration, current_train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'), test_end.strftime('%Y-%m-%d'))

        # Training phase
        train_fixed_args = fixed_args.copy()
        train_fixed_args['start_date'] = current_train_start.strftime('%Y-%m-%d')
        train_fixed_args['end_date'] = train_end.strftime('%Y-%m-%d')
        train_fixed_args['experiment_id'] = experiment_id
        train_fixed_args['window_id'] = window_id

        def objective(**params):
            metrics, train_backtest_id = run_backtest(params, script_path, train_fixed_args)
            if 'error' in metrics:
                value = -1e10
            elif optimize_metric == 'Composite Score':
                value = compute_composite_score(metrics)
            else:
                value = metrics.get(optimize_metric, -1e10)
            # Save metrics and composite
            if 'error' not in metrics and train_backtest_id:
                db.save_performance_metrics(train_backtest_id, metrics, 'in_sample')
                sharpe = metrics.get('Sharpe Ratio', 0)
                sortino = metrics.get('Sortino Ratio', 0)
                calmar = metrics.get('Calmar Ratio', 0)
                pf = metrics.get('Profit Factor', 0)
                composite = 0.3 * sharpe + 0.3 * sortino + 0.3 * calmar + 0.1 * pf
                db.save_composite_metrics(train_backtest_id, sharpe, sortino, calmar, pf, composite, 'in_sample')
            # Log iteration
            db.save_iteration(str(uuid.uuid4()), str(pd.Timestamp.now()), iteration * 1000 + len(param_sets), {}, params, optimize_metric, value, train_backtest_id, window_id)
            return value

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_bounds,
            random_state=42,
        )
        optimizer.maximize(init_points=3, n_iter=2)  # Reduced for speed

        best_params = optimizer.max['params']
        best_value = optimizer.max['target']

        print(f"Best params: {best_params}")
        print(f"Best {optimize_metric}: {best_value}")

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

        # Save out-of-sample metrics
        if 'error' not in test_metrics and test_backtest_id:
            db.save_performance_metrics(test_backtest_id, test_metrics, 'out_of_sample')
            sharpe = test_metrics.get('Sharpe Ratio', 0)
            sortino = test_metrics.get('Sortino Ratio', 0)
            calmar = test_metrics.get('Calmar Ratio', 0)
            pf = test_metrics.get('Profit Factor', 0)
            composite = 0.3 * sharpe + 0.3 * sortino + 0.3 * calmar + 0.1 * pf
            db.save_composite_metrics(test_backtest_id, sharpe, sortino, calmar, pf, composite, 'out_of_sample')

        print(f"Out-of-sample {optimize_metric}: {test_value}")

        window_results.append({
            'iteration': iteration,
            'train_start': current_train_start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'best_params': best_params,
            'in_sample_value': best_value,
            'out_sample_value': test_value
        })

        param_sets.append(tuple(sorted(best_params.items())))

        # Move window
        current_train_start = train_end - relativedelta(months=test_months)  # Overlap

    # Determine most robust params
    if param_sets:
        most_common = Counter(param_sets).most_common(1)[0][0]
        best_params_final = dict(most_common)
    else:
        best_params_final = {}

    print(f"\nFinal Best Params (majority vote): {best_params_final}")

    # Save summary
    db.save_optimization_run(opt_run_id, str(pd.Timestamp.now()), 'hybrid_' + fixed_args.get('strategy_name', 'unknown'), optimize_metric, best_params_final, 0, None, experiment_id)

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
    parser.add_argument('--intraday', type=bool, default=False, help='Intraday mode (True for MIS, False for CNC)')

    args = parser.parse_args()

    fixed_args = {
        'data_file': args.data_file,
        'initial_cash': args.initial_cash,
        'strategy_name': 'ai_v3_streak',
        'optimize_metric': args.optimize_metric,
        'intraday':args.intraday
    }

    results, best_params = hybrid_optimize_strategy(STRATEGY_FILE, param_bounds, fixed_args, args.optimize_metric, args.train_years, args.test_months, args.start_date, args.end_date)

    print("\nWindow Results:")
    for res in results:
        print(res)
    print(f"\nOverall Best Params: {best_params}")