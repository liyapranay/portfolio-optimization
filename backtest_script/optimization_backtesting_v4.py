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
import optuna
import numpy as np
import uuid
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import Counter

# Strategy file being optimized
STRATEGY_FILE = 'ai_v3_streak_backtesting.py'

def generate_experiment_name():
    animals = ["Tiger","Wolf","Falcon","Shark","Eagle","Panther"]
    rand = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{random.choice(animals)}-{rand}"

def run_backtest(params, script_path, fixed_args, logger):
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

def compute_profit_drawdown_robust_composite(metrics,
                                             profit_scale=10000,
                                             mdd_scale=0.5,
                                             dur_scale=365):
    """
    Composite Score: ProfitDrawdownRobustComposite
    Balances profit, risk-adjusted returns, and drawdown control.
    """
    profit = metrics.get('Net Profit', 0)
    sharpe = metrics.get('Sharpe Ratio', 0)
    sortino = metrics.get('Sortino Ratio', 0)
    calmar = metrics.get('Calmar Ratio', 0)
    pf = metrics.get('Profit Factor', 0)
    psr = metrics.get('PSR', 0)
    dsr = metrics.get('DSR', 0)
    mdd = metrics.get('Max Drawdown', 0)
    mdd_dur = metrics.get('Max Drawdown Duration', 0)

    # Normalize
    profit_score = min(profit / profit_scale, 1.0)
    sharpe_score = np.tanh(sharpe)
    sortino_score = np.tanh(sortino)
    calmar_score = np.tanh(calmar)
    pf_score = min(pf / 5.0, 1.0)
    psr_score = np.clip(psr, 0, 1)
    dsr_score = np.clip(dsr, 0, 1)
    mdd_score = 1 - min(mdd / mdd_scale, 1.0)
    mdd_dur_score = 1 - min(mdd_dur / dur_scale, 1.0)

    # Weighted blend
    composite = (
        0.25 * dsr_score +
        0.15 * psr_score +
        0.15 * profit_score +
        0.10 * sharpe_score +
        0.10 * sortino_score +
        0.10 * calmar_score +
        0.05 * pf_score +
        0.05 * mdd_score +
        0.05 * mdd_dur_score
    )
    return composite
def optuna_optimize(script_path, fixed_args, optimize_metric, window_id, n_trials, n_jobs, logger):
    def objective(trial):
        params = {
            'ema_fast': trial.suggest_int("ema_fast", 5, 30),
            'ema_slow': trial.suggest_int("ema_slow", 20, 100),
            'adx_threshold': trial.suggest_int("adx_threshold", 10, 30),
            'momentum_threshold': trial.suggest_float("momentum_threshold", -5, 5),
            'natr_multiplier': trial.suggest_float("natr_multiplier", 0.5, 2.0),
            'profit_target': trial.suggest_int("profit_target", 100, 400)
        }
        metrics, backtest_id, exec_time = run_backtest(params, script_path, fixed_args, logger)
        if 'error' in metrics:
            value = -1e10
        elif optimize_metric == 'Composite Score':
            value = compute_composite_score(metrics)
        elif optimize_metric == 'ProfitDrawdownRobustComposite':
            value = compute_profit_drawdown_robust_composite(metrics)
        else:
            value = metrics.get(optimize_metric, -1e10)
        try:
            best_so_far = trial.study.best_value
        except ValueError:
            best_so_far = value  # For the first trial or if no completed trials
        logger.info("Optuna Trial %d completed in %.2f seconds, value: %.4f", trial.number, exec_time, value)
        # Save to DB
        save_iteration({
            '_id': str(uuid.uuid4()),
            'timestamp': str(pd.Timestamp.now()),
            'trial_id': trial.number,
            'target_metric': optimize_metric,
            'target': value,
            'best_so_far': best_so_far,
            'parameters': params,
            'phase': 'optuna',
            'backtest_id': backtest_id,
            'window_id': window_id,
            'exec_time': exec_time,
            'pruned': False  # Assuming completed trials are not pruned
        })
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    best_params = study.best_params
    best_value = study.best_value
    logger.info("Optuna Optimization completed")
    return best_params, best_value

def select_global_best_params(window_results, method='majority'):
    if method == 'majority':
        # Majority vote per parameter
        param_votes = {'ema_fast': [], 'ema_slow': [], 'adx_threshold': [], 'momentum_threshold': [], 'natr_multiplier': [], 'profit_target': []}
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

def optuna_optimize_strategy(script_path, fixed_args, optimize_metric, train_years, test_months, start_date, end_date, n_trials, n_jobs, experiment_name, logger):
    experiment_start = time.time()
    experiment_id = uuid.uuid4()
    experiment_name_final = experiment_name or generate_experiment_name()

    ticker = fixed_args['data_file'].split("_")[-1].split(".")[0]

    # Setup logging (unified)
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"../logs/{experiment_name_final}_{timestamp_str}.log"
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    experiment = {
        "_id": experiment_id,
        "experiment_name": experiment_name_final,
        "timestamp": str(pd.Timestamp.now())
    }
    save_experiment(experiment)
    logger.info("Created experiment '%s' (_id=%s)", experiment_name_final, experiment_id)

    # Create opt_run_doc
    strategy_id = fixed_args.get('strategy_name', 'unknown')
    stock_symbol = ticker
    timeframe = '5minute'
    opt_run_doc = {
        "_id": uuid.uuid4(),
        "experiment_id": experiment["_id"],
        "experiment_name": experiment["experiment_name"],
        "strategy_id": strategy_id,
        "stock_symbol": stock_symbol,
        "timeframe": timeframe,
        "optimize_metric": optimize_metric,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "best_params": None,
        "best_value": None,
        "best_backtest_id": None,
        "timestamp": datetime.utcnow()
    }
    save_optimization_run(opt_run_doc)
    opt_run_id = opt_run_doc["_id"]
    logger.info("Created optimization run for experiment %s (%s)", experiment["experiment_name"], experiment["_id"])
    print(f"Experiment Name: {experiment_name_final}")

    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    window_results = []

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

        print(f"Running Optuna Optimization for window {iteration}: Train {current_train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, Test {train_end.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

        window_id = str(uuid.uuid4())
        save_window({'_id': window_id, 'optimization_run_id': opt_run_id, 'window_number': iteration, 'train_start': current_train_start.strftime('%Y-%m-%d'), 'train_end': train_end.strftime('%Y-%m-%d'), 'test_start': train_end.strftime('%Y-%m-%d'), 'test_end': test_end.strftime('%Y-%m-%d')})

        # Training phase
        train_fixed_args = fixed_args.copy()
        train_fixed_args['start_date'] = current_train_start.strftime('%Y-%m-%d')
        train_fixed_args['end_date'] = train_end.strftime('%Y-%m-%d')
        train_fixed_args['experiment_id'] = experiment_id
        train_fixed_args['window_id'] = window_id

        # Optuna Optimization
        best_params, best_value = optuna_optimize(script_path, train_fixed_args, optimize_metric, window_id, n_trials, n_jobs, logger)

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
        elif optimize_metric == 'ProfitDrawdownRobustComposite':
            test_value = compute_profit_drawdown_robust_composite(test_metrics)
        else:
            test_value = test_metrics.get(optimize_metric, -1e10)


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

        # Move window
        current_train_start = train_end - relativedelta(months=test_months)  # Overlap

    # Global best parameter selection
    global_selection_method = 'majority'
    global_params = select_global_best_params(window_results, global_selection_method)
    logger.info("Global selection method = %s", global_selection_method)
    logger.info("Global best params: %s", global_params)

    # Global backtest run
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
    save_experiment({'_id': experiment_id, 'experiment_name': experiment_name_final, 'experiment_total_time': total_time})
    db.optimization_runs.update_one(
        {"_id": opt_run_doc["_id"]},
        {"$set": {"best_params": global_params, "best_value": global_composite, "best_backtest_id": global_backtest_id}}
    )
    logger.info("Updated optimization run %s with best params and value", opt_run_doc['_id'])

    return window_results, global_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optuna Walk-Forward Optimization for Trading Strategies v4.0')
    parser.add_argument('--data_file', type=str, default='../data/5minute/910_NSE_EICHERMOT.csv', help='Path to the data CSV file')
    parser.add_argument('--stock_symbol', type=str, default='EICHERMOT', help='Stock symbol')
    parser.add_argument('--stock_market', type=str, default='NSE', help='Stock market')
    parser.add_argument('--timeframe', type=str, default='5minute', help='Timeframe')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Overall start date for optimization')
    parser.add_argument('--end_date', type=str, default='2025-09-25', help='Overall end date for optimization')
    parser.add_argument('--initial_cash', type=int, default=15000, help='Initial cash')
    parser.add_argument('--optimize_metric', type=str, default='Composite Score', choices=['Net Profit', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Profit Factor', 'Composite Score', 'ProfitDrawdownRobustComposite'], help='Metric to optimize')
    parser.add_argument('--train_years', type=int, default=2, help='Training window in years')
    parser.add_argument('--test_months', type=int, default=6, help='Testing window in months')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
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
    results, global_params = optuna_optimize_strategy(STRATEGY_FILE, fixed_args, args.optimize_metric, args.train_years, args.test_months, args.start_date, args.end_date, args.n_trials, args.n_jobs, args.experiment_name, logger)

    print("\nWindow Results:")
    for res in results:
        print(res)
    print(f"\nGlobal Best Params: {global_params}")