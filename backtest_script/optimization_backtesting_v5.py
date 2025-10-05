import sys
sys.path.append('..')

import optuna
from modules.core.strategy_runner import run_strategy
from datetime import datetime
import uuid
import random
import string
import logging
import os
import pandas as pd
import time
from dateutil.relativedelta import relativedelta
from collections import Counter
from backtest_script.ai_v3_streak_backtesting_v2 import MyStrategy
from modules.database_mongo_helper import *

def generate_experiment_name():
    animals = ["Tiger","Wolf","Falcon","Shark","Eagle","Panther"]
    rand = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{random.choice(animals)}-{rand}"

def compute_profit_drawdown_robust_composite(metrics):
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
    mdd_dur = metrics.get('Max Drawdown Duration (days)', 0)

    # Normalize
    profit_score = min(profit / 10000, 1.0)
    sharpe_score = min(sharpe / 5.0, 1.0)
    sortino_score = min(sortino / 5.0, 1.0)
    calmar_score = min(calmar / 5.0, 1.0)
    pf_score = min(pf / 5.0, 1.0)
    psr_score = min(psr, 1.0)
    dsr_score = min(dsr, 1.0)
    mdd_score = 1 - min(mdd / 0.5, 1.0)
    mdd_dur_score = 1 - min(mdd_dur / 365, 1.0)

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

def select_global_best_params(window_results, variable_params, method='majority'):
    if method == 'majority':
        # Majority vote per parameter
        param_votes = {param_name: [] for param_name in variable_params.keys()}
        for res in window_results:
            for p, v in res['best_params'].items():
                if p in param_votes:
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

def optuna_optimize(strategy_file, fixed_args, optimize_metric, window_id, n_trials, n_jobs, logger, variable_params):
    def objective(trial):
        trial_start_time = time.time()

        params = {}
        for param_name, param_config in variable_params.items():
            param_type, min_val, max_val = param_config
            if param_type == "int":
                params[param_name] = trial.suggest_int(param_name, min_val, max_val)
            elif param_type == "float":
                params[param_name] = trial.suggest_float(param_name, min_val, max_val)

        start_date = datetime.strptime(fixed_args['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(fixed_args['end_date'], '%Y-%m-%d')
        metrics, backtest_id, *_ = run_strategy(
            strategy_file,
            params,
            start_date=start_date,
            end_date=end_date,
            data_file=fixed_args['data_file'],
            initial_cash=fixed_args['initial_cash'],
            broker_name=fixed_args.get('broker_name', 'zerodha'),
            product_type=fixed_args.get('product_type', 'CNC'),
            strategy_class=fixed_args.get('strategy_class', MyStrategy)
        )

        exec_time = time.time() - trial_start_time

        if optimize_metric == 'ProfitDrawdownRobustComposite':
            value = compute_profit_drawdown_robust_composite(metrics)
        else:
            value = metrics.get(optimize_metric, -1e10)

        try:
            best_so_far = trial.study.best_value
        except ValueError:
            best_so_far = value
        logger.info("Optuna Trial %d completed in %.2f seconds, value: %.4f", trial.number, exec_time, value)
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
            'pruned': False
        })
        return value
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    best_params = study.best_params
    best_value = study.best_value
    # Get train metrics for best params
    start_date = datetime.strptime(fixed_args['start_date'], '%Y-%m-%d')
    end_date = datetime.strptime(fixed_args['end_date'], '%Y-%m-%d')
    train_metrics, train_backtest_id, *_ = run_strategy(
        strategy_file,
        best_params,
        start_date=start_date,
        end_date=end_date,
        data_file=fixed_args['data_file'],
        initial_cash=fixed_args['initial_cash'],
        broker_name=fixed_args.get('broker_name', 'zerodha'),
        product_type=fixed_args.get('product_type', 'CNC'),
        strategy_class=fixed_args.get('strategy_class', MyStrategy)
    )
    logger.info("Optuna Optimization completed")
    return best_params, best_value, train_metrics, train_backtest_id


def optuna_optimize_strategy(strategy_file, fixed_args, optimize_metric, train_years, test_months, start_date, end_date, n_trials, n_jobs, experiment_name, logger, variable_params):
    experiment_start = time.time()
    experiment_id = uuid.uuid4()
    experiment_name_final = experiment_name or generate_experiment_name()

    ticker = fixed_args['data_file'].split("_")[-1].split(".")[0]

    # Setup logging (unified) - but logger is already set up in main
    experiment = {
        "_id": experiment_id,
        "experiment_name": experiment_name_final,
        "timestamp": str(pd.Timestamp.now())
    }
    save_experiment(experiment)
    logger.info("Created experiment '%s' (_id=%s)", experiment_name_final, experiment_id)

    # Create opt_run_doc
    strategy_id = fixed_args.get('strategy_name', 'ai_v3_streak')
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
        best_params, best_value, train_metrics, train_backtest_id = optuna_optimize(strategy_file, train_fixed_args, optimize_metric, window_id, n_trials, n_jobs, logger, variable_params)

        # Save explicit train backtest
        train_doc = {
            "_id": train_backtest_id,
            "experiment_id": experiment_id,
            "strategy_id": strategy_id,
            "stock_symbol": stock_symbol,
            "timeframe": timeframe,
            "params": best_params,
            "metrics": train_metrics,
            "scope": "train",
            "window_number": iteration,
            "timestamp": datetime.utcnow()
        }
        save_backtest(train_doc, train_metrics)

        # Testing phase
        test_fixed_args = fixed_args.copy()
        test_fixed_args['start_date'] = train_end.strftime('%Y-%m-%d')
        test_fixed_args['end_date'] = test_end.strftime('%Y-%m-%d')
        test_fixed_args['experiment_id'] = experiment_id
        test_fixed_args['window_id'] = window_id

        test_start_date = datetime.strptime(test_fixed_args['start_date'], '%Y-%m-%d')
        test_end_date = datetime.strptime(test_fixed_args['end_date'], '%Y-%m-%d')
        test_metrics, test_backtest_id, *_ = run_strategy(
            strategy_file,
            best_params,
            start_date=test_start_date,
            end_date=test_end_date,
            data_file=test_fixed_args['data_file'],
            initial_cash=test_fixed_args['initial_cash'],
            broker_name=test_fixed_args.get('broker_name', 'zerodha'),
            product_type=test_fixed_args.get('product_type', 'CNC'),
            strategy_class=test_fixed_args.get('strategy_class', MyStrategy)
        )

        if optimize_metric == 'ProfitDrawdownRobustComposite':
            test_value = compute_profit_drawdown_robust_composite(test_metrics)
        else:
            test_value = test_metrics.get(optimize_metric, -1e10)

        # Save window backtest with train and test metrics
        backtest_doc = {
            "_id": test_backtest_id,
            "experiment_id": experiment_id,
            "strategy_id": strategy_id,
            "stock_symbol": stock_symbol,
            "timeframe": timeframe,
            "params": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "scope": "window",
            "window_number": iteration,
            "window_id": window_id,
            "optimization_run_id": opt_run_id,
            "timestamp": datetime.utcnow()
        }
        composite = {'sharpe': test_metrics.get('Sharpe Ratio', 0), 'sortino': test_metrics.get('Sortino Ratio', 0), 'calmar': test_metrics.get('Calmar Ratio', 0), 'profit_factor': test_metrics.get('Profit Factor', 0), 'composite_score': test_value}
        backtest_doc['train_metrics'] = train_metrics
        backtest_doc['test_metrics'] = test_metrics
        save_backtest(backtest_doc, None, composite)

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
        current_train_start = train_end - relativedelta(months=test_months)

    # Global best parameter selection
    global_selection_method = 'majority'
    global_params = select_global_best_params(window_results, variable_params, global_selection_method)
    logger.info("Global selection method = %s", global_selection_method)
    logger.info("Global best params: %s", global_params)

    # Global backtest run
    logger.info("Global backtest run started with params: %s", global_params)
    global_fixed_args = fixed_args.copy()
    global_fixed_args['start_date'] = start_date
    global_fixed_args['end_date'] = end_date
    global_fixed_args['experiment_id'] = experiment_id
    global_fixed_args['window_id'] = None

    global_start_date = pd.to_datetime(global_fixed_args['start_date'])
    global_end_date = pd.to_datetime(global_fixed_args['end_date'])
    global_metrics, global_backtest_id, *_ = run_strategy(
        strategy_file,
        global_params,
        start_date=global_start_date,
        end_date=global_end_date,
        data_file=global_fixed_args['data_file'],
        initial_cash=global_fixed_args['initial_cash'],
        broker_name=global_fixed_args.get('broker_name', 'zerodha'),
        product_type=global_fixed_args.get('product_type', 'CNC'),
        strategy_class=global_fixed_args.get('strategy_class', MyStrategy)
    )

    global_composite = compute_profit_drawdown_robust_composite(global_metrics) if optimize_metric == 'ProfitDrawdownRobustComposite' else global_metrics.get(optimize_metric, 0)
    logger.info("Global backtest completed, metrics: %s", global_metrics)

    # Save global backtest
    if global_backtest_id:
        global_backtest_doc = {
            '_id': global_backtest_id,
            'timestamp': str(pd.Timestamp.now()),
            'strategy_id': strategy_id,
            'params': global_params,
            'stock_symbol': stock_symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
            'initial_cash': fixed_args['initial_cash'],
            'final_value': global_metrics.get('Final Portfolio Value', fixed_args['initial_cash']),
            'window_id': None,
            'experiment_id': experiment_id,
            'scope': 'global',
            'optimization_run_id': opt_run_id,
            'exec_time': 0.0,
            'test_metrics': global_metrics
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
    import argparse

    parser = argparse.ArgumentParser(description='Optuna Walk-Forward Optimization for Trading Strategies v5.0')
    parser.add_argument('--strategy_file', type=str, default='backtest_script.ai_v3_streak_backtesting_v2', help='Strategy file path')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--start_date', type=str, required=True, help='Overall start date for optimization')
    parser.add_argument('--end_date', type=str, required=True, help='Overall end date for optimization')
    parser.add_argument('--initial_cash', type=int, default=15000, help='Initial cash')
    parser.add_argument('--broker_name', type=str, default='zerodha', help='Broker name')
    parser.add_argument('--product_type', type=str, default='CNC', help='Product type')
    parser.add_argument('--optimize_metric', type=str, default='ProfitDrawdownRobustComposite', choices=['Net Profit', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Profit Factor', 'Composite Score', 'ProfitDrawdownRobustComposite'], help='Metric to optimize')
    parser.add_argument('--train_years', type=int, default=2, help='Training window in years')
    parser.add_argument('--test_months', type=int, default=6, help='Testing window in months')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')

    args = parser.parse_args()

    # Define variable parameters to optimize (type, min, max)
    variable_params = {
        'ema_fast': ("int", 5, 30),
        'ema_slow': ("int", 20, 100),
        'adx_threshold': ("int", 10, 30),
        'momentum_threshold': ("float", -5, 5),
        'natr_multiplier': ("float", 0.5, 2.0),
        'profit_target': ("int", 100, 400)
    }

    fixed_args = {
        'data_file': args.data_file,
        'initial_cash': args.initial_cash,
        'strategy_name': 'ai_v3_streak',
        'optimize_metric': args.optimize_metric,
        'strategy_class': MyStrategy,
        'broker_name': args.broker_name,
        'product_type': args.product_type
    }

    # Generate or use provided experiment name
    experiment_name = args.experiment_name or generate_experiment_name()
    print(f"Experiment Name: {experiment_name}")

    # Setup logging (unified)
    timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"../logs/{experiment_name}_{timestamp_str}.log"
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    results, global_params = optuna_optimize_strategy(args.strategy_file, fixed_args, args.optimize_metric, args.train_years, args.test_months, args.start_date, args.end_date, args.n_trials, args.n_jobs, experiment_name, logger, variable_params)

    print("\nWindow Results:")
    for res in results:
        print(res)
    print(f"\nGlobal Best Params: {global_params}")