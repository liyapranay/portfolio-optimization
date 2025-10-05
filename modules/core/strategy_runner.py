import importlib

def run_strategy(strategy_file, params, **kwargs):
    """
    Dynamically import and execute strategy backtest.
    strategy_file: relative path or module name (e.g., 'backtest_script.ai_v3_streak_backtesting_v2')
    params: dict of optimization parameters
    kwargs: dynamic args passed from optimizer (start_date, end_date, data_file, etc.)
    """
    module_name = strategy_file.replace('/', '.').replace('.py', '')
    strategy_module = importlib.import_module(module_name)
    run_func = getattr(strategy_module, 'run_backtest')
    return run_func(params, **kwargs)