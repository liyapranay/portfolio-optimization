import numpy as np
from scipy.stats import norm

def calculate_sharpe(returns):
    if np.std(returns) == 0:
        return 0
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

def calculate_sortino(returns):
    downside = [r for r in returns if r < 0]
    if len(downside) == 0:
        return 0
    return np.mean(returns) / (np.std(downside) * np.sqrt(252))

def calculate_calmar(cagr, max_dd):
    return cagr / abs(max_dd) if max_dd != 0 else np.nan

def calculate_psr(returns, sharpe):
    n = len(returns)
    if n < 2 or np.std(returns, ddof=1) == 0:
        return 0.0
    psr = norm.cdf((sharpe - 0) * np.sqrt(n - 1))
    return psr

def calculate_dsr(returns, sharpe):
    n = len(returns)
    if n < 20 or np.std(returns, ddof=1) == 0:
        return calculate_psr(returns, sharpe)
    skew = ((returns - np.mean(returns))**3).mean() / (np.std(returns, ddof=1)**3)
    kurt = ((returns - np.mean(returns))**4).mean() / (np.std(returns, ddof=1)**4)
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    dsr = norm.cdf((sharpe - 0) / se)
    return float(np.clip(dsr, 0.0, 0.9999))

def calculate_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return dd.min()

def calculate_max_drawdown_duration(equity_curve):
    peaks = np.maximum.accumulate(equity_curve)
    underwater = equity_curve < peaks
    duration = 0
    max_duration = 0
    for u in underwater:
        if u:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return max_duration

def calculate_max_drawdown_duration_days(equity_curve, timeframe):
    """
    Convert Max Drawdown Duration bars → trading days.
    """
    max_duration = calculate_max_drawdown_duration(equity_curve)
    bar_seconds = {
        "minute": 60,
        "3minute": 180,
        "5minute": 300,
        "10minute": 600,
        "15minute": 900,
        "30minute": 1800,
        "60minute": 3600,
        "day": 86400
    }.get(timeframe, 1)
    return (max_duration * bar_seconds) / 86400