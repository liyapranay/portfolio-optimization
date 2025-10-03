import backtrader as bt
import pandas as pd
import datetime
import csv
import ta
import numpy as np
from scipy.stats import norm
from tabulate import tabulate
import os
import uuid
import sys
import logging
import time
sys.path.append('..')
from modules.database_mongo_helper import *
import modules.plotting as plotting
import argparse

def str_to_bool(value):
    if value.lower() in ('true', '1', 'yes', 'on'):
        return True
    elif value.lower() in ('false', '0', 'no', 'off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def calculate_psr(returns, sharpe, benchmark=0.0):
    """
    Probabilistic Sharpe Ratio (PSR).
    Probability that true Sharpe > benchmark.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    sr = mean / std
    psr = norm.cdf((sr - benchmark) * np.sqrt(n - 1))
    return float(psr)

def calculate_dsr(returns, sharpe, benchmark=0.0):
    """
    Deflated Sharpe Ratio (Bailey et al. 2014).
    Adjusts PSR for skewness and kurtosis.
    """
    n = len(returns)
    if n < 20:
        return calculate_psr(returns, sharpe, benchmark)
    if np.std(returns, ddof=1) == 0:
        return 0.0

    skew = ((returns - np.mean(returns))**3).mean() / (np.std(returns, ddof=1)**3)
    kurt = ((returns - np.mean(returns))**4).mean() / (np.std(returns, ddof=1)**4)

    # standard error with skew & kurtosis adjustment
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n - 1))
    if se == 0:
        return 0.0

    dsr = norm.cdf((sharpe - benchmark) / se)
    dsr = float(np.clip(dsr, 0.0, 0.9999))
    return dsr

def calculate_max_drawdown_duration(equity_curve):
    """
    Max Drawdown Duration = longest period equity stays below its peak.
    Returns duration in bars.
    """
    peaks = np.maximum.accumulate(equity_curve)
    duration = 0
    max_duration = 0
    for eq, pk in zip(equity_curve, peaks):
        if eq < pk:
            duration += 1
            max_duration = max(max_duration, duration)
        else:
            duration = 0
    return int(max_duration)
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

def finalize_balance_curve(balance_curve, broker):
    final_equity = broker.getvalue()
    if len(balance_curve) == 0 or abs(balance_curve[-1] - final_equity) > 1e-6:
        balance_curve.append(final_equity)
    return balance_curve
    return (max_duration * bar_seconds) / 86400

class NSECommInfo(bt.CommInfoBase):
    params = (
        ('stocklike', True),
    )

    def getcommission(self, size, price, pseudoexec=False):
        value = abs(size) * price
        if pseudoexec:
            return 0.0

        # Brokerage: min(0.03% of trade value, ₹20)
        brokerage = min(0.0003 * value, 20)

        # STT: 0.025% on sell side
        stt = 0.00025 * value if size < 0 else 0

        # Transaction Charges: 0.003%
        trans_charges = 0.00003 * value

        # SEBI: 0.0001%
        sebi = 0.000001 * value

        # GST: 18% on (Brokerage + SEBI + Transaction charges)
        gst = 0.18 * (brokerage + sebi + trans_charges)

        # Stamp Duty: 0 (not applied in examples)
        stamp_duty = 0

        total_comm = brokerage + stt + trans_charges + sebi + gst + stamp_duty
        return total_comm

class MyPandasData(bt.feeds.PandasData):
    lines = ('vwap', 'kst', 'adx', 'ema_fast', 'ema_slow', 'rsi', 'volume_sma', 'momentum', 'mfi_ma', 'natr')
    params = (
        ('vwap', 'vwap'),
        ('kst', 'kst'),
        ('adx', 'adx'),
        ('ema_fast', 'ema_fast'),
        ('ema_slow', 'ema_slow'),
        ('rsi', 'rsi'),
        ('volume_sma', 'volume_sma'),
        ('momentum', 'momentum'),
        ('mfi_ma', 'mfi_ma'),
        ('natr', 'natr'),
    )

class MyStrategy(bt.Strategy):
    # Strategy parameters:
    # only_long=1: Only long trades
    # only_short=1: Only short trades
    # both=1: Both long and short trades
    # intraday=1: Intraday trading with time restrictions and NSE commissions
    # Usage: cerebro.addstrategy(MyStrategy, only_long=1) for long only
    params = (
        ('only_long', 1),
        ('only_short', 0),
        ('both', 0),
        ('intraday', True),
        ('stoploss_pct', 0.05),  # 5% stop loss
        ('takeprofit_pct', 0.10),  # 10% take profit
        ('stock_symbol', ''),  # Stock symbol for logging
        ('initial_cash', 15000),
        ('ema_fast_window', 9),
        ('ema_slow_window', 21),
        ('adx_threshold', 20),
        ('momentum_threshold', 0),
        ('mfi_ma_threshold', 70),
        ('natr_multiplier', 1),
        ('profit_target', 250),
    )

    def __init__(self):
        self.vwap = self.data.vwap
        self.kst = self.data.kst
        self.adx = self.data.adx
        self.ema_fast = self.data.ema_fast
        self.ema_slow = self.data.ema_slow
        self.rsi = self.data.rsi
        self.volume_sma = self.data.volume_sma
        self.momentum = self.data.momentum
        self.mfi_ma = self.data.mfi_ma
        self.natr = self.data.natr

        self.take_profit = None
        self.stop_loss = None
        self.trade_id = 0
        self.cum_pnl = 0
        self.buy_comm = None
        self.sell_comm = None
        self.trade_size = None
        self.intraday = self.p.intraday
        self.stoploss_pct = self.p.stoploss_pct
        self.takeprofit_pct = self.p.takeprofit_pct
        self.entry_price = None
        self.stoploss_price = None
        self.takeprofit_price = None
        self.exit_reason = None
        self.ticker = self.p.stock_symbol
        self.entry_pending = False
        self.pending_size = 0
        self.trades = []
        self.balance = self.p.initial_cash
        self.peak_balance = self.p.initial_cash
        self.balance_curve = []
        self.logger = logging.getLogger()

    def stop(self):
        pass

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0).isoformat()
        print('%s, %s' % (dt, txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.datetime(0).isoformat()
            if order.isbuy():
                self.buy_comm = order.executed.comm
                self.trade_size = order.executed.size
                self.sell_price = order.executed.price
                strategy_type = 'MIS' if self.intraday else 'CNC'
                trade_data = {
                    'Datetime': dt,
                    'Transaction ID': self.trade_id,
                    'Transaction type': 'BUY',
                    'Price': round(order.executed.price, 2),
                    'Qty': order.executed.size,
                    'Profit/Loss': 0,
                    'Commission': 0,
                    'Actual Profit/Loss': 0,
                    'Cumulative run': round(self.cum_pnl, 2),
                    'Strategy Type': strategy_type,
                    'Exit Reason': ''
                }
                self.trades.append(trade_data)
                self.logger.info('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                          (order.executed.price,
                              order.executed.value,
                              order.executed.comm))

            elif order.issell():
                self.sell_comm = order.executed.comm
                self.sell_price = order.executed.price
                self.logger.info('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                          (order.executed.price,
                              order.executed.value,
                              order.executed.comm))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.info('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):
        if trade.isclosed:
            if self.intraday:
                if self.buy_comm is not None and self.sell_comm is not None:
                    total_comm = self.buy_comm + self.sell_comm
                    actual_pnl = trade.pnl - total_comm
                else:
                    total_comm = 0
                    actual_pnl = trade.pnl
            else:
                # Custom commissions for non-intraday
                total_comm = 23.78 if trade.pnl > 0 else 21.71
                actual_pnl = trade.pnl - total_comm
                # Deduct commission from broker cash for non-intraday
                self.broker.add_cash(-total_comm)
            self.cum_pnl += actual_pnl
            value = self.broker.getvalue()
            self.peak_balance = max(self.peak_balance, value)
            self.logger.info(f"Trade closed: balance={value}")
            strategy_type = 'MIS' if self.intraday else 'CNC'
            exit_reason = self.exit_reason or ''
            dt = self.datas[0].datetime.datetime(0).isoformat()
            trade_data = {
                'Datetime': dt,
                'Transaction ID': self.trade_id,
                'Transaction type': 'SELL',
                'Price': round(self.sell_price, 2),
                'Qty': self.trade_size,
                'Profit/Loss': round(trade.pnl, 2),
                'Commission': round(total_comm, 2),
                'Actual Profit/Loss': round(actual_pnl, 2),
                'Cumulative run': round(self.cum_pnl, 2),
                'Strategy Type': strategy_type,
                'Exit Reason': exit_reason
            }
            self.trades.append(trade_data)
            self.logger.info('OPERATION %d PROFIT, GROSS %.2f, NET %.2f, ACTUAL %.2f' %
                      (self.trade_id, trade.pnl, trade.pnlcomm, actual_pnl))
            self.take_profit = None
            self.stop_loss = None
            self.buy_comm = None
            self.sell_comm = None
            self.trade_size = None
            self.sell_price = None
            self.entry_price = None
            self.stoploss_price = None
            self.takeprofit_price = None
            self.exit_reason = None
            self.entry_pending = False
            self.pending_size = 0


    def next(self):
        price = self.data.close[0]
        volume = self.data.volume[0]
        current_time = self.data.datetime.time(0)

        # Update balance curve per bar
        dt = self.data.datetime.datetime(0).isoformat()
        balance = self.broker.getvalue()
        self.peak_balance = max(self.peak_balance, balance)
        drawdown = (self.peak_balance - balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.balance_curve.append({"datetime": dt, "balance": balance, "drawdown": drawdown})

        # Execute pending entry on this bar's open (simulated by placing at close of previous signal bar, but delayed)
        if self.entry_pending and not self.position:
            self.trade_id += 1
            self.buy(size=self.pending_size)
            self.entry_price = price  # Use current close as proxy for next open
            self.stoploss_price = self.entry_price * (1 - self.stoploss_pct)
            self.takeprofit_price = self.entry_price * (1 + self.takeprofit_pct)
            self.logger.info(f'BUY {self.trade_id}, Size: {self.pending_size}, Price: %.2f, SL: %.2f, TP: %.2f' % (self.entry_price, self.stoploss_price, self.takeprofit_price))
            self.entry_pending = False
            self.pending_size = 0

        # Intraday rules: no new trades after 15:00, square off at 15:15
        if self.intraday and current_time >= datetime.time(15, 15) and self.position.size != 0:
            self.exit_reason = "Condition"
            self.close()
            return  # No further actions

        # Only allow new entries before 15:00 if intraday
        entry_allowed = not self.intraday or current_time < datetime.time(15, 0)
        if entry_allowed:
            # --- LONG ENTRY ---
            if self.p.only_long or self.p.both:
                if (self.ema_fast[0] > self.ema_slow[0] and
                    self.vwap[0] < price and
                    self.momentum[0] > self.p.momentum_threshold and
                    self.adx[0] > self.p.adx_threshold and
                    not self.position and not self.entry_pending):
                    self.entry_pending = True
                    self.pending_size = int(self.broker.getcash() // price)
                    # Order will be placed on next bar

        # --- EXIT CONDITIONS ---
        if self.position.size > 0:  # Long position
            entry_price = self.position.price
            exit_signal = ((price < self.ema_slow[0] or
                            self.mfi_ma[0] > self.p.mfi_ma_threshold or
                            price < self.vwap[0] - self.natr[0] * self.p.natr_multiplier) and
                           (price - entry_price >= self.p.profit_target))
            sl_tp_hit = (self.stoploss_price is not None and price <= self.stoploss_price) or \
                        (self.takeprofit_price is not None and price >= self.takeprofit_price)
            if exit_signal or sl_tp_hit:
                self.exit_reason = "Condition" if exit_signal else "TP/SL"
                self.close()


def run_backtest(params, start_date, end_date, data_file, initial_cash=15000, intraday=False, strategy_name='ai_v3_streak', optimize_metric='Net Profit', experiment_id=None, window_id=None):
    # Use unified logger from optimizer
    logger = logging.getLogger()

    # Load data with pandas
    df = pd.read_csv(data_file)
    ticker = data_file.split("_")[-1].split(".")[0]  # Extract from filename if needed

    # Compute indicators using ta
    df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['kst'] = ta.trend.kst(df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=params.get('ema_fast', 9))
    df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=params.get('ema_slow', 21))
    df['rsi'] = ta.momentum.rsi(df['close'])
    df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
    df['momentum'] = ta.momentum.roc(df['close'], window=14)
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['mfi_ma'] = ta.trend.sma_indicator(df['mfi'], window=20)
    df['natr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14) / df['close'] * 100

    # Create datetime index
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)

    cerebro = bt.Cerebro()
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    # Add strategy with parameters
    cerebro.addstrategy(MyStrategy, only_long=1, intraday=intraday, stoploss_pct=params.get('stoploss_pct', 0.07), takeprofit_pct=params.get('takeprofit_pct', 0.5), stock_symbol=ticker,
                        initial_cash=initial_cash, ema_fast_window=params.get('ema_fast', 9), ema_slow_window=params.get('ema_slow', 21), adx_threshold=params.get('adx_threshold', 20),
                        momentum_threshold=params.get('momentum_threshold', 0), mfi_ma_threshold=params.get('mfi_ma_threshold', 70),
                        natr_multiplier=params.get('natr_multiplier', 1), profit_target=params.get('profit_target', 250))

    data = MyPandasData(dataname=df, fromdate=start_date, todate=end_date, timeframe=bt.TimeFrame.Minutes, compression=5)

    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)
    if intraday:
        cerebro.broker.addcommissioninfo(NSECommInfo())
    else:
        cerebro.broker.addcommissioninfo(bt.CommInfoBase())

    bt_start = time.time()
    results = cerebro.run()
    bt_time = time.time() - bt_start
    strat = results[0]

    # Create balance DataFrame
    balance_df = pd.DataFrame(strat.balance_curve)
    balance_df["datetime"] = pd.to_datetime(balance_df["datetime"])
    balance_df.set_index("datetime", inplace=True)

    # Ensure final balance is correct
    final_balance = cerebro.broker.getvalue()
    if abs(balance_df["balance"].iloc[-1] - final_balance) > 1e-6:
        balance_df.loc[balance_df.index[-1], "balance"] = final_balance

    logger.info("Backtest execution time: %.2f seconds", bt_time)

    # Final metrics
    final_cash = cerebro.broker.getcash()
    final_value = cerebro.broker.getvalue()
    open_value = final_value - final_cash

    # Extract timeframe from data file path
    path_parts = data_file.split('/')
    timeframe = path_parts[2] if len(path_parts) > 2 else 'day'  # Default to 'day' if not found

    # Generate UUID for this backtest
    backtest_id = str(uuid.uuid4())

    # Save strategy
    save_strategy({'_id': strategy_name, 'name': 'AI V3 Streak Strategy', 'description': 'AI-based streak strategy', 'asset_class': 'equity', 'timeframe': timeframe})

    # Prepare backtest doc
    timestamp = datetime.datetime.now().isoformat()
    backtest_doc = {'_id': backtest_id, 'timestamp': timestamp, 'strategy_id': strategy_name, 'params': params, 'stock_symbol': ticker, 'timeframe': timeframe, 'start_date': start_date.strftime('%Y-%m-%d'), 'end_date': end_date.strftime('%Y-%m-%d'), 'initial_cash': initial_cash, 'final_value': cerebro.broker.getvalue(), 'window_id': window_id, 'experiment_id': experiment_id, 'datetime_range': {'start': start_date.strftime('%Y-%m-%d'), 'end': end_date.strftime('%Y-%m-%d')}}

    # Save trade logs
    save_trades(backtest_id, strat.trades)

    # Save balance curve
    for point in strat.balance_curve:
        point['backtest_id'] = backtest_id
    save_balance_curve(backtest_id, strat.balance_curve)

    # Get analyzer results
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    tradeanalyzer = strat.analyzers.tradeanalyzer.get_analysis()

    # Get timereturn for equity curve and metrics
    timereturn = strat.analyzers.timereturn.get_analysis()
    dates = list(timereturn.keys())

    # Calculate metrics manually
    total_return = (cerebro.broker.getvalue() / initial_cash) - 1
    period_days = (end_date - start_date).days
    if period_days > 0:
        annualized_return = (1 + total_return) ** (365.25 / period_days) - 1
    else:
        annualized_return = 0
    net_profit = strat.cum_pnl
    # Calculate equity curve and drawdown with cumulative compounding
    equity_series = pd.Series(dtype=float)
    peak_equity = initial_cash
    for d in dates:
        equity = initial_cash * (1 + timereturn[d])
        peak_equity = max(peak_equity, equity)
        equity_series[d] = equity
    # Calculate max drawdown from balance curve
    balance_values = np.array([p['balance'] for p in strat.balance_curve])
    if len(balance_values) > 0:
        peaks = np.maximum.accumulate(balance_values)
        drawdowns = (peaks - balance_values) / peaks
        max_drawdown = np.max(drawdowns)
    else:
        max_drawdown = 0
    total_trades = tradeanalyzer.get('total', {}).get('total', 0)
    won_trades = tradeanalyzer.get('won', {}).get('total', 0)
    lost_trades = total_trades - won_trades
    winning_streak = tradeanalyzer.get('streak', {}).get('won', {}).get('longest', 0)
    losing_streak = tradeanalyzer.get('streak', {}).get('lost', {}).get('longest', 0)
    # Get trades from strategy
    trades = strat.trades
    total_commission = sum(t['Commission'] for t in trades)

    bars_per_year = {
        'minute': 375 * 252,
        '3minute': 125 * 252,
        '5minute': 75 * 252,
        '10minute': 37 * 252,
        '15minute': 25 * 252,
        '30minute': 12 * 252,
        '60minute': 6 * 252,
        'day': 252
    }.get(timeframe, 252)

    # Calculate additional ratios
    # Sharpe Ratio
    returns_series = pd.Series(list(timereturn.values()), index=pd.to_datetime(list(timereturn.keys())))
    if not returns_series.empty:
        daily_returns = returns_series.resample('1D').sum()
        mean_return = daily_returns.mean()
        std_return = daily_returns.std(ddof=1)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    equity_curve = np.array([p['balance'] for p in strat.balance_curve])  # full bar-level balance history
    returns = np.diff(equity_curve) / equity_curve[:-1] if len(equity_curve) > 1 else np.array([0.0])
    sharpe = sharpe_ratio
    psr = calculate_psr(returns, sharpe)
    dsr = calculate_dsr(returns, sharpe)
    mdd_bars = calculate_max_drawdown_duration(equity_curve)
    mdd_days = calculate_max_drawdown_duration_days(equity_curve, timeframe)


    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if not downside_returns.empty:
        downside_dev = downside_returns.std(ddof=1)
        sortino_ratio = (mean_return / downside_dev) * np.sqrt(252) if downside_dev > 0 else 0
    else:
        sortino_ratio = 0

    # Calmar Ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Profit Factor
    gross_profit = tradeanalyzer.get('won', {}).get('pnl', {}).get('total', 0)
    gross_loss = abs(tradeanalyzer.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)

    # Prepare metrics
    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Net Profit': net_profit,
        'Max Drawdown': max_drawdown,
        'Winning Streak': winning_streak,
        'Losing Streak': losing_streak,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Profit Factor': profit_factor,
        'PSR': psr,
        'DSR': dsr,
        'Max Drawdown Duration (bars)': mdd_bars,
        'Max Drawdown Duration (days)': mdd_days,
        'Final Cash': final_cash,
        'Final Portfolio Value': final_value
    }
    composite_score = 0.3 * sharpe_ratio + 0.3 * sortino_ratio + 0.3 * calmar_ratio + 0.1 * profit_factor
    composite = {
        'sharpe': sharpe_ratio,
        'sortino': sortino_ratio,
        'calmar': calmar_ratio,
        'profit_factor': profit_factor,
        'composite_score': composite_score
    }

    # Save backtest run
    backtest_doc["metrics"] = metrics
    save_backtest(backtest_doc, metrics, composite)

    return metrics, backtest_id, total_trades, won_trades, lost_trades, total_commission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtesting Script for AI V3 Streak Strategy')
    parser.add_argument('--data_file', type=str, default='../data/5minute/910_NSE_EICHERMOT.csv', help='Path to the data CSV file')
    parser.add_argument('--start_date', type=str, default='2022-01-01', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-09-25', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--initial_cash', type=int, default=15000, help='Initial cash for backtesting')
    parser.add_argument('--intraday', type=str_to_bool, default=False, help='Intraday mode (True/False)')
    parser.add_argument('--stoploss_pct', type=float, default=0.07, help='Stop loss percentage')
    parser.add_argument('--takeprofit_pct', type=float, default=0.5, help='Take profit percentage')
    parser.add_argument('--ema_fast', type=int, default=9, help='EMA fast window')
    parser.add_argument('--ema_slow', type=int, default=21, help='EMA slow window')
    parser.add_argument('--adx_threshold', type=int, default=20, help='ADX threshold')
    parser.add_argument('--momentum_threshold', type=float, default=0, help='Momentum threshold')
    parser.add_argument('--mfi_ma_threshold', type=int, default=70, help='MFI MA threshold')
    parser.add_argument('--natr_multiplier', type=float, default=1, help='NATR multiplier')
    parser.add_argument('--profit_target', type=int, default=250, help='Profit target in rupees')
    parser.add_argument('--strategy_name', type=str, default='ai_v3_streak', help='Strategy name')
    parser.add_argument('--optimize_metric', type=str, default='Net Profit', help='Metric to optimize')
    parser.add_argument('--experiment_id', type=str, default=None, help='Experiment ID')
    parser.add_argument('--window_id', type=str, default=None, help='Window ID')

    args = parser.parse_args()

    params = {
        'ema_fast': args.ema_fast,
        'ema_slow': args.ema_slow,
        'adx_threshold': args.adx_threshold,
        'momentum_threshold': args.momentum_threshold,
        'natr_multiplier': args.natr_multiplier,
        'profit_target': args.profit_target,
        'stoploss_pct': args.stoploss_pct,
        'takeprofit_pct': args.takeprofit_pct
    }

    start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')

    metrics, backtest_id, total_trades, won_trades, lost_trades, total_commission = run_backtest(params, start_date, end_date, args.data_file, args.initial_cash, args.intraday, args.strategy_name, args.optimize_metric, args.experiment_id, args.window_id)

    print(f'Backtest UUID: {backtest_id}')
    print(f'Initial Cash: {args.initial_cash}')
    print('Starting Portfolio Value: %.2f' % args.initial_cash)
    print('Final Portfolio Value: %.2f' % metrics.get('Final Portfolio Value', 0))

    # Create table
    table = [
        ["Total/Cumulative Return", f"{metrics['Total Return']:.2%}"],
        ["Annualized Return (CAGR)", f"{metrics['Annualized Return']:.2%}"],
        ["Net Profit", f"{metrics['Net Profit']:.2f}"],
        ["Maximum Drawdown (MDD)", f"{metrics['Max Drawdown']:.2%}"],
        ["Total number of trades", str(total_trades)],
        ["Number of wins", str(won_trades)],
        ["Number of losses", str(lost_trades)],
        ["Winning streak", str(int(metrics['Winning Streak']))],
        ["Losing streak", str(int(metrics['Losing Streak']))],
        ["Total commission (Zerodha)", f"{total_commission:.2f}"],
        ["Actual profit", f"{metrics['Net Profit']:.2f}"],
        ["Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}"],
        ["Sortino Ratio", f"{metrics['Sortino Ratio']:.3f}"],
        ["Calmar Ratio", f"{metrics['Calmar Ratio']:.3f}"],
        ["Profit Factor", f"{metrics['Profit Factor']:.3f}"],
        ["PSR", f"{metrics['PSR']:.3f}"],
        ["DSR", f"{metrics['DSR']:.3f}"],
        ["Max Drawdown Duration (bars)", f"{metrics['Max Drawdown Duration (bars)']}"],
        ["Max Drawdown Duration (days)", f"{metrics['Max Drawdown Duration (days)']:.2f}"],
    ]

    print("\nStrategy Performance Metrics:")
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))

    print(f"Max drawdown: {metrics['Max Drawdown']:.2%}")

    # Output metrics for optimizer
    for key, value in metrics.items():
        print(f"{key}: {value}")
    optimized_value = metrics.get(args.optimize_metric, metrics['Net Profit'])
    print(f"Optimized Metric ({args.optimize_metric}): {optimized_value}")
