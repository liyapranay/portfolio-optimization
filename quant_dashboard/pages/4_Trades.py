import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
from modules.database_mongo_helper import db, get_trades, get_balance_curve, get_optimization_runs, get_windows

def format_metric(name, value):
    if value is None:
        return "-"
    if "return" in name.lower() or "cagr" in name.lower():
        return f"{value:.2%}"
    elif name.lower() in ["sharpe ratio", "sortino ratio", "calmar ratio", "profit factor"]:
        return f"{value:.2f}"
    elif any(kw in name.lower() for kw in ["profit", "cash", "commission", "value"]):
        return f"₹{value:,.2f}"
    else:
        return str(value)

st.title("Trades")

selected_strategy = st.session_state.selected_strategy
selected_experiment = st.session_state.selected_experiment

# Get optimization runs for the selected experiment
opt_runs_list = get_optimization_runs(experiment_id=selected_experiment)
opt_runs = pd.DataFrame(opt_runs_list)
if '_id' in opt_runs.columns:
    opt_runs['_id'] = opt_runs['_id'].astype(str)

if opt_runs.empty:
    st.write("No optimization runs found for this experiment.")
    st.stop()

selected_opt_run = st.selectbox("Select Optimization Run", opt_runs['_id'].tolist(), format_func=lambda x: f"{x} ({opt_runs[opt_runs['_id']==x]['timestamp'].iloc[0]})")

# Get windows
windows_list = get_windows(selected_opt_run)
windows_df = pd.DataFrame(windows_list)
if '_id' in windows_df.columns:
    windows_df['_id'] = windows_df['_id'].astype(str)

if windows_df.empty:
    st.write("No windows found.")
    st.stop()

# Backtest options
backtest_options = windows_df['_id'].tolist() + ["global"]
backtest_labels = [f"Window {windows_df[windows_df['_id']==x]['window_number'].iloc[0]}" for x in windows_df['_id'].tolist()] + ["Global"]
selected_backtest = st.selectbox("Select Backtest", backtest_options, format_func=lambda x: backtest_labels[backtest_options.index(x)])

# Get backtest
if selected_backtest == "global":
    backtest = db.backtests.find_one({'optimization_run_id': uuid.UUID(selected_opt_run), 'scope': 'global'})
else:
    backtest = db.backtests.find_one({'window_id': selected_backtest})

if not backtest:
    st.write("No backtest found.")
    st.stop()

backtest_id = backtest.get('_id')

# Get trade logs
trades_list = get_trades(backtest_id)
trades_df = pd.DataFrame(trades_list)
if '_id' in trades_df.columns:
    trades_df['_id'] = trades_df['_id'].astype(str)

# Get balance curve
balance_list = get_balance_curve(backtest_id)
balance_list = [{k: v for k, v in item.items() if k not in ['_id', 'backtest_id']} for item in balance_list] if balance_list else []
balance_df = pd.DataFrame(balance_list)

if trades_df.empty:
    st.write("No trades available.")
    st.stop()

# Trade-level summary metrics
st.subheader("Trade Summary")

# Exit Reason counts
exit_reasons = trades_df['Exit Reason'].value_counts().reset_index()
exit_reasons.columns = ['Exit Reason', 'Count']
col1, col2 = st.columns(2)
with col1:
    st.write("Exit Reason Counts")
    st.table(exit_reasons)
with col2:
    fig = px.bar(exit_reasons, x='Exit Reason', y='Count', title="Exit Reasons", color='Exit Reason')
    st.plotly_chart(fig)

# Total Commission
total_commission = trades_df['Commission'].sum()
st.metric("Total Commission", format_metric("Commission", total_commission))

st.subheader("Trade Logs")
# Format the dataframe
formatted_trades = trades_df.copy()
formatted_trades['Price'] = formatted_trades['Price'].apply(lambda x: format_metric("Price", x))
formatted_trades['Profit/Loss'] = formatted_trades['Profit/Loss'].apply(lambda x: format_metric("Profit", x))
formatted_trades['Commission'] = formatted_trades['Commission'].apply(lambda x: format_metric("Commission", x))
formatted_trades['Actual Profit/Loss'] = formatted_trades['Actual Profit/Loss'].apply(lambda x: format_metric("Profit", x))
formatted_trades['Cumulative run'] = formatted_trades['Cumulative run'].apply(lambda x: format_metric("Profit", x))
st.dataframe(formatted_trades)

# Download trades
csv_trades = trades_df.to_csv(index=False)
st.download_button("Download Trades CSV", csv_trades, "trades.csv", "text/csv")

if not balance_df.empty:
    st.subheader("Balance Curve")
    fig = px.line(balance_df, x='datetime', y='balance', title="Balance Curve", line_shape='hv')
    st.plotly_chart(fig)

    # Download balance
    csv_balance = balance_df.to_csv(index=False)
    st.download_button("Download Balance CSV", csv_balance, "balance.csv", "text/csv")

# For candlestick, since no OHLC, use line chart with markers
st.subheader("Trades on Balance")
if not balance_df.empty and not trades_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=balance_df['datetime'], y=balance_df['balance'], mode='lines', name='Balance', line=dict(color='grey'), line_shape='hv'))
    # Add buy/sell markers
    buy_trades = trades_df[trades_df['Transaction type'].str.lower() == 'buy']
    sell_trades = trades_df[trades_df['Transaction type'].str.lower() == 'sell']
    fig.add_trace(go.Scatter(x=buy_trades['Datetime'], y=buy_trades['Price'], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=sell_trades['Datetime'], y=sell_trades['Price'], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down')))
    fig.update_layout(title="Balance Curve with Trade Markers")
    st.plotly_chart(fig)