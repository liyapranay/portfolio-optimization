import streamlit as st
import pandas as pd
import plotly.express as px
import time
from modules.database_mongo_helper import get_best_optimization_run, get_backtest_by_id, get_balance_curve, get_rolling_metrics, get_optimization_runs

st.title("Overview")

selected_stock = st.session_state.selected_stock
selected_timeframe = st.session_state.selected_timeframe
selected_strategy = st.session_state.selected_strategy
selected_experiment = st.session_state.selected_experiment

# Get optimization runs for the selected experiment
start_time = time.time()
opt_runs = get_optimization_runs(experiment_id=selected_experiment)
query_time = time.time() - start_time
st.write(f"Query executed in {query_time:.4f} seconds")

if not opt_runs:
    st.write("No optimization runs found for selected experiment.")
    st.stop()

# Get the best run (highest best_value)
opt_run = max(opt_runs, key=lambda x: x.get('best_value', 0))

best_params = opt_run['best_params']
best_value = opt_run['best_value']
optimize_metric = opt_run['optimize_metric']
backtest_id = opt_run.get('best_backtest_id')
n_trials = opt_run.get('n_trials', 'N/A')
n_jobs = opt_run.get('n_jobs', 'N/A')

if not backtest_id:
    st.write("Optimization not completed yet.")
    st.stop()

# Get backtest details
backtest = get_backtest_by_id(backtest_id)
if not backtest:
    st.write("No backtest found.")
    st.stop()

metrics = backtest.get('metrics', {}) or backtest.get('test_metrics', {})
composite = backtest.get('composite', {})

sharpe = composite.get('sharpe', 0)
sortino = composite.get('sortino', 0)
calmar = composite.get('calmar', 0)
profit_factor = composite.get('profit_factor', 0)
composite_score = composite.get('composite_score', 0)

# Format metrics
total_return = metrics.get('Total Return', 0)
annualized_return = metrics.get('Annualized Return', 0)
net_profit = metrics.get('Net Profit', 0)
max_drawdown = metrics.get('Max Drawdown', 0)

# Get balance curve
balance_data = get_balance_curve(backtest_id)
balance_data = [{k: v for k, v in item.items() if k not in ['_id', 'backtest_id']} for item in balance_data] if balance_data else []
balance_df = pd.DataFrame(balance_data) if balance_data else pd.DataFrame()

# Get rolling metrics
rolling_data = get_rolling_metrics(backtest_id)
rolling_data = [{k: v for k, v in item.items() if k not in ['_id', 'backtest_id']} for item in rolling_data] if rolling_data else []
rolling_df = pd.DataFrame(rolling_data) if rolling_data else pd.DataFrame()

# UI
st.subheader("Optimization Summary")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Optimization Method:** Optuna (TPE + Hyperband)")
    st.write(f"**Optimize Metric:** {optimize_metric}")
    st.write(f"**Best Value:** {best_value:.4f}")
with col2:
    st.write(f"**n_trials:** {n_trials}")
    st.write(f"**n_jobs:** {n_jobs}")

st.subheader("Best Parameters")
st.json(best_params)

st.subheader("Performance Metrics")
metrics_df = pd.DataFrame({
    'Metric': ['Total Return', 'Annualized Return', 'Net Profit', 'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Profit Factor', 'Composite Score'],
    'Value': [f"{total_return:.2%}", f"{annualized_return:.2%}", f"₹{net_profit:,.2f}", f"{max_drawdown:.2%}", f"{sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}", f"{profit_factor:.2f}", f"{composite_score:.2f}"]
})
st.table(metrics_df)
st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False), "metrics.csv", "text/csv")

st.subheader("End State")
final_cash = metrics.get('Final Cash', 0)
final_value = metrics.get('Final Portfolio Value', 0)
open_value = final_value - final_cash
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Final Cash", f"₹{final_cash:,.2f}")
with col2:
    st.metric("Open Positions Value", f"₹{open_value:,.2f}")
with col3:
    st.metric("Final Portfolio Value", f"₹{final_value:,.2f}")

st.subheader("Balance Curve")
if not balance_df.empty:
    fig = px.line(balance_df, x='datetime', y='balance', title="Balance Curve", line_shape='hv', color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No balance data available.")

st.subheader("Drawdown Curve")
if not balance_df.empty:
    balance_df['drawdown_pct'] = balance_df['drawdown'] * 100
    fig = px.line(balance_df, x='datetime', y='drawdown_pct', title="Drawdown Curve (%)", color_discrete_sequence=['red'])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No drawdown data available.")

st.subheader("Rolling Metrics")
if not rolling_df.empty:
    melted_df = rolling_df.melt(id_vars='datetime', var_name='Metric', value_name='Value')
    fig = px.line(melted_df, x='datetime', y='Value', color='Metric', title="Rolling Metrics")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No rolling metrics available.")