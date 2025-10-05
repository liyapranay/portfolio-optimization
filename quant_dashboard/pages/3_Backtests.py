import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
from modules.database_mongo_helper import get_optimization_runs, get_windows, db, get_balance_curve

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

st.title("Backtests")

selected_strategy = st.session_state.selected_strategy
selected_experiment = st.session_state.selected_experiment

# Get optimization runs for the selected experiment
opt_runs_list = get_optimization_runs(experiment_id=selected_experiment)
opt_runs = pd.DataFrame(opt_runs_list)

if opt_runs.empty:
    st.write("No optimization runs found for this experiment.")
    st.stop()

selected_opt_run = st.selectbox("Select Optimization Run", opt_runs['_id'].tolist(), format_func=lambda x: f"{x} ({opt_runs[opt_runs['_id']==x]['timestamp'].iloc[0]})")
selected_opt_run_str = str(selected_opt_run)

# Get windows
windows_list = get_windows(selected_opt_run_str)
windows_df = pd.DataFrame(windows_list)

if windows_df.empty:
    st.write("No windows found.")
    st.stop()

# Convert UUIDs to strings for querying
windows_df['_id'] = windows_df['_id'].astype(str)

st.subheader("Windows")
st.dataframe(windows_df[['_id', 'window_number', 'train_start', 'train_end', 'test_start', 'test_end']])

# Get composite metrics for all windows
window_ids = windows_df['_id'].tolist()
backtests = list(db.backtests.find({'window_id': {'$in': window_ids}}))

composite_data = []
for b in backtests:
    if 'composite' in b:
        comp = b['composite']
        window = windows_df[windows_df['_id'] == b['window_id']]
        if not window.empty:
            window_number = int(window['window_number'].iloc[0])  # Convert to regular int
            composite_data.append({
                'sharpe': float(comp.get('sharpe', 0)),
                'sortino': float(comp.get('sortino', 0)),
                'calmar': float(comp.get('calmar', 0)),
                'profit_factor': float(comp.get('profit_factor', 0)),
                'composite_score': float(comp.get('composite_score', 0)),
                'window_number': window_number
            })
composite_df = pd.DataFrame(composite_data)

if not composite_df.empty:
    st.subheader("Composite Score Trend")
    if len(composite_df) == 1:
        # Single point - use scatter plot with larger markers
        fig = px.scatter(composite_df, x='window_number', y='composite_score',
                        title="Composite Score Across Windows",
                        size=[20],  # Larger marker
                        color_discrete_sequence=['red'])
        fig.update_traces(mode='markers+text',
                         text=composite_df['composite_score'].round(3),
                         textposition="top center")
    else:
        # Multiple points - use line plot
        fig = px.line(composite_df, x='window_number', y='composite_score',
                     title="Composite Score Across Windows",
                     markers=True)  # Add markers to line
    st.plotly_chart(fig)
else:
    st.write("No composite data available")

# Select window for detailed view
window_options = windows_df['_id'].tolist() + ["global"]
window_labels = [f"Window {row['window_number']} - Train {row['train_start']} to {row['train_end']} - Test {row['test_start']} to {row['test_end']}" for _, row in windows_df.iterrows()] + ["Global"]
selected_window = st.selectbox("Select Window for Details", window_options, format_func=lambda x: window_labels[window_options.index(x)])

# Get backtest for selected window
if selected_window == "global":
    backtest = db.backtests.find_one({
        'optimization_run_id': uuid.UUID(selected_opt_run_str),
        'scope': 'global'
    })
else:
    backtest = db.backtests.find_one({
        'window_id': selected_window,
        'scope': 'window'    # FIX: only pick window docs, not train
    })

if backtest:
    backtest_id = backtest.get('_id')
    train_metrics_raw = backtest.get('train_metrics', {})
    test_metrics_raw = backtest.get('test_metrics', {}) or backtest.get('metrics', {})

    train_metrics = pd.DataFrame([{'metric_name': k, 'value': format_metric(k, v)} for k, v in train_metrics_raw.items()]) if train_metrics_raw else pd.DataFrame()
    test_metrics = pd.DataFrame([{'metric_name': k, 'value': format_metric(k, v)} for k, v in test_metrics_raw.items()]) if test_metrics_raw else pd.DataFrame()

    st.subheader("Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Train Metrics")
        if train_metrics.empty:
            st.write("No train metrics recorded.")
        else:
            st.dataframe(train_metrics)
    with col2:
        st.write("Test Metrics")
        if test_metrics.empty:
            st.write("No test metrics recorded.")
        else:
            st.dataframe(test_metrics)

    # Overfitting check
    if not train_metrics.empty and not test_metrics.empty:
        train_sharpe = train_metrics_raw.get('Sharpe Ratio')
        test_sharpe = test_metrics_raw.get('Sharpe Ratio')
        if train_sharpe is not None and test_sharpe is not None and train_sharpe != 0:
            diff_pct = abs(train_sharpe - test_sharpe) / abs(train_sharpe)
            if diff_pct > 0.3:
                st.warning(f"⚠️ Overfitting Alert: Train Sharpe {train_sharpe:.2f} vs Test Sharpe {test_sharpe:.2f} ({diff_pct:.1%} difference)")

    # Balance curve
    balance_data = get_balance_curve(backtest_id)
    balance_data = [{k: v for k, v in item.items() if k not in ['_id', 'backtest_id']} for item in balance_data] if balance_data else []
    balance_df = pd.DataFrame(balance_data) if balance_data else pd.DataFrame()
    if not balance_df.empty:
        fig = px.line(balance_df, x='datetime', y='balance', title="Balance Curve", line_shape='hv', color_discrete_sequence=['green'])
        st.plotly_chart(fig)
    else:
        st.write("No balance data available.")

    if not test_metrics.empty:
        st.download_button("Download Metrics CSV", test_metrics.to_csv(index=False), "metrics.csv", "text/csv")
else:
    st.write("No backtest found for selected option.")