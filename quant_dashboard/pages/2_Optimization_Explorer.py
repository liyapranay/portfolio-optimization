import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from modules.database_mongo_helper import get_optimization_runs, get_windows, get_iterations
import numpy as np

st.title("Optimization Explorer")

selected_strategy = st.session_state.selected_strategy
selected_experiment = st.session_state.selected_experiment

# Get optimization runs for the selected experiment
opt_runs_list = get_optimization_runs(experiment_id=selected_experiment)
opt_runs = pd.DataFrame(opt_runs_list)

if opt_runs.empty:
    st.warning("No optimization runs found for this experiment.")
    st.stop()

selected_opt_run = st.selectbox("Select Optimization Run", opt_runs['_id'].tolist(), format_func=lambda x: f"{x} ({opt_runs[opt_runs['_id']==x]['timestamp'].iloc[0]})")

# Get windows
windows_list = get_windows(str(selected_opt_run))
windows = pd.DataFrame(windows_list)

if windows.empty:
    st.write("No windows found.")
    st.stop()

selected_window = st.selectbox("Select Window", windows['_id'].tolist(), format_func=lambda x: f"Window {windows[windows['_id']==x]['window_number'].iloc[0]}: {windows[windows['_id']==x]['train_start'].iloc[0]} to {windows[windows['_id']==x]['test_end'].iloc[0]}")

# Get iterations
iterations_list = get_iterations(window_id=selected_window)
iterations_df = pd.DataFrame(iterations_list)

if iterations_df.empty:
    st.write("No iterations found.")
    st.stop()

# Data preparation
iterations_df['iteration'] = iterations_df.get('iteration', iterations_df['trial_id'])
iterations_df['trial_id'] = pd.to_numeric(iterations_df.get('trial_id', iterations_df['iteration']), errors='coerce').fillna(0).astype(int)
iterations_df['target'] = pd.to_numeric(iterations_df['target'], errors='coerce').fillna(0.0)
iterations_df['exec_time'] = pd.to_numeric(iterations_df.get('exec_time', 0), errors='coerce').fillna(0.0)
iterations_df['pruned'] = iterations_df.get('pruned', False)
iterations_df['datetime'] = pd.to_datetime(iterations_df['timestamp'], errors='coerce')

# Sort by trial_id or datetime
if 'trial_id' in iterations_df.columns:
    iterations_df = iterations_df.sort_values('trial_id').reset_index(drop=True)
else:
    iterations_df = iterations_df.sort_values('datetime').reset_index(drop=True)

# Create unique sequence index
iterations_df['seq'] = iterations_df['trial_id'] if 'trial_id' in iterations_df.columns else np.arange(1, len(iterations_df) + 1)

# Compute Best So Far
iterations_df['best_so_far_overall'] = iterations_df['target'].cummax()
iterations_df['best_so_far_phase'] = iterations_df.groupby('phase')['target'].cummax()

# Rolling Average
iterations_df['target_rolling'] = iterations_df['target'].rolling(window=3, min_periods=1).mean()

# Debug check
if not iterations_df['best_so_far_overall'].is_monotonic_increasing:
    st.warning("Best so far overall is not monotonic increasing. Check data.")
    st.write(iterations_df.head(10))

# Parse parameters
iterations_df['params_dict'] = iterations_df['parameters']

param_stats_df = pd.DataFrame()  # Empty dataframe

# UI
st.subheader("Optuna Optimization Results")

# Target vs Trial ID
fig = px.scatter(iterations_df, x='trial_id', y='target', color='pruned', title="Target vs Trial ID",
                 color_discrete_map={False: 'blue', True: 'grey'}, hover_data=['trial_id', 'parameters', 'exec_time', 'datetime'])
fig.update_traces(mode='markers')
st.plotly_chart(fig, use_container_width=True)

# Best So Far vs Trial ID
fig = go.Figure()
fig.add_trace(go.Scatter(x=iterations_df['trial_id'], y=iterations_df['best_so_far_overall'], mode='lines+markers', name='Best So Far', line=dict(color='royalblue', width=3)))
fig.update_layout(title="Best So Far vs Trial ID", xaxis_title="Trial ID", yaxis_title="Best So Far")
st.plotly_chart(fig, use_container_width=True)

# Execution Time per Trial
fig = px.bar(iterations_df, x='trial_id', y='exec_time', title="Execution Time per Trial")
st.plotly_chart(fig, use_container_width=True)

# Parameter Importance (placeholder, since we can't compute from DB)
st.subheader("Parameter Importance")
st.write("Parameter importance calculation requires Optuna study object. Not available from database.")

st.subheader("Parameter Analysis")
if not param_stats_df.empty:
    st.table(param_stats_df)

# Parameter distributions
st.subheader("Parameter Distributions")
param_names = list(iterations_df['params_dict'].iloc[0].keys())
selected_param = st.selectbox("Select Parameter for Distribution", param_names)

param_values = [p[selected_param] for p in iterations_df['params_dict']]
fig = px.histogram(x=param_values, title=f"Distribution of {selected_param}")
st.plotly_chart(fig)

# Scatter: parameter vs target
st.subheader("Parameter vs Target")
fig = px.scatter(iterations_df, x=[p[selected_param] for p in iterations_df['params_dict']], y='target', color='phase', title=f"{selected_param} vs Target", color_discrete_map={'random': 'orange', 'bayesian': 'teal', 'optuna': 'green'})
st.plotly_chart(fig)