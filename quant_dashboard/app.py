import streamlit as st
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database_mongo_helper import db, get_strategies, get_experiments

def get_filter_options():
    stocks = db.backtests.distinct('stock_symbol')
    timeframes = db.backtests.distinct('timeframe')
    scopes = db.backtests.distinct('scope')
    strategies_list = get_strategies()
    strategies = {s['_id']: s.get('name', s['_id']) for s in strategies_list}
    experiments_list = get_experiments()
    experiments = {e['name']: e['id'] for e in experiments_list}
    return stocks, timeframes, scopes, strategies, experiments

st.set_page_config(page_title="Quant Dashboard", layout="wide")

stocks, timeframes, scopes, strategies, experiments = get_filter_options()

with st.sidebar:
    st.header("Filters")
    selected_stock = st.selectbox("Stock Symbol", stocks, key="selected_stock")
    selected_timeframe = st.selectbox("Timeframe", timeframes, key="selected_timeframe")
    selected_scope = st.selectbox("Scope", scopes, key="selected_scope")
    selected_strategy = st.selectbox("Strategy", list(strategies.keys()), format_func=lambda x: strategies.get(x, x), key="selected_strategy")
    selected_experiment_name = st.selectbox("Experiment", list(experiments.keys()), key="selected_experiment_name")
    selected_experiment = experiments[selected_experiment_name]
    st.session_state.selected_experiment = selected_experiment
    intraday = st.selectbox("Intraday", [False, True], key="intraday")

    st.header("Optimization Info")
    st.write("**Optimizer:** Optuna")
    st.write("**Method:** TPE + Hyperband")
    # Get optimize_metric from selected experiment or opt_run
    # For simplicity, hardcode or fetch
    st.write("**Metric:** Composite Score")

pages = [
    st.Page("pages/1_Overview.py", title="Overview"),
    st.Page("pages/2_Optimization_Explorer.py", title="Optimization Explorer"),
    st.Page("pages/3_Backtests.py", title="Backtests"),
    st.Page("pages/4_Trades.py", title="Trades"),
]

pg = st.navigation(pages)
pg.run()