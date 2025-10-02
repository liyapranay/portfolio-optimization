import pandas as pd
import modules.database as db
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc

def run_dashboard(backtest_id):
    # Query data from DB
    sell_df = db.get_sell_trades(backtest_id)
    equity_data = db.get_equity_curve(backtest_id)
    backtest_run = db.get_backtest_run(backtest_id)
    metrics = db.get_performance_metrics(backtest_id)

    # Extract data
    dates = [pd.to_datetime(row[0]) for row in equity_data]
    strategy_returns = [row[2] for row in equity_data]  # cumulative_return %
    pnl_list = sell_df['actual_pnl'].tolist() if not sell_df.empty else []
    prices = sell_df['price'].tolist() if not sell_df.empty else []

    # Calculate cumulative profits
    if equity_data:
        initial_value = equity_data[0][1]
        strategy_profits = [row[1] - initial_value for row in equity_data]
        buy_hold_profits = [0] * len(equity_data)  # Assuming buy and hold has 0 profit (holding cash)
    else:
        strategy_profits = []
        buy_hold_profits = []

    # Get metrics
    max_dd_pct = float(metrics.get('Max Drawdown', 0))
    max_dd_rupees = max_dd_pct * initial_value if initial_value else 0
    total_return = float(metrics.get('Total Return', 0))
    annualized_return = float(metrics.get('Annualized Return', 0))
    net_profit = float(metrics.get('Net Profit', 0))
    total_trades = len(sell_df) if not sell_df.empty else 0
    won_trades = len(sell_df[sell_df['actual_pnl'] > 0]) if not sell_df.empty else 0
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    profitable_trades = f"{win_rate:.2f}% ({won_trades}/{total_trades})"
    # Calculate profit factor
    if not sell_df.empty:
        positive_pnl = sell_df[sell_df['actual_pnl'] > 0]['actual_pnl'].sum()
        negative_pnl = abs(sell_df[sell_df['actual_pnl'] < 0]['actual_pnl'].sum())
        profit_factor = positive_pnl / negative_pnl if negative_pnl > 0 else float('inf')
    else:
        profit_factor = 0
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{'colspan': 2}, None],
            [{'colspan': 2}, None],
            [{}, {}]
        ],
        subplot_titles=('Trading Strategy Performance', 'Cumulative Profit vs Buy and Hold', 'Monthly Returns Heatmap', 'Trade Returns Distribution'),
        vertical_spacing=0.1
    )

    # Header metrics
    header_text = f"Total P&L: ₹{net_profit:.2f} | Max Equity Drawdown: ₹{max_dd_rupees:.2f} | Total Trades: {total_trades} | Profitable Trades: {profitable_trades} | Profit Factor: {profit_factor:.3f}"
    fig.add_annotation(
        text=header_text,
        x=0.5, y=1.05,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(size=10),
        align='center',
        bgcolor='wheat',
        opacity=0.8
    )

    # 1st row: Cumulative P&L line and Trade P&L bars
    fig.add_trace(
        go.Scatter(x=dates, y=strategy_profits, mode='lines', name='Cumulative P&L', line=dict(color='red', width=2)),
        row=1, col=1
    )
    if not sell_df.empty:
        sell_dates = pd.to_datetime(sell_df['datetime'], errors='coerce')
        pnl_values = sell_df['actual_pnl']
        colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
        fig.add_trace(
            go.Bar(x=sell_dates, y=pnl_values, marker_color=colors, name='Trade P&L',
                   hovertemplate='Trade P&L: ₹%{y:.2f}<extra></extra>'),
            row=1, col=1
        )

    # 2nd row: Cumulative profit vs buy and hold
    fig.add_trace(
        go.Scatter(x=dates, y=strategy_profits, mode='lines', name='Strategy Cumulative Profit'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=buy_hold_profits, mode='lines', name='Buy & Hold Cumulative Profit', line=dict(dash='dash')),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash='solid', line_color='black', row=2, col=1)

    # 3rd row: Monthly returns heatmap
    if not sell_df.empty:
        sell_df['datetime'] = pd.to_datetime(sell_df['datetime'], errors='coerce')
        sell_df = sell_df.dropna(subset=['datetime'])
        sell_df['Month'] = sell_df['datetime'].dt.to_period('M')
        monthly_pnl = sell_df.groupby('Month')['actual_pnl'].sum().reset_index()
        monthly_pnl['Year'] = monthly_pnl['Month'].dt.year
        monthly_pnl['Month_Num'] = monthly_pnl['Month'].dt.month
        pivot = monthly_pnl.pivot(index='Year', columns='Month_Num', values='actual_pnl')
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                text=[[f'{val:.0f}' for val in row] for row in pivot.values],
                texttemplate='%{text}',
                textfont={"size":10},
                hoverongaps=False
            ),
            row=3, col=1
        )

    # 3rd row: Trade returns distribution
    if pnl_list:
        fig.add_trace(
            go.Histogram(x=pnl_list, nbinsx=20, marker_color='blue', name='P&L Distribution',
                         hovertemplate='P&L: ₹%{x:.2f}<br>Count: %{y}<extra></extra>'),
            row=3, col=2
        )

    fig.update_layout(
        title='Strategy Performance Dashboard',
        height=900,
        showlegend=True
    )
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='P&L (INR)', row=1, col=1)
    fig.update_yaxes(title_text='Profit (INR)', row=2, col=1)
    fig.update_xaxes(title_text='Month', row=3, col=1)
    fig.update_yaxes(title_text='Year', row=3, col=1)
    fig.update_xaxes(title_text='P&L (INR)', row=3, col=2)
    fig.update_yaxes(title_text='Frequency', row=3, col=2)

    # Create Dash app
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1('Strategy Performance Dashboard'),
        dcc.Graph(figure=fig)
    ])

    print("Launching Dash dashboard...")
    app.run(debug=True, port=8050)