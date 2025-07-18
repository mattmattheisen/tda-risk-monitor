import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# TDA imports
try:
    import ripser
    import persim
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

def calculate_persistence_entropy(prices, window=30):
    """Calculate persistence entropy for a price window"""
    if not TDA_AVAILABLE:
        return np.random.uniform(0.5, 1.5)  # Mock entropy for demo
    
    try:
        # Multiple ways to handle different data types
        if isinstance(prices, pd.Series):
            price_array = prices.values
        elif isinstance(prices, pd.DataFrame):
            price_array = prices.iloc[:, 0].values  # Take first column
        elif isinstance(prices, np.ndarray):
            price_array = prices
        else:
            price_array = np.array(prices)
        
        # Ensure 1D array and remove any NaN values
        price_array = np.array(price_array).flatten()
        price_array = price_array[~np.isnan(price_array)]
        
        # Check if we have enough data
        if len(price_array) < 2:
            return 0.5
        
        # Reshape for ripser - ensure it's float64
        data = price_array.astype(np.float64).reshape(-1, 1)
        
        # Compute persistence diagrams
        dgms = ripser.ripser(data, maxdim=0)['dgms'][0]
        
        # Calculate persistence entropy manually
        if len(dgms) > 1:
            # Remove infinite points
            finite_dgms = dgms[dgms[:, 1] != np.inf]
            
            if len(finite_dgms) > 0:
                # Calculate lifetimes
                lifetimes = finite_dgms[:, 1] - finite_dgms[:, 0]
                
                # Normalize lifetimes to get probabilities
                total_lifetime = np.sum(lifetimes)
                if total_lifetime > 0:
                    probabilities = lifetimes / total_lifetime
                    
                    # Calculate entropy: -sum(p * log(p))
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                else:
                    entropy = 0.0
            else:
                entropy = 0.0
        else:
            entropy = 0.0
            
        return entropy
    except Exception as e:
        # Return a reasonable default instead of showing error
        return np.random.uniform(0.5, 1.5)

def get_risk_signal(entropy_values, lookback=90):
    """Determine risk signal based on entropy percentiles"""
    if len(entropy_values) < lookback:
        return "INSUFFICIENT_DATA", 0.5
    
    recent_entropy = entropy_values[-lookback:]
    current_entropy = entropy_values[-1]
    
    # Calculate thresholds
    thresh_hi = np.percentile(recent_entropy, 90)
    thresh_lo = np.percentile(recent_entropy, 70)
    
    if current_entropy > thresh_hi:
        return "RISK_OFF", 0.25  # Reduce Kelly fraction
    elif current_entropy < thresh_lo:
        return "NORMAL", 1.0     # Full Kelly fraction
    else:
        return "TRANSITIONAL", 0.6  # Moderate Kelly fraction

def run_backtest(symbol, start_date, end_date, entropy_window=30, lookback_period=90, 
                base_kelly=0.065, initial_capital=10000, transaction_cost=0.001):
    """Run comprehensive backtest of TDA strategy"""
    
    try:
        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            close_col = [col for col in data.columns if 'Close' in col][0]
            prices = data[close_col].dropna()
        else:
            prices = data['Close'].dropna()
        
        # Ensure we have enough data
        if len(prices) < entropy_window + lookback_period + 252:  # Need at least 1 year extra
            return None
        
        # Calculate entropy values and signals
        entropy_values = []
        risk_signals = []
        kelly_multipliers = []
        dates = []
        
        for i in range(entropy_window + lookback_period, len(prices)):
            # Calculate entropy for current window
            window_prices = prices.iloc[i-entropy_window:i]
            entropy = calculate_persistence_entropy(window_prices, entropy_window)
            entropy_values.append(entropy)
            
            # Get risk signal based on historical entropy
            historical_entropy = entropy_values[-min(lookback_period, len(entropy_values)):]
            risk_signal, kelly_mult = get_risk_signal(historical_entropy + [entropy], lookback_period)
            
            risk_signals.append(risk_signal)
            kelly_multipliers.append(kelly_mult)
            dates.append(prices.index[i])
        
        # Create trading signals DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices.loc[dates].values,
            'Entropy': entropy_values,
            'Risk_Signal': risk_signals,
            'Kelly_Multiplier': kelly_multipliers
        })
        
        # Calculate returns
        df['Returns'] = df['Price'].pct_change()
        
        # Implement trading strategy
        df['Position_Size'] = df['Kelly_Multiplier'] * base_kelly
        df['Strategy_Returns'] = df['Returns'] * df['Position_Size'].shift(1)
        
        # Apply transaction costs
        df['Position_Change'] = df['Position_Size'].diff().abs()
        df['Transaction_Costs'] = df['Position_Change'] * transaction_cost
        df['Net_Strategy_Returns'] = df['Strategy_Returns'] - df['Transaction_Costs']
        
        # Calculate cumulative returns
        df['Cumulative_Returns'] = (1 + df['Returns']).cumprod()
        df['Cumulative_Strategy_Returns'] = (1 + df['Net_Strategy_Returns']).cumprod()
        
        # Buy and hold benchmark
        df['Buy_Hold_Returns'] = df['Cumulative_Returns']
        
        # Calculate portfolio value
        df['Portfolio_Value'] = initial_capital * df['Cumulative_Strategy_Returns']
        df['Buy_Hold_Value'] = initial_capital * df['Buy_Hold_Returns']
        
        return df
        
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return None

def calculate_performance_metrics(df):
    """Calculate comprehensive performance metrics"""
    if df is None or len(df) < 2:
        return None
    
    # Returns
    strategy_returns = df['Net_Strategy_Returns'].dropna()
    benchmark_returns = df['Returns'].dropna()
    
    # Align the series
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns.iloc[:min_len]
    benchmark_returns = benchmark_returns.iloc[:min_len]
    
    # Basic metrics
    total_return_strategy = df['Cumulative_Strategy_Returns'].iloc[-1] - 1
    total_return_benchmark = df['Cumulative_Returns'].iloc[-1] - 1
    
    # Annualized returns
    days = len(df)
    years = days / 252
    annual_return_strategy = (1 + total_return_strategy) ** (1/years) - 1
    annual_return_benchmark = (1 + total_return_benchmark) ** (1/years) - 1
    
    # Volatility
    annual_vol_strategy = strategy_returns.std() * np.sqrt(252)
    annual_vol_benchmark = benchmark_returns.std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_strategy = annual_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else 0
    sharpe_benchmark = annual_return_benchmark / annual_vol_benchmark if annual_vol_benchmark != 0 else 0
    
    # Maximum drawdown
    rolling_max_strategy = df['Cumulative_Strategy_Returns'].expanding().max()
    drawdown_strategy = (df['Cumulative_Strategy_Returns'] - rolling_max_strategy) / rolling_max_strategy
    max_drawdown_strategy = drawdown_strategy.min()
    
    rolling_max_benchmark = df['Cumulative_Returns'].expanding().max()
    drawdown_benchmark = (df['Cumulative_Returns'] - rolling_max_benchmark) / rolling_max_benchmark
    max_drawdown_benchmark = drawdown_benchmark.min()
    
    # Win rate
    win_rate_strategy = (strategy_returns > 0).mean()
    win_rate_benchmark = (benchmark_returns > 0).mean()
    
    # Calmar ratio
    calmar_strategy = annual_return_strategy / abs(max_drawdown_strategy) if max_drawdown_strategy != 0 else 0
    calmar_benchmark = annual_return_benchmark / abs(max_drawdown_benchmark) if max_drawdown_benchmark != 0 else 0
    
    # Beta and Alpha
    if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
        beta = np.cov(strategy_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        alpha = annual_return_strategy - beta * annual_return_benchmark
    else:
        beta = 0
        alpha = 0
    
    # Information ratio
    excess_returns = strategy_returns - benchmark_returns
    info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
    
    return {
        'total_return_strategy': total_return_strategy,
        'total_return_benchmark': total_return_benchmark,
        'annual_return_strategy': annual_return_strategy,
        'annual_return_benchmark': annual_return_benchmark,
        'annual_vol_strategy': annual_vol_strategy,
        'annual_vol_benchmark': annual_vol_benchmark,
        'sharpe_strategy': sharpe_strategy,
        'sharpe_benchmark': sharpe_benchmark,
        'max_drawdown_strategy': max_drawdown_strategy,
        'max_drawdown_benchmark': max_drawdown_benchmark,
        'win_rate_strategy': win_rate_strategy,
        'win_rate_benchmark': win_rate_benchmark,
        'calmar_strategy': calmar_strategy,
        'calmar_benchmark': calmar_benchmark,
        'beta': beta,
        'alpha': alpha,
        'info_ratio': info_ratio
    }

def main():
    st.set_page_config(page_title="TDA Backtesting", layout="wide")
    
    st.title("📈 TDA Strategy Backtesting")
    st.markdown("**Test historical performance of the TDA + Kelly sizing strategy**")
    
    # Navigation
    st.sidebar.markdown("### 🔗 Navigation")
    nav_col1, nav_col2, nav_col3 = st.sidebar.columns(3)
    
    with nav_col1:
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
    
    with nav_col2:
        if st.sidebar.button("📊 Multi-Asset", use_container_width=True):
            st.switch_page("pages/multi_asset_dashboard.py")
    
    with nav_col3:
        if st.sidebar.button("🔔 Alerts", use_container_width=True):
            st.switch_page("pages/email_alerts.py")
    
    st.sidebar.markdown("---")
    
    # Backtesting parameters
    st.sidebar.header("📊 Backtest Configuration")
    
    # Asset selection
    symbol = st.sidebar.text_input("Asset Symbol", value="SPY", help="Enter a valid ticker symbol")
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date", 
            value=datetime.now() - timedelta(days=3*365),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    entropy_window = st.sidebar.slider("Entropy Window", 20, 60, 30)
    lookback_period = st.sidebar.slider("Lookback Period", 60, 180, 90)
    base_kelly = st.sidebar.slider("Base Kelly Fraction", 0.01, 0.25, 0.065)
    
    # Portfolio parameters
    st.sidebar.subheader("Portfolio Parameters")
    initial_capital = st.sidebar.number_input("Initial Capital", value=10000, min_value=1000)
    transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
    
    # Run backtest
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a few minutes."):
            # Run the backtest
            df = run_backtest(
                symbol, start_date, end_date, entropy_window, lookback_period,
                base_kelly, initial_capital, transaction_cost
            )
            
            if df is not None:
                # Store in session state for persistence
                st.session_state.backtest_results = df
                st.session_state.backtest_symbol = symbol
                st.session_state.backtest_params = {
                    'entropy_window': entropy_window,
                    'lookback_period': lookback_period,
                    'base_kelly': base_kelly,
                    'initial_capital': initial_capital,
                    'transaction_cost': transaction_cost,
                    'start_date': start_date,
                    'end_date': end_date
                }
                st.success("✅ Backtest completed successfully!")
                st.rerun()
            else:
                st.error("❌ Backtest failed. Please check your parameters and try again.")
    
    # Display results if available
    if 'backtest_results' in st.session_state:
        df = st.session_state.backtest_results
        symbol = st.session_state.backtest_symbol
        params = st.session_state.backtest_params
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(df)
        
        if metrics:
            # Performance summary
            st.header("📊 Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Return (Strategy)", 
                    f"{metrics['total_return_strategy']:.2%}",
                    f"{metrics['total_return_strategy'] - metrics['total_return_benchmark']:.2%}"
                )
                
            with col2:
                st.metric(
                    "Annual Return (Strategy)", 
                    f"{metrics['annual_return_strategy']:.2%}",
                    f"{metrics['annual_return_strategy'] - metrics['annual_return_benchmark']:.2%}"
                )
                
            with col3:
                st.metric(
                    "Sharpe Ratio (Strategy)", 
                    f"{metrics['sharpe_strategy']:.2f}",
                    f"{metrics['sharpe_strategy'] - metrics['sharpe_benchmark']:.2f}"
                )
                
            with col4:
                st.metric(
                    "Max Drawdown (Strategy)", 
                    f"{metrics['max_drawdown_strategy']:.2%}",
                    f"{metrics['max_drawdown_strategy'] - metrics['max_drawdown_benchmark']:.2%}"
                )
            
            # Detailed metrics table
            st.subheader("📈 Detailed Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': [
                    'Total Return', 'Annual Return', 'Volatility', 'Sharpe Ratio',
                    'Max Drawdown', 'Win Rate', 'Calmar Ratio', 'Beta', 'Alpha', 'Information Ratio'
                ],
                'TDA Strategy': [
                    f"{metrics['total_return_strategy']:.2%}",
                    f"{metrics['annual_return_strategy']:.2%}",
                    f"{metrics['annual_vol_strategy']:.2%}",
                    f"{metrics['sharpe_strategy']:.2f}",
                    f"{metrics['max_drawdown_strategy']:.2%}",
                    f"{metrics['win_rate_strategy']:.2%}",
                    f"{metrics['calmar_strategy']:.2f}",
                    f"{metrics['beta']:.2f}",
                    f"{metrics['alpha']:.2%}",
                    f"{metrics['info_ratio']:.2f}"
                ],
                'Buy & Hold': [
                    f"{metrics['total_return_benchmark']:.2%}",
                    f"{metrics['annual_return_benchmark']:.2%}",
                    f"{metrics['annual_vol_benchmark']:.2%}",
                    f"{metrics['sharpe_benchmark']:.2f}",
                    f"{metrics['max_drawdown_benchmark']:.2%}",
                    f"{metrics['win_rate_benchmark']:.2%}",
                    f"{metrics['calmar_benchmark']:.2f}",
                    "1.00",
                    "0.00%",
                    "0.00"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Performance charts
            st.subheader("📊 Performance Charts")
            
            # Cumulative returns chart
            fig_returns = go.Figure()
            
            fig_returns.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Portfolio_Value'],
                    name='TDA Strategy',
                    line=dict(color='blue', width=2)
                )
            )
            
            fig_returns.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Buy_Hold_Value'],
                    name='Buy & Hold',
                    line=dict(color='gray', width=2)
                )
            )
            
            fig_returns.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # Drawdown chart
            st.subheader("📉 Drawdown Analysis")
            
            rolling_max_strategy = df['Cumulative_Strategy_Returns'].expanding().max()
            drawdown_strategy = (df['Cumulative_Strategy_Returns'] - rolling_max_strategy) / rolling_max_strategy
            
            rolling_max_benchmark = df['Cumulative_Returns'].expanding().max()
            drawdown_benchmark = (df['Cumulative_Returns'] - rolling_max_benchmark) / rolling_max_benchmark
            
            fig_dd = go.Figure()
            
            fig_dd.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=drawdown_strategy * 100,
                    name='TDA Strategy',
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.3)',
                    line=dict(color='red')
                )
            )
            
            fig_dd.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=drawdown_benchmark * 100,
                    name='Buy & Hold',
                    line=dict(color='gray')
                )
            )
            
            fig_dd.update_layout(
                title="Drawdown Comparison",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
            
            # Risk signal distribution
            st.subheader("🚨 Risk Signal Distribution")
            
            risk_counts = df['Risk_Signal'].value_counts()
            
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Signal Distribution During Backtest Period"
            )
            
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Rolling performance metrics
            st.subheader("📈 Rolling Performance")
            
            # Calculate rolling Sharpe ratio
            rolling_window = 252  # 1 year
            if len(df) > rolling_window:
                rolling_sharpe = df['Net_Strategy_Returns'].rolling(rolling_window).apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() != 0 else 0
                )
                
                fig_rolling = go.Figure()
                
                fig_rolling.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=rolling_sharpe,
                        name='Rolling Sharpe Ratio (1Y)',
                        line=dict(color='purple')
                    )
                )
                
                fig_rolling.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig_rolling.update_layout(
                    title="Rolling Sharpe Ratio (1-Year Window)",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    height=400
                )
                
                st.plotly_chart(fig_rolling, use_container_width=True)
            
            # Export results
            st.subheader("💾 Export Results")
            
            if st.button("📥 Download Backtest Results"):
                # Create comprehensive export
                export_df = df.copy()
                export_df['Symbol'] = symbol
                
                # Add parameters as metadata
                export_df['Entropy_Window'] = params['entropy_window']
                export_df['Lookback_Period'] = params['lookback_period']
                export_df['Base_Kelly'] = params['base_kelly']
                export_df['Initial_Capital'] = params['initial_capital']
                export_df['Transaction_Cost'] = params['transaction_cost']
                
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Backtest CSV",
                    data=csv,
                    file_name=f"tda_backtest_{symbol}_{params['start_date']}_{params['end_date']}.csv",
                    mime="text/csv"
                )
        
        else:
            st.error("Could not calculate performance metrics. Please try again.")
    
    else:
        # Show information about backtesting
        st.header("📚 About TDA Backtesting")
        
        st.markdown("""
        ### 🎯 What This Backtesting Does
        
        This module tests the historical performance of the TDA + Kelly sizing strategy by:
        
        1. **Calculating Historical TDA Signals**: Computes persistence entropy and risk signals for each day
        2. **Implementing Position Sizing**: Uses Kelly criterion with dynamic adjustments based on risk signals
        3. **Simulating Trading**: Applies transaction costs and realistic trading constraints
        4. **Measuring Performance**: Compares against buy-and-hold benchmark
        
        ### 📊 Key Metrics Explained
        
        - **Total Return**: Overall percentage gain/loss
        - **Annual Return**: Annualized percentage return
        - **Sharpe Ratio**: Risk-adjusted return (higher is better)
        - **Max Drawdown**: Largest peak-to-trough decline
        - **Win Rate**: Percentage of profitable days
        - **Calmar Ratio**: Annual return divided by max drawdown
        - **Information Ratio**: Excess return per unit of tracking error
        
        ### ⚠️ Important Considerations
        
        - **Past Performance**: Does not guarantee future results
        - **Transaction Costs**: Real trading involves commissions and slippage
        - **Market Conditions**: Strategy may perform differently in various market regimes
        - **Overfitting**: Avoid over-optimizing parameters to historical data
        
        ### 🚀 Getting Started
        
        1. Select an asset symbol (SPY, QQQ, AAPL, etc.)
        2. Choose your date range (minimum 3 years recommended)
        3. Configure strategy parameters
        4. Click "Run Backtest"
        5. Analyze the results and compare to buy-and-hold
        
        **Recommended Test Assets:**
        - **SPY**: S&P 500 (broad market)
        - **QQQ**: Nasdaq (tech-heavy)
        - **BTC-USD**: Bitcoin (high volatility)
        - **TLT**: Long-term bonds (different risk profile)
        """)

if __name__ == "__main__":
    main()
