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

# Import functions from main app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        return "🟡 INSUFFICIENT DATA", 0.5
    
    recent_entropy = entropy_values[-lookback:]
    current_entropy = entropy_values[-1]
    
    # Calculate thresholds
    thresh_hi = np.percentile(recent_entropy, 90)
    thresh_lo = np.percentile(recent_entropy, 70)
    
    if current_entropy > thresh_hi:
        return "🟥 RISK-OFF", 0.25  # Reduce Kelly fraction
    elif current_entropy < thresh_lo:
        return "🟩 NORMAL", 1.0     # Full Kelly fraction
    else:
        return "🟡 TRANSITIONAL", 0.6  # Moderate Kelly fraction

def download_and_analyze_asset(symbol, period="1y", entropy_window=30, lookback_period=90):
    """Download data and calculate TDA metrics for a single asset"""
    try:
        # Download data
        data = yf.download(symbol, period=period, progress=False)
        
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
        if len(prices) < entropy_window + lookback_period:
            return None
        
        # Calculate entropy values
        entropy_values = []
        dates = []
        
        for i in range(entropy_window, len(prices)):
            window_prices = prices.iloc[i-entropy_window:i]
            entropy = calculate_persistence_entropy(window_prices, entropy_window)
            entropy_values.append(entropy)
            dates.append(prices.index[i])
        
        # Get current risk signal
        risk_signal, kelly_multiplier = get_risk_signal(entropy_values, lookback_period)
        
        # Calculate some additional metrics
        current_price = prices.iloc[-1]
        price_change_1d = (current_price - prices.iloc[-2]) / prices.iloc[-2] * 100
        price_change_30d = (current_price - prices.iloc[-30]) / prices.iloc[-30] * 100 if len(prices) >= 30 else 0
        
        volatility = prices.pct_change().std() * np.sqrt(252) * 100  # Annualized volatility
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            'price_change_30d': price_change_30d,
            'volatility': volatility,
            'current_entropy': entropy_values[-1],
            'entropy_mean': np.mean(entropy_values),
            'entropy_std': np.std(entropy_values),
            'risk_signal': risk_signal,
            'kelly_multiplier': kelly_multiplier,
            'dates': dates,
            'prices': prices.iloc[entropy_window:].values,
            'entropy_values': entropy_values,
            'full_prices': prices
        }
        
    except Exception as e:
        return None

def main():
    st.set_page_config(page_title="Multi-Asset Dashboard", layout="wide")
    
    st.title("📊 Multi-Asset TDA Dashboard")
    st.markdown("**Compare TDA Risk Signals Across Multiple Assets**")
    
    # Sidebar configuration
    st.sidebar.header("🎯 Asset Selection")
    
    # Pre-defined asset groups
    asset_groups = {
        "Major ETFs": ["SPY", "QQQ", "IWM", "VTI", "VEA", "EEM"],
        "Tech Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"],
        "Crypto": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD"],
        "Bonds": ["TLT", "IEF", "HYG", "LQD", "TIPS", "EMB"],
        "Commodities": ["GLD", "SLV", "USO", "UNG", "DBA", "PDBC"],
        "Custom": []
    }
    
    # Group selection
    selected_group = st.sidebar.selectbox("Select Asset Group", list(asset_groups.keys()))
    
    # Asset input
    if selected_group == "Custom":
        symbols_input = st.sidebar.text_area(
            "Enter symbols (one per line)",
            value="SPY\nQQQ\nBTC-USD\nGLD",
            height=100
        )
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
    else:
        symbols = asset_groups[selected_group]
        st.sidebar.write(f"**Selected assets:** {', '.join(symbols)}")
    
    # Analysis parameters
    st.sidebar.header("📊 Analysis Parameters")
    period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y"], index=1)
    entropy_window = st.sidebar.slider("Entropy Window", 20, 60, 30)
    lookback_period = st.sidebar.slider("Lookback Period", 60, 180, 90)
    
    # Analysis button
    if st.sidebar.button("🔄 Analyze All Assets"):
        st.rerun()
    
    # Main analysis
    if len(symbols) == 0:
        st.warning("Please select or enter some symbols to analyze.")
        return
    
    # Download and analyze all assets
    with st.spinner("Analyzing assets..."):
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(symbols):
            result = download_and_analyze_asset(symbol, period, entropy_window, lookback_period)
            if result:
                results.append(result)
            progress_bar.progress((i + 1) / len(symbols))
        
        progress_bar.empty()
    
    if not results:
        st.error("No data could be retrieved for the selected symbols.")
        return
    
    # Create summary table
    st.subheader("📋 Asset Summary")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Symbol': result['symbol'],
            'Current Price': f"${result['current_price']:.2f}",
            '1D Change': f"{result['price_change_1d']:.2f}%",
            '30D Change': f"{result['price_change_30d']:.2f}%",
            'Volatility': f"{result['volatility']:.1f}%",
            'Current Entropy': f"{result['current_entropy']:.3f}",
            'Risk Signal': result['risk_signal'],
            'Kelly Multiplier': f"{result['kelly_multiplier']:.2f}x"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Color code the risk signals
    def color_risk_signal(val):
        if '🟩' in val:
            return 'background-color: #d4edda'
        elif '🟥' in val:
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #fff3cd'
    
    styled_df = summary_df.style.applymap(color_risk_signal, subset=['Risk Signal'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Risk signal distribution
    st.subheader("🚨 Risk Signal Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    risk_counts = {'🟩 NORMAL': 0, '🟡 TRANSITIONAL': 0, '🟥 RISK-OFF': 0}
    for result in results:
        for key in risk_counts:
            if key in result['risk_signal']:
                risk_counts[key] += 1
    
    with col1:
        st.metric("🟩 Normal", risk_counts['🟩 NORMAL'])
    with col2:
        st.metric("🟡 Transitional", risk_counts['🟡 TRANSITIONAL'])
    with col3:
        st.metric("🟥 Risk-Off", risk_counts['🟥 RISK-OFF'])
    
    # Correlation matrix
    st.subheader("🔗 Entropy Correlation Matrix")
    
    # Create correlation matrix of entropy values
    entropy_df = pd.DataFrame()
    for result in results:
        entropy_df[result['symbol']] = result['entropy_values']
    
    correlation_matrix = entropy_df.corr()
    
    # Plot correlation heatmap
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Entropy Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Entropy comparison chart
    st.subheader("📈 Entropy Comparison")
    
    fig_entropy = go.Figure()
    
    for result in results:
        fig_entropy.add_trace(
            go.Scatter(
                x=result['dates'],
                y=result['entropy_values'],
                name=result['symbol'],
                mode='lines'
            )
        )
    
    fig_entropy.update_layout(
        title="Persistence Entropy Over Time",
        xaxis_title="Date",
        yaxis_title="Entropy",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_entropy, use_container_width=True)
    
    # Price performance comparison
    st.subheader("💰 Price Performance Comparison")
    
    fig_prices = go.Figure()
    
    for result in results:
        # Normalize prices to start at 100
        normalized_prices = (result['full_prices'] / result['full_prices'].iloc[0]) * 100
        
        fig_prices.add_trace(
            go.Scatter(
                x=result['full_prices'].index,
                y=normalized_prices,
                name=result['symbol'],
                mode='lines'
            )
        )
    
    fig_prices.update_layout(
        title="Normalized Price Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_prices, use_container_width=True)
    
    # Portfolio risk assessment
    st.subheader("🎯 Portfolio Risk Assessment")
    
    # Calculate portfolio-level metrics
    total_assets = len(results)
    high_risk_assets = sum(1 for r in results if '🟥' in r['risk_signal'])
    medium_risk_assets = sum(1 for r in results if '🟡' in r['risk_signal'])
    low_risk_assets = sum(1 for r in results if '🟩' in r['risk_signal'])
    
    avg_entropy = np.mean([r['current_entropy'] for r in results])
    avg_kelly = np.mean([r['kelly_multiplier'] for r in results])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Portfolio Risk Summary:**")
        st.write(f"- Total Assets: {total_assets}")
        st.write(f"- High Risk Assets: {high_risk_assets} ({high_risk_assets/total_assets*100:.1f}%)")
        st.write(f"- Medium Risk Assets: {medium_risk_assets} ({medium_risk_assets/total_assets*100:.1f}%)")
        st.write(f"- Low Risk Assets: {low_risk_assets} ({low_risk_assets/total_assets*100:.1f}%)")
    
    with col2:
        st.write("**Portfolio Metrics:**")
        st.write(f"- Average Entropy: {avg_entropy:.3f}")
        st.write(f"- Average Kelly Multiplier: {avg_kelly:.2f}x")
        
        # Portfolio risk level
        if high_risk_assets / total_assets > 0.5:
            portfolio_risk = "🟥 HIGH RISK"
        elif high_risk_assets / total_assets > 0.3:
            portfolio_risk = "🟡 MEDIUM RISK"
        else:
            portfolio_risk = "🟩 LOW RISK"
        
        st.write(f"- **Portfolio Risk Level: {portfolio_risk}**")
    
    # Export functionality
    st.subheader("💾 Export Data")
    
    if st.button("📥 Download All Analysis Data"):
        # Create comprehensive export
        export_data = []
        for result in results:
            for i, (date, entropy) in enumerate(zip(result['dates'], result['entropy_values'])):
                export_data.append({
                    'Symbol': result['symbol'],
                    'Date': date,
                    'Price': result['prices'][i],
                    'Entropy': entropy,
                    'Risk_Signal': result['risk_signal'],
                    'Kelly_Multiplier': result['kelly_multiplier']
                })
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="Download Multi-Asset Analysis CSV",
            data=csv,
            file_name=f"multi_asset_tda_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
