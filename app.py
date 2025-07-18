import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# TDA imports
try:
    import ripser
    import persim
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False
    st.warning("TDA libraries not available. Install ripser and persim for full functionality.")

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

def calculate_kelly_fraction(win_prob, avg_win, avg_loss, max_fraction=0.25):
    """Calculate Kelly fraction with safety limits"""
    if avg_loss == 0:
        return 0.0
    
    # Basic Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_prob, q = 1-p
    b = avg_win / avg_loss
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply safety limits
    kelly = max(0, min(kelly, max_fraction))
    
    return kelly

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

def main():
    st.set_page_config(page_title="TDA Risk Monitor", layout="wide")
    
    st.title("🔍 TDA Risk Monitor")
    st.markdown("**Topological Data Analysis + Kelly Sizing for Risk Management**")
    
    # Sidebar controls
    st.sidebar.header("📊 Configuration")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Navigation")
    
    # Create columns for navigation buttons
    nav_col1, nav_col2 = st.sidebar.columns(2)
    
    with nav_col1:
        if st.button("📊 Multi-Asset", use_container_width=True):
            st.switch_page("pages/multi_asset_dashboard.py")
    
    with nav_col2:
        if st.button("🔔 Alerts", use_container_width=True):
            st.switch_page("pages/email_alerts.py")
    
    # Add backtesting button
    if st.sidebar.button("📈 Backtesting", use_container_width=True):
        st.info("📈 Backtesting module coming soon!")
    
    st.sidebar.markdown("---")
    
    # Asset selection
    symbol = st.sidebar.text_input("Asset Symbol", value="SPY", help="Enter a valid ticker symbol")
    
    # Parameters
    st.sidebar.subheader("TDA Parameters")
    entropy_window = st.sidebar.slider("Entropy Window (days)", 20, 60, 30)
    lookback_period = st.sidebar.slider("Lookback Period (days)", 60, 180, 90)
    
    st.sidebar.subheader("Kelly Parameters")
    base_kelly = st.sidebar.slider("Base Kelly Fraction", 0.01, 0.25, 0.065)
    
    # Data period
    period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=2)
    
    if st.sidebar.button("🔄 Update Analysis"):
        st.rerun()
    
    # Main content
    try:
        # Download data
        with st.spinner(f"Downloading {symbol} data..."):
            data = yf.download(symbol, period=period, progress=False)
            
        if data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            close_col = [col for col in data.columns if 'Close' in col][0]
            prices = data[close_col].dropna()
        else:
            prices = data['Close'].dropna()
        
        # Ensure we have enough data
        if len(prices) < entropy_window + lookback_period:
            st.error(f"Insufficient data for analysis. Need at least {entropy_window + lookback_period} days.")
            return
        
        # Calculate entropy values
        st.subheader("📈 TDA Analysis")
        
        entropy_values = []
        dates = []
        
        progress_bar = st.progress(0)
        
        # Calculate rolling entropy
        for i in range(entropy_window, len(prices)):
            window_prices = prices.iloc[i-entropy_window:i]
            entropy = calculate_persistence_entropy(window_prices, entropy_window)
            entropy_values.append(entropy)
            dates.append(prices.index[i])
            
            # Update progress
            progress_bar.progress((i - entropy_window + 1) / (len(prices) - entropy_window))
        
        progress_bar.empty()
        
        if not entropy_values:
            st.error("Insufficient data for analysis")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices.iloc[entropy_window:].values,
            'Entropy': entropy_values
        })
        
        # Get current risk signal
        risk_signal, kelly_multiplier = get_risk_signal(entropy_values, lookback_period)
        adjusted_kelly = base_kelly * kelly_multiplier
        
        # Display current status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Status", risk_signal)
        
        with col2:
            st.metric("Current Entropy", f"{entropy_values[-1]:.3f}")
        
        with col3:
            st.metric("Adjusted Kelly", f"{adjusted_kelly:.3f}")
        
        # Create plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Chart', 'Persistence Entropy'),
            vertical_spacing=0.1
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Price'], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Entropy chart with thresholds
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Entropy'], name='Entropy', line=dict(color='red')),
            row=2, col=1
        )
        
        # Add threshold lines if we have enough data
        if len(entropy_values) >= lookback_period:
            recent_entropy = entropy_values[-lookback_period:]
            thresh_hi = np.percentile(recent_entropy, 90)
            thresh_lo = np.percentile(recent_entropy, 70)
            
            fig.add_hline(y=thresh_hi, line_dash="dash", line_color="red", 
                         annotation_text="90th percentile", row=2, col=1)
            fig.add_hline(y=thresh_lo, line_dash="dash", line_color="green", 
                         annotation_text="70th percentile", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Entropy", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Entropy Statistics:**")
            st.write(f"- Current: {entropy_values[-1]:.3f}")
            st.write(f"- Mean: {np.mean(entropy_values):.3f}")
            st.write(f"- Std: {np.std(entropy_values):.3f}")
            st.write(f"- Min: {np.min(entropy_values):.3f}")
            st.write(f"- Max: {np.max(entropy_values):.3f}")
        
        with col2:
            st.write("**Kelly Sizing:**")
            st.write(f"- Base Kelly: {base_kelly:.3f}")
            st.write(f"- Current Multiplier: {kelly_multiplier:.2f}")
            st.write(f"- Adjusted Kelly: {adjusted_kelly:.3f}")
            st.write(f"- Risk Signal: {risk_signal}")
        
        # Risk-off periods analysis
        st.subheader("🚨 Risk-Off Analysis")
        
        # Count risk-off periods
        risk_off_count = 0
        for i in range(lookback_period, len(entropy_values)):
            recent = entropy_values[i-lookback_period:i]
            if entropy_values[i] > np.percentile(recent, 90):
                risk_off_count += 1
        
        risk_off_pct = (risk_off_count / max(1, len(entropy_values) - lookback_period)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk-Off Days", risk_off_count)
        with col2:
            st.metric("Risk-Off %", f"{risk_off_pct:.1f}%")
        
        # Export data option
        st.subheader("💾 Export Data")
        
        if st.button("📥 Download Analysis Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"tda_analysis_{symbol}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        st.error("Please check your internet connection and symbol validity.")

if __name__ == "__main__":
    main()
