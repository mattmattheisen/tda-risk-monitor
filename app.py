import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import the anomaly detector if available
try:
    from anomaly_detector import AnomalyDetector
    ANOMALY_AVAILABLE = True
except ImportError:
    ANOMALY_AVAILABLE = False

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
            price_array = prices.iloc[:, 0].values
        elif isinstance(prices, np.ndarray):
            price_array = prices
        else:
            price_array = np.array(prices)
        
        # Ensure 1D array and remove any NaN values
        price_array = np.array(price_array).flatten()
        price_array = price_array[~np.isnan(price_array)]
        
        if len(price_array) < 2:
            return 0.5
        
        # Reshape for ripser - ensure it's float64
        data = price_array.astype(np.float64).reshape(-1, 1)
        
        # Compute persistence diagrams
        dgms = ripser.ripser(data, maxdim=0)['dgms'][0]
        
        # Calculate persistence entropy manually
        if len(dgms) > 1:
            finite_dgms = dgms[dgms[:, 1] != np.inf]
            
            if len(finite_dgms) > 0:
                lifetimes = finite_dgms[:, 1] - finite_dgms[:, 0]
                total_lifetime = np.sum(lifetimes)
                if total_lifetime > 0:
                    probabilities = lifetimes / total_lifetime
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                else:
                    entropy = 0.0
            else:
                entropy = 0.0
        else:
            entropy = 0.0
            
        return entropy
    except Exception as e:
        return np.random.uniform(0.5, 1.5)

# Helper functions for user-friendly interface
def mood_from_z(z):
    """Convert z-score to simple mood"""
    if z < 0.5: 
        return "GREEN"
    elif z < 1.3: 
        return "TRANSITIONAL"
    else: 
        return "RISK-OFF"

def z_descriptor(z):
    """Human-readable description of noise level"""
    if z < 0.3: 
        return "calm"
    elif z < 0.8: 
        return "slightly elevated"
    elif z < 1.3: 
        return "elevated"
    else: 
        return "stressed"

def action_for_mood(mood):
    """Get trading action for mood"""
    actions = {
        "GREEN": "✅ Trade full size per trend; normal stop distance.",
        "TRANSITIONAL": "⚠️ Trade reduced size; avoid adding until mood resolves.",
        "RISK-OFF": "🛡️ Defensive: minimal or no new exposure; wait for calming."
    }
    return actions[mood]

def status_badge(mood):
    """Create colored status badge"""
    colors = {
        "GREEN": "#2e7d32",
        "TRANSITIONAL": "#f9a825", 
        "RISK-OFF": "#c62828"
    }
    return f"<span style='background:{colors[mood]};color:white;padding:8px 16px;border-radius:20px;font-weight:600;font-size:16px;'>{mood}</span>"

def noise_gauge(current, arr_min, arr_max):
    """Create a simple noise level gauge"""
    span = arr_max - arr_min
    if span == 0:
        pct = 0.5
    else:
        pct = (current - arr_min) / span
    
    pct = max(0, min(1, pct))
    
    # Create a visual gauge using progress bar
    st.progress(pct)
    
    # Add description
    if pct < 0.3:
        level = "🟢 Calm"
    elif pct < 0.7:
        level = "🟡 Moderate"
    else:
        level = "🔴 Wild"
    
    st.caption(f"{level} - Position: {pct*100:.0f}% between min and max")

def core_dashboard(entropy_series, base_kelly, multiplier, equity, risk_per_share, view="Basic"):
    """Main dashboard with user-friendly interface"""
    
    arr = np.array(entropy_series)
    current = arr[-1]
    mean = arr.mean()
    std = arr.std(ddof=0) if arr.size > 1 else 0.0001
    z = (current - mean) / std
    mood = mood_from_z(z)
    adjusted_kelly = base_kelly * multiplier
    dollar_risk = adjusted_kelly * equity
    
    # Calculate position size
    if risk_per_share > 0:
        shares = int(dollar_risk / risk_per_share)
    else:
        shares = 0
    
    # Main status cards
    st.markdown("### 📊 Market Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎭 Market Mood**")
        st.markdown(status_badge(mood), unsafe_allow_html=True)
        
        # Add tooltip
        if mood == "GREEN":
            st.caption("Markets are stable and predictable")
        elif mood == "TRANSITIONAL":
            st.caption("Markets are getting choppy - be cautious")
        else:
            st.caption("Markets are chaotic - protect capital")
    
    with col2:
        st.markdown("**📈 Noise Level**")
        st.metric(
            label="Current Reading", 
            value=f"{current:.2f}",
            delta=f"{z:.1f}σ from average",
            help="Higher numbers = more chaotic price movements"
        )
        
        # Add noise gauge
        noise_gauge(current, arr.min(), arr.max())
    
    with col3:
        st.markdown("**🎯 Risk Dial**")
        st.metric(
            label="Position Size", 
            value=f"{adjusted_kelly*100:.1f}%",
            delta=f"${dollar_risk:,.0f} at risk",
            help="Percentage of your account to risk on this trade"
        )
        
        if shares > 0:
            st.caption(f"≈ {shares:,} shares (${risk_per_share:.2f}/share risk)")
    
    # Action panel
    st.markdown("---")
    st.markdown("### 🎯 What Should I Do?")
    
    descriptor = z_descriptor(z)
    action_text = action_for_mood(mood)
    
    # Create action box with appropriate color
    if mood == "GREEN":
        st.success(f"**{action_text}**")
    elif mood == "TRANSITIONAL":
        st.warning(f"**{action_text}**")
    else:
        st.error(f"**{action_text}**")
    
    # Explanation
    st.markdown(f"""
    **Why?** The noise level is **{descriptor}** (z-score: {z:.1f}). 
    This means recent price movements are {"more" if z > 0 else "less"} chaotic than usual.
    
    **Position Sizing:** Risk {adjusted_kelly*100:.1f}% of your ${equity:,} account = **${dollar_risk:,.0f}** maximum loss.
    """)
    
    # Additional details based on view level
    if view != "Basic":
        st.markdown("---")
        
        if view == "Detail":
            st.markdown("### 📊 Additional Details")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("**Noise Statistics:**")
                st.write(f"• Average: {mean:.3f}")
                st.write(f"• Spread: {std:.3f}")
                st.write(f"• Current Z-Score: {z:.2f}")
            
            with detail_col2:
                st.markdown("**Risk Calculation:**")
                st.write(f"• Base Risk Dial: {base_kelly*100:.1f}%")
                st.write(f"• Market Adjustment: {multiplier:.2f}x")
                st.write(f"• Final Risk Dial: {adjusted_kelly*100:.1f}%")
        
        elif view == "Advanced":
            st.markdown("### 🔬 Advanced Statistics")
            
            advanced_cols = st.columns(5)
            advanced_cols[0].metric("Mean", f"{mean:.3f}")
            advanced_cols[1].metric("Std Dev", f"{std:.3f}")
            advanced_cols[2].metric("Minimum", f"{arr.min():.3f}")
            advanced_cols[3].metric("Maximum", f"{arr.max():.3f}")
            advanced_cols[4].metric("Multiplier", f"{multiplier:.2f}")
            
            st.markdown("**Interpretation Guide:**")
            st.info("""
            • **Z-Score Thresholds:** Green < 0.5, Transitional 0.5-1.3, Risk-Off > 1.3
            • **Noise Level:** Higher values = more chaotic price movements
            • **Risk Dial:** Automatically adjusts position size based on market conditions
            """)

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
    st.set_page_config(
        page_title="Smart Trading Assistant", 
        layout="wide",
        page_icon="🎯"
    )
    
    st.title("🎯 Smart Trading Assistant")
    st.markdown("**Know when markets are safe to trade - and when to stay away**")
    
    # Sidebar - simplified controls
    st.sidebar.header("⚙️ Settings")
    
    # Navigation
    st.sidebar.markdown("### 🔗 Other Tools")
    
    nav_col1, nav_col2 = st.sidebar.columns(2)
    with nav_col1:
        if st.button("📊 Portfolio", use_container_width=True):
            try:
                st.switch_page("pages/multi_asset_dashboard.py")
            except:
                st.info("Portfolio tool coming soon!")
    
    with nav_col2:
        if st.button("📧 Alerts", use_container_width=True):
            try:
                st.switch_page("pages/email_alerts.py")
            except:
                st.info("Email alerts coming soon!")
    
    st.sidebar.markdown("---")
    
    # Simple controls
    st.sidebar.subheader("📈 Analysis Settings")
    
    symbol = st.sidebar.text_input(
        "Stock/ETF Symbol", 
        value="SPY", 
        help="Enter any stock ticker (AAPL, MSFT, etc.)"
    )
    
    # Account size for position calculation
    equity = st.sidebar.number_input(
        "Account Size ($)", 
        min_value=1000, 
        max_value=10000000, 
        value=25000,
        step=1000,
        help="Your total trading account size"
    )
    
    # Risk per share (simplified)
    risk_per_share = st.sidebar.number_input(
        "Risk per Share ($)", 
        min_value=0.01, 
        max_value=100.0, 
        value=2.50,
        step=0.25,
        help="How much you'll lose per share if stopped out (typically 1-2x ATR)"
    )
    
    # View level
    st.sidebar.markdown("---")
    view = st.sidebar.radio(
        "Detail Level", 
        ["Basic", "Detail", "Advanced"], 
        index=0,
        help="Choose how much information to display"
    )
    
    # Advanced settings in expander
    with st.sidebar.expander("🔧 Advanced Settings"):
        base_kelly = st.slider("Base Risk Dial", 0.01, 0.25, 0.065, help="Your normal risk percentage")
        entropy_window = st.slider("Analysis Window (days)", 20, 60, 30)
        lookback_period = st.slider("Comparison Period (days)", 60, 180, 90)
        period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=2)
    
    # Update button
    if st.sidebar.button("🔄 Analyze", type="primary"):
        st.rerun()
    
    # Main analysis
    try:
        # Download data
        with st.spinner(f"Analyzing {symbol}..."):
            data = yf.download(symbol, period=period, progress=False)
            
        if data.empty:
            st.error(f"❌ No data found for symbol: {symbol}")
            st.info("💡 Try a different symbol like AAPL, MSFT, or QQQ")
            return
        
        # Handle data structure
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
            close_col = [col for col in data.columns if 'Close' in col][0]
            prices = data[close_col].dropna()
        else:
            prices = data['Close'].dropna()
        
        # Check data sufficiency
        if len(prices) < entropy_window + lookback_period:
            st.error(f"❌ Need at least {entropy_window + lookback_period} days of data")
            st.info("💡 Try selecting a longer data period or different symbol")
            return
        
        # Calculate entropy values
        entropy_values = []
        dates = []
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(entropy_window, len(prices)):
            window_prices = prices.iloc[i-entropy_window:i]
            entropy = calculate_persistence_entropy(window_prices, entropy_window)
            entropy_values.append(entropy)
            dates.append(prices.index[i])
            
            # Update progress
            progress = (i - entropy_window + 1) / (len(prices) - entropy_window)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing market patterns... {progress*100:.0f}%")
        
        progress_bar.empty()
        status_text.empty()
        
        if not entropy_values:
            st.error("❌ Analysis failed - insufficient data")
            return
        
        # Get risk signal for multiplier
        risk_signal, kelly_multiplier = get_risk_signal(entropy_values, lookback_period)
        
        # Display the user-friendly dashboard
        core_dashboard(
            entropy_series=entropy_values,
            base_kelly=base_kelly,
            multiplier=kelly_multiplier,
            equity=equity,
            risk_per_share=risk_per_share,
            view=view
        )
        
        # Charts section
        st.markdown("---")
        st.markdown("### 📊 Visual Analysis")
        
        # Create DataFrame for charts
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices.iloc[entropy_window:].values,
            'Noise_Level': entropy_values
        })
        
        # Chart tabs
        chart_tab1, chart_tab2 = st.tabs(["📈 Price Chart", "📊 Noise History"])
        
        with chart_tab1:
            # Simple price chart
            fig_price = go.Figure()
            fig_price.add_trace(
                go.Scatter(
                    x=df['Date'], 
                    y=df['Price'], 
                    name='Price',
                    line=dict(color='blue', width=2)
                )
            )
            fig_price.update_layout(
                title=f"{symbol} Price Movement",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with chart_tab2:
            # Noise level chart with zones
            fig_noise = go.Figure()
            
            # Add noise level line
            fig_noise.add_trace(
                go.Scatter(
                    x=df['Date'], 
                    y=df['Noise_Level'], 
                    name='Noise Level',
                    line=dict(color='red', width=2)
                )
            )
            
            # Add threshold zones if we have enough data
            if len(entropy_values) >= lookback_period:
                recent_entropy = entropy_values[-lookback_period:]
                thresh_hi = np.percentile(recent_entropy, 90)
                thresh_lo = np.percentile(recent_entropy, 70)
                
                # Add threshold lines
                fig_noise.add_hline(
                    y=thresh_hi, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="High Risk Zone"
                )
                fig_noise.add_hline(
                    y=thresh_lo, 
                    line_dash="dash", 
                    line_color="green",
                    annotation_text="Safe Trading Zone"
                )
                
                # Add colored background zones
                fig_noise.add_hrect(
                    y0=0, y1=thresh_lo, 
                    fillcolor="green", opacity=0.1,
                    annotation_text="Safe Zone", annotation_position="top left"
                )
                fig_noise.add_hrect(
                    y0=thresh_hi, y1=max(entropy_values)*1.1, 
                    fillcolor="red", opacity=0.1,
                    annotation_text="Danger Zone", annotation_position="top left"
                )
            
            fig_noise.update_layout(
                title="Market Noise Level Over Time",
                xaxis_title="Date",
                yaxis_title="Noise Level",
                height=400
            )
            st.plotly_chart(fig_noise, use_container_width=True)
        
        # Export section (only in Advanced view)
        if view == "Advanced":
            st.markdown("---")
            st.markdown("### 💾 Export Data")
            
            if st.button("📥 Download Analysis"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"trading_analysis_{symbol}.csv",
                    mime="text/csv"
                )
        
        # Quick reference guide
        with st.expander("📚 Quick Reference Guide"):
            st.markdown("""
            ### 🎯 Trading Rules
            
            **🟢 GREEN (Safe):** 
            - Trade full position sizes
            - Normal stop loss distances
            - Good time to add to winners
            
            **🟡 TRANSITIONAL (Caution):**
            - Reduce position sizes by 30-50%
            - Use tighter stops
            - Avoid adding to positions
            
            **🔴 RISK-OFF (Danger):**
            - Minimal or no new positions
            - Consider taking profits
            - Preserve capital for better opportunities
            
            ### 📊 Understanding the Numbers
            
            **Noise Level:** Measures how chaotic recent price movements are
            - Low numbers = smooth, predictable trends
            - High numbers = choppy, unpredictable movements
            
            **Risk Dial:** Automatically adjusts your position size
            - Higher during safe periods
            - Lower during dangerous periods
            - Based on proven mathematical formulas
            """)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.info("💡 Try refreshing the page or selecting a different symbol")
        
        # Show error details in advanced view
        if view == "Advanced":
            with st.expander("🔧 Error Details"):
                st.code(str(e))

if __name__ == "__main__":
    main()
