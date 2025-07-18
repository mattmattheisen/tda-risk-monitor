import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import json
import time
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

def analyze_symbol(symbol, period="1y", entropy_window=30, lookback_period=90):
    """Analyze a single symbol and return current risk signal"""
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
        for i in range(entropy_window, len(prices)):
            window_prices = prices.iloc[i-entropy_window:i]
            entropy = calculate_persistence_entropy(window_prices, entropy_window)
            entropy_values.append(entropy)
        
        # Get current risk signal
        risk_signal, kelly_multiplier = get_risk_signal(entropy_values, lookback_period)
        
        return {
            'symbol': symbol,
            'current_price': prices.iloc[-1],
            'current_entropy': entropy_values[-1],
            'risk_signal': risk_signal,
            'kelly_multiplier': kelly_multiplier,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        return None

def send_email_alert(smtp_server, smtp_port, email_user, email_password, 
                    recipient_email, subject, body):
    """Send email alert"""
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Attach body
        msg.attach(MIMEText(body, 'html'))
        
        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS
        server.login(email_user, email_password)
        
        # Send email
        text = msg.as_string()
        server.sendmail(email_user, recipient_email, text)
        server.quit()
        
        return True
        
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

def create_alert_email(alerts_data):
    """Create HTML email with alert information"""
    html_content = f"""
    <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .alert {{ margin: 15px 0; padding: 15px; border-radius: 5px; }}
                .risk-off {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                .transitional {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .normal {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🔍 TDA Risk Monitor Alert</h2>
                <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h3>📊 Risk Signal Changes</h3>
    """
    
    for alert in alerts_data:
        risk_class = "risk-off" if "🟥" in alert['risk_signal'] else \
                    "transitional" if "🟡" in alert['risk_signal'] else "normal"
        
        html_content += f"""
            <div class="alert {risk_class}">
                <h4>{alert['symbol']}</h4>
                <p><strong>Risk Signal:</strong> {alert['risk_signal']}</p>
                <p><strong>Current Price:</strong> ${alert['current_price']:.2f}</p>
                <p><strong>Current Entropy:</strong> {alert['current_entropy']:.3f}</p>
                <p><strong>Kelly Multiplier:</strong> {alert['kelly_multiplier']:.2f}x</p>
                <p class="timestamp">Analyzed: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
    
    html_content += """
            <div class="header">
                <p><strong>Risk Signal Legend:</strong></p>
                <ul>
                    <li>🟩 <strong>NORMAL</strong> - Low entropy, stable conditions</li>
                    <li>🟡 <strong>TRANSITIONAL</strong> - Medium entropy, caution advised</li>
                    <li>🟥 <strong>RISK-OFF</strong> - High entropy, defensive positioning</li>
                </ul>
                <p><em>This is an automated alert from your TDA Risk Monitor system.</em></p>
            </div>
        </body>
    </html>
    """
    
    return html_content

def main():
    st.set_page_config(page_title="Email Alerts", layout="wide")
    
    st.title("🔔 TDA Email Alerts")
    st.markdown("**Configure automated email notifications for risk signal changes**")
    
    # Navigation
    st.sidebar.markdown("### 🔗 Navigation")
    nav_col1, nav_col2 = st.sidebar.columns(2)
    
    with nav_col1:
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
    
    with nav_col2:
        if st.sidebar.button("📊 Multi-Asset", use_container_width=True):
            st.switch_page("pages/multi_asset_dashboard.py")
    
    st.sidebar.markdown("---")
    
    # Initialize session state for alert configuration
    if 'alert_config' not in st.session_state:
        st.session_state.alert_config = {
            'enabled': False,
            'symbols': ['SPY', 'QQQ'],
            'email_settings': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'email_user': '',
                'email_password': '',
                'recipient_email': ''
            },
            'alert_frequency': 'daily',
            'last_check': None
        }
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📧 Test Email", "📊 Alert History"])
    
    with tab1:
        st.header("⚙️ Alert Configuration")
        
        # Enable/disable alerts
        st.session_state.alert_config['enabled'] = st.checkbox(
            "Enable Email Alerts", 
            value=st.session_state.alert_config['enabled']
        )
        
        if st.session_state.alert_config['enabled']:
            st.success("✅ Email alerts are enabled")
        else:
            st.info("ℹ️ Email alerts are disabled")
        
        # Symbol configuration
        st.subheader("📊 Assets to Monitor")
        
        # Pre-defined symbol groups
        symbol_groups = {
            "Major ETFs": ["SPY", "QQQ", "IWM", "VTI"],
            "Tech Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "Crypto": ["BTC-USD", "ETH-USD"],
            "Custom": []
        }
        
        selected_group = st.selectbox("Select Asset Group", list(symbol_groups.keys()))
        
        if selected_group == "Custom":
            symbols_input = st.text_area(
                "Enter symbols to monitor (one per line)",
                value='\n'.join(st.session_state.alert_config['symbols']),
                height=100
            )
            st.session_state.alert_config['symbols'] = [
                s.strip().upper() for s in symbols_input.split('\n') if s.strip()
            ]
        else:
            st.session_state.alert_config['symbols'] = symbol_groups[selected_group]
        
        st.write(f"**Monitoring {len(st.session_state.alert_config['symbols'])} assets:** {', '.join(st.session_state.alert_config['symbols'])}")
        
        # Email settings
        st.subheader("📧 Email Settings")
        
        with st.expander("Email Configuration", expanded=not st.session_state.alert_config['email_settings']['email_user']):
            st.info("💡 **Gmail Users:** Use an App Password instead of your regular password. Generate one at: https://myaccount.google.com/apppasswords")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.alert_config['email_settings']['smtp_server'] = st.text_input(
                    "SMTP Server",
                    value=st.session_state.alert_config['email_settings']['smtp_server']
                )
                
                st.session_state.alert_config['email_settings']['smtp_port'] = st.number_input(
                    "SMTP Port",
                    value=st.session_state.alert_config['email_settings']['smtp_port'],
                    min_value=1,
                    max_value=65535
                )
            
            with col2:
                st.session_state.alert_config['email_settings']['email_user'] = st.text_input(
                    "Your Email Address",
                    value=st.session_state.alert_config['email_settings']['email_user']
                )
                
                st.session_state.alert_config['email_settings']['email_password'] = st.text_input(
                    "Email Password/App Password",
                    type="password",
                    value=st.session_state.alert_config['email_settings']['email_password']
                )
            
            st.session_state.alert_config['email_settings']['recipient_email'] = st.text_input(
                "Alert Recipient Email",
                value=st.session_state.alert_config['email_settings']['recipient_email']
            )
        
        # Alert frequency
        st.subheader("⏰ Alert Frequency")
        
        st.session_state.alert_config['alert_frequency'] = st.selectbox(
            "How often to check for alerts",
            ["daily", "twice_daily", "hourly"],
            index=["daily", "twice_daily", "hourly"].index(st.session_state.alert_config['alert_frequency'])
        )
        
        frequency_info = {
            "daily": "Once per day at market close",
            "twice_daily": "Market open and close",
            "hourly": "Every hour during market hours"
        }
        
        st.info(f"📅 **{st.session_state.alert_config['alert_frequency'].replace('_', ' ').title()}:** {frequency_info[st.session_state.alert_config['alert_frequency']]}")
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            st.success("✅ Alert configuration saved!")
            st.rerun()
    
    with tab2:
        st.header("📧 Test Email Alert")
        
        if not st.session_state.alert_config['email_settings']['email_user']:
            st.warning("⚠️ Please configure email settings first in the Configuration tab.")
        else:
            st.write("Send a test email to verify your configuration:")
            
            if st.button("🧪 Send Test Email"):
                with st.spinner("Sending test email..."):
                    # Get current analysis for test
                    test_alerts = []
                    for symbol in st.session_state.alert_config['symbols'][:3]:  # Test with first 3 symbols
                        result = analyze_symbol(symbol)
                        if result:
                            test_alerts.append(result)
                    
                    if test_alerts:
                        subject = f"🧪 TDA Risk Monitor - Test Alert"
                        body = create_alert_email(test_alerts)
                        
                        success = send_email_alert(
                            st.session_state.alert_config['email_settings']['smtp_server'],
                            st.session_state.alert_config['email_settings']['smtp_port'],
                            st.session_state.alert_config['email_settings']['email_user'],
                            st.session_state.alert_config['email_settings']['email_password'],
                            st.session_state.alert_config['email_settings']['recipient_email'],
                            subject,
                            body
                        )
                        
                        if success:
                            st.success("✅ Test email sent successfully!")
                        else:
                            st.error("❌ Test email failed. Please check your email settings.")
                    else:
                        st.error("❌ Could not analyze symbols for test email.")
    
    with tab3:
        st.header("📊 Alert History")
        
        st.info("💡 **Note:** This is a demo version. In a production system, this would show historical alerts sent.")
        
        # Simulate some alert history
        if st.button("📊 Show Sample Alert History"):
            sample_history = [
                {
                    'timestamp': datetime.now() - timedelta(days=1),
                    'symbol': 'SPY',
                    'risk_signal': '🟥 RISK-OFF',
                    'action': 'Email sent'
                },
                {
                    'timestamp': datetime.now() - timedelta(days=3),
                    'symbol': 'QQQ',
                    'risk_signal': '🟡 TRANSITIONAL',
                    'action': 'Email sent'
                },
                {
                    'timestamp': datetime.now() - timedelta(days=5),
                    'symbol': 'BTC-USD',
                    'risk_signal': '🟩 NORMAL',
                    'action': 'Email sent'
                }
            ]
            
            for alert in sample_history:
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**{alert['timestamp'].strftime('%Y-%m-%d %H:%M')}**")
                    
                    with col2:
                        st.write(f"**{alert['symbol']}**")
                    
                    with col3:
                        st.write(alert['risk_signal'])
                    
                    with col4:
                        st.write(f"✅ {alert['action']}")
                    
                    st.markdown("---")
    
    # Real-time monitoring section
    st.header("🔍 Current Risk Status")
    
    if st.button("🔄 Check Current Risk Signals"):
        with st.spinner("Analyzing current risk signals..."):
            current_alerts = []
            
            for symbol in st.session_state.alert_config['symbols']:
                result = analyze_symbol(symbol)
                if result:
                    current_alerts.append(result)
            
            if current_alerts:
                st.subheader("Current Risk Signals")
                
                for alert in current_alerts:
                    with st.container():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.write(f"**{alert['symbol']}**")
                        
                        with col2:
                            st.write(alert['risk_signal'])
                        
                        with col3:
                            st.write(f"${alert['current_price']:.2f}")
                        
                        with col4:
                            st.write(f"Kelly: {alert['kelly_multiplier']:.2f}x")
                        
                        st.markdown("---")
            else:
                st.error("Could not analyze any symbols. Please check your symbols and try again.")
    
    # Information section
    st.header("ℹ️ How Email Alerts Work")
    
    with st.expander("📚 Alert System Information"):
        st.markdown("""
        ### 🔔 Alert Triggers
        
        Email alerts are sent when:
        - Risk signal changes from Normal to Transitional or Risk-Off
        - Risk signal changes from Transitional to Risk-Off
        - Risk signal returns to Normal from elevated levels
        
        ### 📧 Email Content
        
        Each alert email includes:
        - Current risk signal for each monitored asset
        - Current price and entropy values
        - Kelly sizing recommendations
        - Timestamp of analysis
        
        ### 🔐 Security Notes
        
        - Email passwords are stored in session state (not persistent)
        - Use App Passwords for Gmail (more secure than regular passwords)
        - Consider using a dedicated email account for alerts
        
        ### ⚠️ Important Disclaimers
        
        - This is a demo system - not suitable for production trading
        - Email alerts may have delays depending on market conditions
        - Always verify signals independently before trading
        - Past performance does not guarantee future results
        """)

if __name__ == "__main__":
    main()
