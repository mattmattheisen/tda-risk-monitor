# TDA Risk Monitor - User Manual

## 1. Overview

The TDA Risk Monitor is a sophisticated trading tool that combines **Topological Data Analysis (TDA)** with **Kelly Sizing** to provide intelligent risk management for financial markets.

### What Makes This Different?

- **TDA Analysis**: Uses persistence entropy to detect market regime changes
- **Kelly Sizing**: Mathematically optimizes position sizing based on risk conditions
- **Real-time Monitoring**: Continuous analysis with automatic risk alerts
- **Multi-timeframe**: Configurable analysis windows for different trading styles

## 2. Getting Started

### Accessing the Application

Your TDA Risk Monitor is available at: [Your Streamlit App URL]

### Basic Navigation

1. **Main Dashboard**: Shows current risk status and entropy analysis
2. **Sidebar Controls**: Configure symbols, parameters, and time periods
3. **Interactive Charts**: Price and entropy visualization with threshold lines
4. **Export Options**: Download analysis data for further processing

## 3. Understanding TDA (Topological Data Analysis)

### The "LEGO Snake" Analogy

Think of your price chart as a LEGO snake:
- **Smooth sections** = Low entropy = Predictable trends
- **Wiggly sections** = High entropy = Chaotic/dangerous periods
- **Persistence entropy** measures how "wiggly" the snake is

### How It Works

1. **Price Window**: Takes the last 30 days of price data
2. **Shape Analysis**: Converts prices into a mathematical "shape"
3. **Entropy Calculation**: Measures the complexity of that shape
4. **Risk Signal**: Compares current entropy to historical percentiles

### Risk Signal Interpretation

| Signal | Meaning | Action |
|--------|---------|--------|
| 🟩 **NORMAL** | Entropy < 70th percentile | Full Kelly position sizing |
| 🟡 **TRANSITIONAL** | Entropy between 70th-90th percentile | Moderate position sizing |
| 🟥 **RISK-OFF** | Entropy > 90th percentile | Reduced position sizing |

## 4. Kelly Sizing Strategy

### What is Kelly Sizing?

The Kelly Criterion mathematically determines the optimal bet size to maximize long-term growth while minimizing risk of ruin.

**Formula**: f = (bp - q) / b
- f = fraction of capital to risk
- b = ratio of win amount to loss amount
- p = probability of winning
- q = probability of losing (1-p)

### Dynamic Adjustment

The TDA Risk Monitor adjusts Kelly sizing based on market conditions:

- **Normal Market**: Kelly × 1.0 (full sizing)
- **Transitional**: Kelly × 0.6 (moderate sizing)
- **Risk-Off**: Kelly × 0.25 (defensive sizing)

### Safety Limits

- **Maximum Kelly**: Capped at 25% of capital
- **Minimum Kelly**: Never below 1% of capital
- **Negative Kelly**: Position size set to 0

## 5. Using the Application

### Step 1: Configure Your Analysis

1. **Asset Symbol**: Enter any valid ticker (SPY, QQQ, AAPL, BTC-USD, etc.)
2. **Entropy Window**: Days to analyze (20-60, default 30)
3. **Lookback Period**: Historical context (60-180, default 90)
4. **Base Kelly**: Your base position sizing (1%-25%, default 6.5%)
5. **Data Period**: How much history to analyze (6mo-5y, default 2y)

### Step 2: Interpret the Results

**Current Status Metrics:**
- **Current Status**: Real-time risk signal
- **Current Entropy**: Latest entropy reading
- **Adjusted Kelly**: Recommended position size

**Charts:**
- **Price Chart**: Shows asset price movement
- **Entropy Chart**: Shows persistence entropy with threshold lines
- **Threshold Lines**: Red (90th percentile), Green (70th percentile)

**Statistics:**
- **Entropy Stats**: Mean, std dev, min/max entropy values
- **Kelly Sizing**: Current multipliers and recommendations
- **Risk-Off Analysis**: Historical frequency of risk-off periods

### Step 3: Export and Use Data

1. Click **"📥 Download Analysis Data"**
2. Save CSV file with Date, Price, and Entropy data
3. Use in your trading platform or further analysis

## 6. Asset Classes and Symbols

### Supported Asset Types

| Asset Class | Example Symbols | Characteristics |
|-------------|----------------|-----------------|
| **US Stocks** | AAPL, MSFT, GOOGL | Individual company analysis |
| **ETFs** | SPY, QQQ, IWM | Broad market exposure |
| **International** | EWJ, EWZ, FXI | Global market analysis |
| **Bonds** | TLT, IEF, HYG | Fixed income analysis |
| **Commodities** | GLD, SLV, USO | Commodity exposure |
| **Crypto** | BTC-USD, ETH-USD | Digital asset analysis |
| **Forex** | EURUSD=X, GBPUSD=X | Currency pairs |

### Recommended Test Symbols

- **SPY**: S&P 500 (stable, good for learning)
- **QQQ**: Nasdaq (more volatile, clear risk signals)
- **BTC-USD**: Bitcoin (highly volatile, dramatic entropy changes)
- **TLT**: Long-term bonds (different risk characteristics)

## 7. Advanced Configuration

### Optimizing Parameters

**Entropy Window (20-60 days):**
- **Shorter (20-30)**: More sensitive to recent changes
- **Longer (40-60)**: More stable, less noise

**Lookback Period (60-180 days):**
- **Shorter (60-90)**: Adapts quickly to new market conditions
- **Longer (120-180)**: More stable thresholds

**Base Kelly (1%-25%):**
- **Conservative (1-5%)**: Lower risk, steady growth
- **Moderate (5-10%)**: Balanced approach
- **Aggressive (10-25%)**: Higher risk, faster growth

### Market-Specific Settings

**Volatile Assets (Crypto, Small Caps):**
- Entropy Window: 20-25 days
- Base Kelly: 3-8%
- Lookback: 60-90 days

**Stable Assets (Large Cap, Bonds):**
- Entropy Window: 30-45 days
- Base Kelly: 5-15%
- Lookback: 90-180 days

## 8. Risk Management Guidelines

### Position Sizing Rules

1. **Never exceed recommended Kelly size**
2. **Reduce size during Risk-Off periods**
3. **Consider correlation between holdings**
4. **Monitor portfolio-level risk**

### Warning Signs

- **Persistent Risk-Off signals**: Consider reducing overall exposure
- **Extremely high entropy**: Market may be in transition
- **Conflicting signals**: Wait for clearer direction

### Best Practices

1. **Diversification**: Use across multiple assets
2. **Regular Review**: Check signals daily
3. **Backtesting**: Test on historical data before live trading
4. **Risk Limits**: Never risk more than you can afford to lose

## 9. Troubleshooting

### Common Issues

**"No data found for symbol":**
- Check symbol spelling and format
- Verify symbol exists on Yahoo Finance
- Try alternative ticker formats

**"Insufficient data for analysis":**
- Choose longer data period
- Reduce entropy window size
- Select more liquid assets

**Charts not updating:**
- Refresh browser page
- Check internet connection
- Try different browser

### Performance Tips

- **Use shorter data periods** for faster loading
- **Reduce entropy window** for quicker calculations
- **Choose liquid assets** for better data quality

## 10. Theoretical Background

### Topological Data Analysis

TDA is a branch of mathematics that studies the shape of data. In finance, it can detect:
- **Regime changes**: Market transitions between bull/bear markets
- **Volatility clusters**: Periods of high/low volatility
- **Structural breaks**: Fundamental changes in market dynamics

### Persistence Entropy

Measures the "information content" of topological features:
- **Low entropy**: Simple, predictable patterns
- **High entropy**: Complex, chaotic patterns
- **Entropy spikes**: Often precede market disruptions

### Kelly Criterion

Developed by John Kelly in 1956, it provides:
- **Optimal bet sizing**: Maximizes long-term growth
- **Risk management**: Prevents overleverage
- **Mathematical foundation**: Provably optimal under certain conditions

## 11. Limitations and Disclaimers

### Model Limitations

- **Past performance**: No guarantee of future results
- **Market conditions**: May not work in all market environments
- **Data quality**: Depends on reliable price data
- **Computational**: Entropy calculations can be noisy

### Risk Warnings

- **Trading involves risk**: You may lose money
- **No financial advice**: This tool is for educational purposes
- **Professional guidance**: Consult qualified financial advisors
- **Position sizing**: Consider your risk tolerance

### Technical Limitations

- **Internet dependency**: Requires connection for data
- **Processing time**: Complex calculations may take time
- **Browser compatibility**: Use modern browsers
- **Mobile experience**: Optimized for desktop use

## 12. Future Enhancements

### Planned Features

- **Multi-asset dashboard**: Compare multiple symbols
- **Email alerts**: Automated risk notifications
- **Backtesting module**: Historical performance analysis
- **API integration**: Connect to trading platforms
- **Mobile app**: Native mobile experience

### Contributing

This is an open-source project. Contributions welcome:
- **Bug reports**: Report issues on GitHub
- **Feature requests**: Suggest improvements
- **Code contributions**: Submit pull requests
- **Documentation**: Help improve this manual

## 13. Contact and Support

### Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Refer to this manual
- **Community**: Join discussions on GitHub

### Version Information

- **Current Version**: 1.0
- **Last Updated**: December 2024
- **Compatibility**: Python 3.8+, Streamlit 1.28+

---

**© 2024 TDA Risk Monitor. This software is provided as-is for educational purposes. Trading involves risk.**
