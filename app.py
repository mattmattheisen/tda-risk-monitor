"""
TDA Risk Monitor
Refactored single-file version: Topological Data Analysis (persistence entropy)
+ Kelly-style dynamic risk dial.

If you later want to modularize, move everything below run_app() into a separate
module (see Option B in the README / instructions).
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Tuple, Dict

# ---- Optional TDA imports ----
try:
    import ripser
    TDA_AVAILABLE = True
except ImportError:
    TDA_AVAILABLE = False

# =========================================================
# -------------------- CONFIG CONSTANTS -------------------
# =========================================================
DEFAULT_ENTROPY_WINDOW = 30
DEFAULT_LOOKBACK = 90
DEFAULT_BASE_KELLY = 0.040
GREEN_Z = 0.5
RED_Z = 1.3
BANDS = [
    (0.30, "calm"),
    (0.80, "slightly elevated"),
    (1.30, "elevated"),
    (9.99, "stressed")
]

# =========================================================
# -------------------- CORE CALCULATIONS ------------------
# =========================================================
def calculate_persistence_entropy(price_window: np.ndarray) -> float:
    """Compute persistence entropy (dimension 0). Fallback if ripser missing."""
    if not TDA_AVAILABLE:
        return float(np.log1p(np.std(price_window)))
    try:
        arr = np.asarray(price_window, dtype=np.float64).flatten()
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return 0.0
        pts = arr.reshape(-1, 1)
        dgms0 = ripser.ripser(pts, maxdim=0)['dgms'][0]
        finite = dgms0[dgms0[:, 1] != np.inf]
        if finite.size == 0:
            return 0.0
        lifetimes = finite[:, 1] - finite[:, 0]
        total = lifetimes.sum()
        if total <= 0:
            return 0.0
        probs = lifetimes / total
        return float(-np.sum(probs * np.log(probs + 1e-12)))
    except Exception:
        return 0.0

def build_entropy_series(prices: pd.Series,
                         window: int) -> Tuple[List[pd.Timestamp], List[float]]:
    dates, ent_vals = [], []
    values = prices.values
    idx = prices.index
    for i in range(window, len(values)):
        sub = values[i - window: i]
        ent = calculate_persistence_entropy(sub)
        dates.append(idx[i])
        ent_vals.append(ent)
    return dates, ent_vals

def entropy_stats(entropy_values: List[float]) -> Dict[str, float]:
    arr = np.array(entropy_values)
    return {
        "current": float(arr[-1]),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0) if arr.size > 1 else 0.0001),
        "min": float(arr.min()),
        "max": float(arr.max())
    }

def z_score(current: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (current - mean) / std

def mood_from_z(z: float) -> str:
    if z < GREEN_Z:
        return "GREEN"
    if z < RED_Z:
        return "TRANSITIONAL"
    return "RISK-OFF"

def z_descriptor(z: float) -> str:
    for thresh, desc in BANDS:
        if z < thresh:
            return desc
    return "elevated"

def percentile_thresholds(entropy_values: List[float],
                          lookback: int) -> Tuple[float, float]:
    if len(entropy_values) < lookback:
        arr = np.array(entropy_values)
    else:
        arr = np.array(entropy_values[-lookback:])
    return (float(np.percentile(arr, 70)), float(np.percentile(arr, 90)))

def kelly_multiplier_from_entropy(current_entropy: float,
                                  entropy_values: List[float],
                                  lookback: int) -> Tuple[str, float]:
    lo, hi = percentile_thresholds(entropy_values, lookback)
    if current_entropy > hi:
        return "RISK-OFF", 0.35
    if current_entropy < lo:
        return "GREEN", 1.00
    return "TRANSITIONAL", 0.60

# =========================================================
# -------------------- DATA ACCESS LAYER ------------------
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_price_history(symbol: str, period: str = "2y") -> pd.Series:
    data = yf.download(symbol, period=period, progress=False)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        close_col = [c for c in data.columns if "Close" in c][0]
        return data[close_col].dropna()
    return data['Close'].dropna()

# =========================================================
# -------------------- UI HELPER COMPONENTS ---------------
# =========================================================
def status_badge(mood: str) -> str:
    colors = {"GREEN": "#2e7d32", "TRANSITIONAL": "#f9a825", "RISK-OFF": "#c62828"}
    return (
        f"<span style='background:{colors[mood]};color:white;"
        f"padding:6px 14px;border-radius:16px;font-weight:600;"
        f"letter-spacing:.5px;'>{mood}</span>"
    )

def action_for_mood(mood: str) -> str:
    return {
        "GREEN": "Trade full size; normal stops.",
        "TRANSITIONAL": "Trade reduced size; no pyramiding.",
        "RISK-OFF": "Defensive stance: minimal / no new positions."
    }[mood]

def noise_gauge(current: float, min_val: float, max_val: float):
    span = max_val - min_val
    pct = 0.5 if span <= 0 else (current - min_val) / span
    pct = max(0, min(1, pct))
    st.progress(pct)
    st.caption(f"Noise position: {pct*100:.0f}% of observed range")

def render_core_panel(entropy_values: List[float],
                      base_kelly: float,
                      multiplier: float,
                      equity: float,
                      risk_per_share: float,
                      view: str):
    stats = entropy_stats(entropy_values)
    current = stats["current"]
    mean = stats["mean"]
    std = stats["std"]
    z = z_score(current, mean, std)
    mood = mood_from_z(z)
    adj_kelly = base_kelly * multiplier
    dollar_risk = adj_kelly * equity
    shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

    st.markdown("### 📌 Market Overview")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Market Mood**")
        st.markdown(status_badge(mood), unsafe_allow_html=True)
        st.caption(action_for_mood(mood))

    with c2:
        st.markdown("**Noise Level**")
        st.metric("Current", f"{current:.3f}", f"{z:.2f}σ")
        noise_gauge(current, stats["min"], stats["max"])
        st.caption(f"Descriptor: **{z_descriptor(z)}**")

    with c3:
        st.markdown("**Risk Dial (Kelly)**")
        st.metric("Adjusted Kelly (risk % loss)", f"{adj_kelly:.3f}",
                  f"{multiplier:.2f}×")
        st.caption(f"≈ ${dollar_risk:,.0f} risk | {shares:,} shares "
                   f"(risk/share {risk_per_share:.2f})")

    st.markdown("---")
    st.markdown("### 🎯 Action")
    narrative = (
        f"Noise is **{z_descriptor(z)}** (z={z:.2f}). "
        f"Market Mood is {mood}. Risk Dial = {adj_kelly*100:.1f}% "
        f"(≈ ${dollar_risk:,.0f} potential loss at stop). "
        f"Recommendation: **{action_for_mood(mood)}**"
    )
    if mood == "GREEN":
        st.success(narrative)
    elif mood == "TRANSITIONAL":
        st.warning(narrative)
    else:
        st.error(narrative)

    st.markdown("### 💰 Position Sizing")
    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"""
**Account:** ${equity:,.0f}  
**Risk % (Adj Kelly):** {adj_kelly*100:.2f}%  
**Max Dollar Risk:** **${dollar_risk:,.0f}**  
""")
    with colB:
        if shares > 0:
            st.markdown(f"""
**Shares:** {shares:,}  
**Risk / Share:** ${risk_per_share:.2f}  
""")
        else:
            st.info("Enter a non-zero risk per share to compute shares.")

    if view != "Basic":
        st.markdown("---")
        if view == "Detail":
            st.markdown("### 📊 Detail Stats")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Mean", f"{mean:.3f}")
            d2.metric("Std Dev", f"{std:.3f}")
            d3.metric("Min", f"{stats['min']:.3f}")
            d4.metric("Max", f"{stats['max']:.3f}")
        elif view == "Advanced":
            st.markdown("### 🔬 Advanced Statistics")
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Mean", f"{mean:.3f}")
            d2.metric("Std Dev", f"{std:.3f}")
            d3.metric("Min", f"{stats['min']:.3f}")
            d4.metric("Max", f"{stats['max']:.3f}")
            d5.metric("Z-Score", f"{z:.2f}")
            st.caption("Color bands: Green <0.5σ, Transitional 0.5–1.3σ, Red >1.3σ")

def render_charts(prices: pd.Series,
                  dates: List[pd.Timestamp],
                  entropy_values: List[float],
                  lookback: int,
                  symbol: str):
    st.markdown("---")
    st.markdown("### 📈 Visuals")

    df = pd.DataFrame({
        "Date": dates,
        "Price": prices.iloc[len(prices) - len(dates):].values,
        "Entropy": entropy_values
    })

    tab_price, tab_entropy = st.tabs(["Price", "Noise History"])

    with tab_price:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Price"],
            name="Price", line=dict(color="blue", width=2)
        ))
        fig.update_layout(
            title=f"{symbol} Price",
            xaxis_title="Date", yaxis_title="Price ($)",
            height=400, margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_entropy:
        figN = go.Figure()
        figN.add_trace(go.Scatter(
            x=df["Date"], y=df["Entropy"],
            name="Noise Level", line=dict(color="red", width=2)
        ))
        if len(entropy_values) >= lookback:
            recent = entropy_values[-lookback:]
            hi = np.percentile(recent, 90)
            lo = np.percentile(recent, 70)
            figN.add_hline(y=hi, line_dash="dash", line_color="red",
                           annotation_text="90th (Risk-Off)")
            figN.add_hline(y=lo, line_dash="dash", line_color="green",
                           annotation_text="70th (Green)")
        figN.update_layout(
            title="Noise Level / Entropy",
            xaxis_title="Date", yaxis_title="Entropy",
            height=420, hovermode="x unified"
        )
        st.plotly_chart(figN, use_container_width=True)

        with st.expander("How to Read"):
            st.write("""
**Green Zone (<70th percentile)**: Smooth trends; normal risk.  
**Yellow Zone (70–90th)**: Choppier; reduce size.  
**Red Zone (>90th)**: Disorderly; defensive stance.
""")

def quick_reference_expander():
    with st.expander("📚 Quick Reference"):
        st.markdown("""
**Market Mood → Action**

| Mood | What it Means | Action |
|------|---------------|--------|
| GREEN | Quiet / orderly | Full size; normal stops |
| TRANSITIONAL | Mixed / choppy | Half size; no add-ons |
| RISK-OFF | Disorderly / whipsaw | Minimal or flat |

**Risk Dial (Adjusted Kelly)** = Fraction of account *at risk* (loss if stop hits).

**Entropy / Noise** = Shape-based measure of recent price complexity.
Higher → more volatility clusters → more whipsaw risk.
""")

# =========================================================
# -------------------- MAIN APP ---------------------------
# =========================================================
def run_app():
    st.set_page_config(page_title="TDA Risk Monitor",
                       page_icon="🧭",
                       layout="wide")

    st.title("🧭 TDA Risk Monitor")
    st.caption("Topological Data Analysis + Kelly Sizing")

    # Sidebar
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Symbol", value="SPY")
    equity = st.sidebar.number_input("Account Size ($)",
                                     min_value=1000, value=5000, step=500)
    risk_per_share = st.sidebar.number_input("Risk per Share ($)",
                                             min_value=0.01, value=1.00,
                                             step=0.05,
                                             help="Typically 1×ATR stop distance.")
    view = st.sidebar.radio("Detail Level",
                            ["Basic", "Detail", "Advanced"],
                            index=0, horizontal=True)
    with st.sidebar.expander("Advanced Parameters"):
        entropy_window = st.slider("Entropy Window (days)",
                                   20, 60, DEFAULT_ENTROPY_WINDOW, step=2)
        lookback = st.slider("Percentile Lookback (days)",
                             60, 180, DEFAULT_LOOKBACK, step=10)
        base_kelly = st.slider("Base Kelly (Normal Risk %)",
                               0.01, 0.20, DEFAULT_BASE_KELLY, step=0.005)
        data_period = st.selectbox("Data Period",
                                   ["6mo", "1y", "2y", "5y"], index=2)

    prices = fetch_price_history(symbol, period=data_period)
    if prices.empty:
        st.error("No price data returned. Try another symbol.")
        return

    needed = entropy_window + lookback
    if len(prices) < needed:
        st.warning(f"Not enough data (need ≥ {needed} days). "
                   f"Pick longer period or reduce lookback.")
        return

    with st.spinner("Computing entropy series..."):
        dates, ent_vals = build_entropy_series(prices, entropy_window)

    if not ent_vals:
        st.error("Failed to compute entropy values.")
        return

    current_entropy = ent_vals[-1]
    mood_pct, multiplier = kelly_multiplier_from_entropy(
        current_entropy, ent_vals, lookback
    )
    stats = entropy_stats(ent_vals)
    z = z_score(stats["current"], stats["mean"], stats["std"])
    mood_z = mood_from_z(z)
    mood_rank = {"GREEN": 0, "TRANSITIONAL": 1, "RISK-OFF": 2}
    final_mood = mood_z if mood_rank[mood_z] > mood_rank[mood_pct] else mood_pct
    if final_mood != mood_pct:
        if final_mood == "RISK-OFF" and multiplier > 0.40:
            multiplier = 0.35
        elif final_mood == "TRANSITIONAL" and multiplier > 0.60:
            multiplier = 0.60

    render_core_panel(ent_vals, base_kelly, multiplier,
                      equity, risk_per_share, view)
    render_charts(prices, dates, ent_vals, lookback, symbol)
    quick_reference_expander()

    if view == "Advanced":
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        out_df = pd.DataFrame({
            "Date": dates,
            "Price": prices.iloc[len(prices) - len(dates):].values,
            "Entropy": ent_vals
        })
        csv = out_df.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv,
                           file_name=f"{symbol}_tda_entropy.csv",
                           mime="text/csv")

if __name__ == "__main__":
    run_app()
