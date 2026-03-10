import streamlit as st
import requests
import pandas as pd
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 10

st.set_page_config(page_title="Trading Brain", layout="wide", initial_sidebar_state="expanded")

st.markdown("# 📈 Trading Brain", unsafe_allow_html=True)
st.markdown("**AI-Powered Manual & Automated Stock Trading Dashboard**", unsafe_allow_html=True)

# ============================================================================
# API Helper Functions
# ============================================================================

def make_api_request(method: str, endpoint: str, data=None, timeout=API_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Make HTTP request to API with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            return None
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Is it running on http://localhost:8000?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timeout")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        logger.error(f"API request error: {e}", exc_info=True)
        return None

@st.cache_data(ttl=300)
def get_stocks():
    """Get available stocks"""
    result = make_api_request("GET", "/stocks")
    return result.get("stocks", []) if result else []

def analyze_stock(stock, train_mode=False):
    """Analyze stock"""
    return make_api_request("POST", "/analyze", {"stock": stock, "train_mode": train_mode})

def get_portfolio_stats():
    """Get portfolio stats"""
    return make_api_request("GET", "/portfolio")

def get_trade_history():
    """Get trade history"""
    result = make_api_request("GET", "/trades")
    # Handle both formats: list or dict with 'trades' key
    if isinstance(result, dict) and "trades" in result:
        return result["trades"]
    return result if isinstance(result, list) else []

def get_model_accuracy():
    """Get model accuracy"""
    return make_api_request("GET", "/model/accuracy")

def start_automation(stocks: list, interval_minutes: int = 5):
    """Start automated trading for multiple stocks"""
    return make_api_request("POST", "/automation/start", {"stocks": stocks, "interval_minutes": interval_minutes})

def stop_automation():
    """Stop automated trading"""
    return make_api_request("POST", "/automation/stop")

def get_automation_status():
    """Get automation status"""
    return make_api_request("GET", "/automation/status")

def get_automation_history():
    """Get automation execution history"""
    result = make_api_request("GET", "/automation/history")
    return result.get("executions", []) if result else []

def clear_automation_history():
    """Clear automation execution history"""
    return make_api_request("POST", "/automation/clear")
    result = make_api_request("GET", "/automation/history")
    return result.get("trades", []) if result else []

# Check API
if make_api_request("GET", "/health") is None:
    st.error("❌ API Server not running. Start with: python api_server.py")
    st.stop()
# Get stocks
all_stocks = get_stocks()
if not all_stocks:
    st.error("Failed to load stocks")
    st.stop()

# Sidebar - Portfolio
with st.sidebar:
    st.markdown("## 💼 Portfolio Summary")
    stats = get_portfolio_stats()
    if stats:
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Cash", f"₹{stats['cash']:,.0f}")
            st.metric("📊 Position", f"{stats['position']:.2f}")
        with col2:
            pnl_color = "🟢" if stats['total_pnl'] >= 0 else "🔴"
            st.metric("💵 Total Value", f"₹{stats['total_value']:,.0f}")
            st.metric(f"{pnl_color} P&L", f"₹{stats['total_pnl']:,.0f}")
        st.divider()
    
    # Model Accuracy Section
    st.markdown("## 🧠 Model Performance")
    accuracy_data = get_model_accuracy()
    if accuracy_data:
        if accuracy_data.get("accuracy") is not None:
            accuracy = accuracy_data["accuracy"]
            precision = accuracy_data.get("precision")
            accuracy_pct = accuracy_data.get("accuracy_percentage", f"{accuracy*100:.2f}%")
            precision_pct = accuracy_data.get("precision_percentage", f"{precision*100:.2f}%" if precision else "N/A")
            
            # Display accuracy and precision with color coding
            col_acc, col_prec = st.columns(2)
            with col_acc:
                if accuracy >= 0.7:
                    st.success(f"✅ Accuracy: {accuracy_pct}", icon="✔️")
                elif accuracy >= 0.5:
                    st.info(f"ℹ️ Accuracy: {accuracy_pct}", icon="ℹ️")
                else:
                    st.warning(f"⚠️ Accuracy: {accuracy_pct}", icon="⚠️")
            
            with col_prec:
                if precision and precision >= 0.7:
                    st.success(f"📊 Precision: {precision_pct}", icon="✔️")
                elif precision and precision >= 0.5:
                    st.info(f"📊 Precision: {precision_pct}", icon="ℹ️")
                else:
                    st.warning(f"📊 Precision: {precision_pct}", icon="⚠️")
            
            # Confusion Matrix
            st.markdown("**Confusion Matrix:**")
            cm = accuracy_data.get("confusion_matrix")
            if cm:
                col_cm1, col_cm2 = st.columns(2)
                with col_cm1:
                    st.metric("✓ True Negatives", accuracy_data.get("tn", 0))
                    st.metric("✗ False Positives", accuracy_data.get("fp", 0))
                with col_cm2:
                    st.metric("✗ False Negatives", accuracy_data.get("fn", 0))
                    st.metric("✓ True Positives", accuracy_data.get("tp", 0))
        else:
            st.warning("⚠️ Model not trained yet")
            st.caption(accuracy_data.get("message", "Train the model first"))

with st.container():
    st.markdown("### 🔍 Stock Analysis")
    
    col_select, col_train = st.columns([3, 1])
    with col_select:
        stock = st.selectbox("Select Stock", all_stocks, label_visibility="collapsed")
    with col_train:
        train_mode = st.checkbox("🔄 Retrain", value=False, help="Train a new model on latest data")
    
    if st.button("🚀 Analyze & Execute", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            result = analyze_stock(stock, train_mode=train_mode)
        
        if result:
            # Analysis results in cards
            st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                signal_emoji = "🟢" if result["signal"] == "BUY" else "🔴"
                st.metric(f"{signal_emoji} Signal", result["signal"])
            with col2:
                st.metric("📊 RSI", f"{result['rsi']:.1f}")
            with col3:
                vol_emoji = "📉" if result['volatility'] < 0.05 else "📈"
                st.metric(f"{vol_emoji} Volatility", f"{result['volatility']:.4f}")
            with col4:
                st.metric("💹 Price", f"₹{result['current_price']:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # AI Decision
            st.divider() 
            st.markdown("### 🤖 AI Decision")
            decision_box = st.container(border=True)
            with decision_box:
                st.write(result["ai_decision"])
            
            # Trade Action
            if "BUY" in result["ai_decision"]:
                st.success(f"✅ **{result['trade_action']}**", icon="✔️")
            elif "SELL" in result["ai_decision"]:
                st.info(f"⚠️ **{result['trade_action']}**")
            else:
                st.warning("⏸️ **HOLD** - No trade signal", icon="⏸️")

# Trade History
st.markdown("---")
st.markdown("### 📋 Trade History")

trades = get_trade_history()
if trades and isinstance(trades, list) and len(trades) > 0:
    df_trades = pd.DataFrame(trades)
    st.dataframe(df_trades, use_container_width=True, hide_index=True)
else:
    st.info("💡 No trades executed yet. Select a stock and click 'Analyze & Execute' to start trading.")

# ============================================================================
# Automated Trading Section
# ============================================================================

st.markdown("---")
st.markdown("### 🤖 Automated Trading")

col_auto1, col_auto2 = st.columns([2, 1])

with col_auto1:
    st.markdown("**Configure and run automated trading**")
    
    col_stocks, col_interval = st.columns(2)
    
    with col_stocks:
        auto_stocks = st.multiselect("Select Stocks for Automation", all_stocks, label_visibility="collapsed", key="auto_stocks", default=[])
    
    with col_interval:
        auto_interval = st.number_input("Interval (minutes)", min_value=1, max_value=60, value=5, step=1, label_visibility="collapsed")
    
    col_start_btn, col_stop_btn = st.columns(2)
    
    with col_start_btn:
        if st.button("▶️ Start Automation", use_container_width=True, type="primary"):
            if auto_stocks:
                result = start_automation(auto_stocks, auto_interval)
                if result and "status" in result:
                    st.success(f"✅ {result['message']}")
                    st.rerun()
            else:
                st.error("Please select at least one stock")
    
    with col_stop_btn:
        if st.button("⏹️ Stop Automation", use_container_width=True):
            result = stop_automation()
            if result and "status" in result:
                st.info(f"ℹ️ {result['message']}")
                st.rerun()

with col_auto2:
    auto_status = get_automation_status()
    is_running = auto_status.get("running", False)
    
    if is_running:
        st.warning("🟢 RUNNING", icon="⚙️")
    else:
        st.info("⚪ IDLE", icon="⏸️")

# Automation Status Details
if auto_status and auto_status.get("enabled"):
    st.markdown("#### 📊 Automation Status")
    
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        stocks_str = ", ".join(auto_status.get("stocks", []))
        st.metric("📈 Stocks", stocks_str if stocks_str else "None")
    
    with col_status2:
        st.metric("⏱️ Interval", f"{auto_status.get('interval_minutes', 0)} min")
    
    with col_status3:
        st.metric("💱 Executions", auto_status.get("total_executions", 0))
    
    with col_status4:
        last_run = auto_status.get("last_run", "Never")
        if last_run and last_run != "Never":
            st.metric("🕐 Last Run", "Recently")
        else:
            st.metric("🕐 Last Run", "N/A")
    
    if auto_status.get("next_run"):
        st.caption(f"Next run: {auto_status.get('next_run')}")

# Automation Trade History
st.markdown("#### 📜 Automation Trade History")

col_hist1, col_hist2 = st.columns([3, 1])

with col_hist1:
    pass

with col_hist2:
    if st.button("🗑️ Clear History", use_container_width=True):
        clear_automation_history()
        st.success("History cleared")
        st.rerun()

auto_trades = get_automation_history()
if auto_trades and isinstance(auto_trades, list) and len(auto_trades) > 0:
    df_auto_trades = pd.DataFrame(auto_trades)
    # Reorder and format columns if they exist
    cols_to_show = ["timestamp", "stock", "signal", "current_price", "rsi", "volatility", "trade_action", "status"]
    cols_available = [col for col in cols_to_show if col in df_auto_trades.columns]
    st.dataframe(df_auto_trades[cols_available], use_container_width=True, hide_index=True)
else:
    st.info("💡 No automated trades yet. Start automation to begin.")
