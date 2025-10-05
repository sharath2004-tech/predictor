# main.py - Main application file
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, date, datetime
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import httpx

# Import our custom modules
from login import show_login_page, is_authenticated, get_current_user, logout
from ml_predictor import show_ml_predictions, explain_ml_models, export_predictions
from services.ollama_client import ollama_chat, essential_system_prompt
from services.gemini_client import gemini_chat, DEFAULT_GEMINI_MODEL
from services.openrouter_client import openrouter_chat, DEFAULT_OPENROUTER_MODEL

load_dotenv()

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

# ========================
# App Configuration
# ========================
st.set_page_config(
    page_title="📊 Advanced Stock Predictor AI", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# Global Styles for Main App
# ========================
def apply_main_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Main background with animated gradient */
        .stApp {
            background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f0f23);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            font-family: 'Poppins', sans-serif;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Main title styling */
        .main-title {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            margin: 2rem 0;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        @keyframes glow {
            from { text-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
            to { text-shadow: 0 0 30px rgba(102, 126, 234, 0.8); }
        }
        
        /* Glass card effect */
        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            background: rgba(255, 255, 255, 0.12);
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        }
        
        /* Technical indicator headers */
        .tech-header {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 1.8rem;
            font-weight: 600;
            margin: 1.5rem 0;
            text-align: center;
        }
        
        /* Metrics styling */
        .metric-box {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            color: white;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .metric-box:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        /* Signal indicators */
        .signal-bullish {
            background: linear-gradient(135deg, #00f093, #00c9ff);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0, 240, 147, 0.4);
        }
        
        .signal-bearish {
            background: linear-gradient(135deg, #ff416c, #ff4757);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        }
        
        .signal-neutral {
            background: linear-gradient(135deg, #ffa726, #ffcc02);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            display: inline-block;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(255, 167, 38, 0.4);
        }
        
        /* Custom text color classes */
        .text-success { color: #00ff88 !important; }
        .text-warning { color: #ffa726 !important; }
        .text-danger { color: #ff4757 !important; }
        .text-info { color: #4facfe !important; }
        .text-purple { color: #667eea !important; }
        .text-cyan { color: #00f2fe !important; }
        .text-gold { color: #ffd700 !important; }
        .text-pink { color: #ff1493 !important; }
        
        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
    </style>
    """, unsafe_allow_html=True)


def _has_secret(var_name: str) -> bool:
    """Return True if the environment or Streamlit secrets provide a non-empty value."""
    try:
        if hasattr(st, "secrets") and st.secrets.get(var_name):
            return True
    except Exception:
        pass

    env_val = os.getenv(var_name)
    if env_val:
        return True

    # Fallback: read directly from a sibling .env file to support hot reloads
    try:
        env_path = Path(__file__).resolve().parent / ".env"
        if env_path.exists():
            values = dotenv_values(env_path)
            candidate = values.get(var_name)
            if candidate:
                # Cache for subsequent checks without re-reading the file
                os.environ.setdefault(var_name, candidate)
                return True
    except Exception:
        pass

    return False

# ========================
# Technical Indicators
# ========================
def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    try:
        # Handle different data types and ensure 1D
        if isinstance(prices, pd.DataFrame):
            if len(prices.columns) == 1:
                prices = prices.iloc[:, 0]
            else:
                prices = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        elif isinstance(prices, np.ndarray):
            if prices.ndim > 1:
                prices = prices.flatten()
        elif hasattr(prices, 'values'):
            prices = prices.values
            if isinstance(prices, np.ndarray) and prices.ndim > 1:
                prices = prices.flatten()
        
        # Convert to pandas Series
        prices = pd.Series(prices).astype(float)
        prices = prices.dropna().reset_index(drop=True)
        
        if len(prices) < window:
            return pd.Series([50] * len(prices))
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        # Avoid division by zero
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    except Exception as e:
        st.error(f"RSI calculation error: {str(e)}")
        # Return a safe default
        try:
            length = len(prices) if hasattr(prices, '__len__') else 100
        except:
            length = 100
        return pd.Series([50] * length)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    try:
        # Handle different data types and ensure 1D
        if isinstance(prices, pd.DataFrame):
            if len(prices.columns) == 1:
                prices = prices.iloc[:, 0]
            else:
                prices = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        elif isinstance(prices, np.ndarray):
            if prices.ndim > 1:
                prices = prices.flatten()
        elif hasattr(prices, 'values'):
            prices = prices.values
            if isinstance(prices, np.ndarray) and prices.ndim > 1:
                prices = prices.flatten()
        
        # Convert to pandas Series
        prices = pd.Series(prices).astype(float)
        prices = prices.dropna().reset_index(drop=True)
        
        if len(prices) < max(fast, slow):
            zeros = pd.Series([0] * len(prices))
            return zeros, zeros, zeros
        
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)
    except Exception as e:
        st.error(f"MACD calculation error: {str(e)}")
        try:
            length = len(prices) if hasattr(prices, '__len__') else 100
        except:
            length = 100
        zeros = pd.Series([0] * length)
        return zeros, zeros, zeros

def calculate_moving_averages(prices):
    """Calculate moving averages"""
    try:
        # Handle different data types and ensure 1D
        if isinstance(prices, pd.DataFrame):
            if len(prices.columns) == 1:
                prices = prices.iloc[:, 0]
            else:
                prices = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        elif isinstance(prices, np.ndarray):
            if prices.ndim > 1:
                prices = prices.flatten()
        elif hasattr(prices, 'values'):
            prices = prices.values
            if isinstance(prices, np.ndarray) and prices.ndim > 1:
                prices = prices.flatten()
        
        # Convert to pandas Series
        prices = pd.Series(prices).astype(float)
        prices = prices.dropna().reset_index(drop=True)
        
        ma_5 = prices.rolling(window=5, min_periods=1).mean()
        ma_10 = prices.rolling(window=10, min_periods=1).mean()
        ma_20 = prices.rolling(window=20, min_periods=1).mean()
        ma_50 = prices.rolling(window=50, min_periods=1).mean()
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        
        return ma_5, ma_10, ma_20, ma_50, ema_12, ema_26
    except Exception as e:
        st.error(f"Moving averages calculation error: {str(e)}")
        try:
            length = len(prices) if hasattr(prices, '__len__') else 100
        except:
            length = 100
        prices_series = pd.Series([0] * length)
        return [prices_series] * 6

# ========================
# Data Functions
# ========================
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data(ticker, period="1y"):
    """Fetch stock data with caching"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        data.reset_index(inplace=True)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_market_overview(tickers):
    """Get market overview data"""
    overview_data = []
    for ticker in tickers[:6]:  # Limit to 6 stocks for performance
        try:
            stock_data = yf.download(ticker, period="2d", progress=False)
            if len(stock_data) >= 2:
                current_val = stock_data["Close"].iloc[-1]
                previous_val = stock_data["Close"].iloc[-2]
                volume_val = stock_data["Volume"].iloc[-1]
                
                # Ensure scalar values
                if hasattr(current_val, 'item'):
                    current_val = current_val.item()
                if hasattr(previous_val, 'item'):
                    previous_val = previous_val.item()
                if hasattr(volume_val, 'item'):
                    volume_val = volume_val.item()
                
                current = float(current_val)
                previous = float(previous_val)
                volume = int(volume_val)
                change = ((current - previous) / previous) * 100
                
                # Pre-format strings to avoid Series formatting issues
                price_str = f"₹{current:.2f}"
                change_str = f"{change:+.2f}%"
                volume_str = f"{volume:,}"
                
                overview_data.append({
                    "Stock": ticker.replace(".NS", ""),
                    "Price": price_str,
                    "Change": change_str,
                    "Volume": volume_str,
                    "Change_Val": change
                })
        except:
            continue
    return pd.DataFrame(overview_data)


@st.cache_data(show_spinner=False)
def search_tickers(query: str, limit: int = 12):
    """Search Yahoo Finance for symbols matching the query."""
    if not query or len(query) < 2:
        return []

    params = {
        "q": query,
        "quotesCount": limit,
        "newsCount": 0,
        "enableFuzzyQuery": False,
        "quotesQueryId": "tss_match_phrase_query",
        "multiQuoteQueryId": "multi_quote_single_token_query",
        "enableNavLinks": False,
        "enableEnhancedTrivialQuery": True,
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(YAHOO_SEARCH_URL, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json() or {}
    except Exception:
        return []

    cleaned = []
    for item in data.get("quotes", []) or []:
        symbol = item.get("symbol")
        if not symbol:
            continue
        name = item.get("shortname") or item.get("longname") or item.get("quoteType") or ""
        exchange = item.get("exchangeDisplay") or item.get("exchDisp") or item.get("exchange") or ""
        cleaned.append({
            "symbol": symbol,
            "name": name,
            "exchange": exchange,
        })
    return cleaned


def _prepare_ai_context(stock: str, data: pd.DataFrame) -> str:
    """Assemble a concise context block for AI providers."""
    if data.empty:
        return f"Stock: {stock}\nNo price data available."

    latest = data.iloc[-1]
    context_lines = [f"Stock: {stock}"]

    try:
        last_close = float(latest.get("Close"))
        context_lines.append(f"Latest close: ₹{last_close:.2f}")
    except Exception:
        pass

    try:
        prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else None
        if prev_close:
            change_pct = ((last_close - prev_close) / prev_close) * 100
            context_lines.append(f"Daily change: {change_pct:+.2f}%")
    except Exception:
        pass

    try:
        high_52 = float(data["Close"].max())
        low_52 = float(data["Close"].min())
        context_lines.append(f"52-week range: ₹{low_52:.2f} - ₹{high_52:.2f}")
    except Exception:
        pass

    try:
        volume = int(latest.get("Volume"))
        context_lines.append(f"Latest volume: {volume:,}")
    except Exception:
        pass

    ml_ctx = st.session_state.get("ml_context")
    if ml_ctx:
        best_model = ml_ctx.get("best_model")
        final_preds = ml_ctx.get("final_predictions") or {}
        target_date = ml_ctx.get("target_date")
        if final_preds:
            summary_parts = []
            for name, price in final_preds.items():
                summary_parts.append(f"{name}: ₹{price:.2f}")
            context_lines.append("AI predictions: " + ", ".join(summary_parts))
        if best_model:
            context_lines.append(f"Best model: {best_model} (confidence {ml_ctx.get('best_confidence', 0):.1f}%)")
        if target_date:
            context_lines.append(f"Prediction target date: {target_date}")

    return "\n".join(context_lines)


def build_trend_snapshot(tickers):
    """Generate a lightweight market snapshot used by the FastAPI service."""
    results = []
    gainers = []
    losers = []

    for symbol in tickers:
        data = get_stock_data(symbol, period="6mo")
        if data.empty:
            continue

        close_series = data["Close"].astype(float)
        try:
            last_price = float(close_series.iloc[-1])
        except Exception:
            continue

        def _pct_change(periods: int):
            if len(close_series) <= periods:
                return None
            try:
                prev = float(close_series.iloc[-(periods + 1)])
                curr = float(close_series.iloc[-1])
                if prev == 0:
                    return None
                return ((curr - prev) / prev) * 100
            except Exception:
                return None

        d1 = _pct_change(1)
        d5 = _pct_change(5)
        d20 = _pct_change(20)

        rsi_series = calculate_rsi(close_series, window=14)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        _, _, ma20, ma50, _, _ = calculate_moving_averages(close_series)
        try:
            ma20_last = float(ma20.iloc[-1])
            ma50_last = float(ma50.iloc[-1])
            if last_price > ma20_last and ma20_last > ma50_last:
                ma_state = "Bullish"
            elif last_price < ma20_last and ma20_last < ma50_last:
                ma_state = "Bearish"
            else:
                ma_state = "Neutral"
        except Exception:
            ma_state = "Neutral"

        try:
            vol_series = data["Volume"].astype(float)
            vol_ratio = float(vol_series.iloc[-1]) / float(vol_series.tail(20).mean()) if len(vol_series) >= 20 else None
        except Exception:
            vol_ratio = None

        last_date = data["Date"].iloc[-1]
        try:
            if isinstance(last_date, str):
                last_dt = datetime.fromisoformat(last_date)
            else:
                last_dt = pd.to_datetime(last_date).to_pydatetime()
        except Exception:
            last_dt = datetime.utcnow()
        stale = (datetime.utcnow() - last_dt).days > 2

        record = {
            "symbol": symbol,
            "last": last_price,
            "d1": d1,
            "d5": d5,
            "d20": d20,
            "rsi14": rsi_val,
            "ma_state": ma_state,
            "vol_ratio": vol_ratio,
            "stale": stale,
        }

        results.append(record)
        if d1 is not None:
            gainers.append(record)
            losers.append(record)

    gainers_sorted = sorted(gainers, key=lambda x: x.get("d1") or -999, reverse=True)[:3]
    losers_sorted = sorted(losers, key=lambda x: x.get("d1") or 999)[:3]

    summary_lines = []
    if gainers_sorted:
        summary_lines.append("Top gainers: " + ", ".join(f"{g['symbol']} {g['d1']:+.2f}%" for g in gainers_sorted if g.get("d1") is not None))
    if losers_sorted:
        summary_lines.append("Top losers: " + ", ".join(f"{l['symbol']} {l['d1']:+.2f}%" for l in losers_sorted if l.get("d1") is not None))

    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary_lines": summary_lines,
        "tickers": results,
    }


def show_ai_insights(stock: str, data: pd.DataFrame):
    """Render the AI assistant section allowing users to query different providers."""
    st.markdown("---")
    st.markdown('<h2 class="tech-header">🧠 AI Market Insights</h2>', unsafe_allow_html=True)

    providers = ["Local Ollama", "OpenRouter", "Google Gemini"]
    provider = st.selectbox("Model Provider", providers, key="ai_provider_select")

    if provider == "Local Ollama":
        models = ["mistral", "llama3", "nvidia/nemotron-nano-9b-v2:free"]
        default_model = 0
        helper_text = "Runs locally via Ollama. Ensure `ollama serve` is active."
        key_available = True
    elif provider == "OpenRouter":
        models = [
            "google/gemma-3-4b-it:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "anthropic/claude-3-haiku:beta",
        ]
        default_model = 0
        helper_text = "Hosted models via OpenRouter. Requires OPENROUTER_API_KEY."
        key_available = _has_secret("OPENROUTER_API_KEY")
        if not key_available:
            st.warning("⚠️ Add OPENROUTER_API_KEY to your .env or Streamlit secrets to use hosted models.")
    else:
        models = [DEFAULT_GEMINI_MODEL, "gemini-1.5-pro-latest", "learnlm-1.5-pro-experimental"]
        default_model = 0
        helper_text = "Uses Google Gemini API. Requires GEMINI_API_KEY."
        key_available = _has_secret("GEMINI_API_KEY")
        if not key_available:
            st.warning("⚠️ Add GEMINI_API_KEY to your .env or Streamlit secrets to enable Gemini access.")

    model_choice = st.selectbox("Model", models, index=default_model, key="ai_model_select", help=helper_text)

    if "ai_history" not in st.session_state:
        st.session_state["ai_history"] = []

    question = st.text_area(
        "Ask a question about the current stock",
        value="Summarize the current trend and highlight key risks.",
        height=120,
        key="ai_question_text",
    )

    col_ask, col_clear = st.columns([3, 1])
    ask_clicked = col_ask.button("🤖 Ask AI", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("🗑️ Clear Chat", use_container_width=True)

    if clear_clicked:
        st.session_state["ai_history"] = []

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question for the AI assistant.")
        elif provider != "Local Ollama" and not key_available:
            st.error("Missing API key for the selected provider. Update your configuration and try again.")
        else:
            context = _prepare_ai_context(stock, data)
            user_prompt = f"{context}\n\nUser Question:\n{question.strip()}"
            messages = [
                {
                    "role": "system",
                    "content": essential_system_prompt + " Respond with clear bullet points. Always include a short disclaimer.",
                },
                {"role": "user", "content": user_prompt},
            ]

            with st.spinner("Contacting AI model..."):
                if provider == "Local Ollama":
                    answer = ollama_chat(
                        messages,
                        model=model_choice,
                        host=os.getenv("OLLAMA_HOST"),
                        timeout=90.0,
                        temperature=0.2,
                    )
                elif provider == "OpenRouter":
                    answer = openrouter_chat(
                        messages,
                        model=model_choice,
                        temperature=0.2,
                        timeout=90.0,
                    )
                else:
                    answer = gemini_chat(messages, model=model_choice, timeout=90.0)

            st.session_state["ai_history"].insert(0, {"question": question.strip(), "answer": answer, "provider": provider, "model": model_choice})
            st.session_state["ai_history"] = st.session_state["ai_history"][:6]

    if st.session_state["ai_history"]:
        st.markdown("### Recent Conversations")
        for exchange in st.session_state["ai_history"]:
            with st.chat_message("user"):
                st.markdown(exchange["question"])
            with st.chat_message("assistant"):
                answer_text = exchange["answer"]
                if answer_text.startswith("AI error"):
                    st.error(answer_text)
                else:
                    st.markdown(answer_text)

# ========================
# Main Application
# ========================
def main_app():
    """Main application dashboard"""
    
    # Apply styles
    apply_main_styles()
    
    # Get current user
    user_info = get_current_user()
    
    # Header
    st.markdown('<h1 class="main-title">📊 Advanced Stock Market Predictor</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #8892b0; font-size: 1.2rem;">Welcome back, {user_info["email"]}! 🚀</p>', unsafe_allow_html=True)
    
    # Penny stocks list
    PENNY_STOCKS = [
        "YESBANK.NS", "SUZLON.NS", "PNB.NS", "IDEA.NS", "RPOWER.NS",
        "JPPOWER.NS", "IRFC.NS", "ONGC.NS", "IOB.NS", "TATAPOWER.NS"
    ]
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎛️ Control Panel")
        
        # User info
        user_type = "🎯 Demo User" if user_info["type"] == "demo" else "👑 Registered User"
        st.markdown(f"**Status:** {user_type}")
        
        st.markdown("---")
        
        # Stock selection
        stock = st.selectbox(
            "📈 Select Stock",
            PENNY_STOCKS,
            help="Choose a penny stock to analyze"
        )

        search_symbol = None
        search_query = st.text_input(
            "🔍 Search any global ticker",
            help="Type at least 2 characters to search Yahoo Finance. Example: RELIANCE.NS or TSLA",
            key="ticker_search_input"
        )

        cleaned_query = search_query.strip()
        if len(cleaned_query) >= 2:
            matches = search_tickers(cleaned_query, limit=15)
            if matches:
                options = []
                for item in matches:
                    name = item["name"].strip() if item.get("name") else ""
                    exchange = item.get("exchange") or ""
                    label_parts = [item["symbol"]]
                    if name:
                        label_parts.append(f"{name}")
                    if exchange:
                        label_parts.append(f"[{exchange}]")
                    options.append(" · ".join(label_parts))

                selected_label = st.selectbox(
                    "Suggested matches",
                    options,
                    key="_search_result_select"
                )
                try:
                    selected_index = options.index(selected_label)
                    search_symbol = matches[selected_index]["symbol"]
                except ValueError:
                    search_symbol = matches[0]["symbol"]
            else:
                st.info("No matches found. Showing default list above.")

        if search_symbol:
            stock = search_symbol
            st.markdown(
                f"<div class='glass-card' style='margin-top: 0.5rem; text-align: center;'>Using search result: <strong>{stock}</strong></div>",
                unsafe_allow_html=True,
            )
        
        # Analysis options
        st.markdown("### 📊 Analysis Options")
        show_technical = st.checkbox("Technical Indicators", value=True)
        show_ml = st.checkbox("AI Predictions", value=True)
        show_overview = st.checkbox("Market Overview", value=True)
        show_ai_chat = st.checkbox("AI Assistant", value=True)
        
        # Technical settings
        st.markdown("### ⚙️ Technical Settings")
        rsi_period = st.slider("RSI Period", 5, 30, 14)
        chart_period = st.selectbox("Chart Period", ["3mo", "6mo", "1y", "2y"], index=2)
        
        # Prediction settings with date picker
        st.markdown("### 🔮 Prediction Settings")
        
        # Date picker for prediction target
        today = date.today()
        max_prediction_date = today + timedelta(days=365)
        
        prediction_date = st.date_input(
            "📅 Predict Price For Date",
            value=today + timedelta(days=7),
            min_value=today + timedelta(days=1),
            max_value=max_prediction_date,
            help="Select the date you want to predict the stock price for"
        )
        
        # Calculate days ahead
        days_ahead = (prediction_date - today).days
        
        # Display prediction info
        st.info(f"🎯 Predicting price for **{days_ahead} days ahead**\n\n📅 Target Date: **{prediction_date.strftime('%B %d, %Y')}**")
        
        # Additional ML settings
        with st.expander("🤖 AI Model Settings"):
            model_confidence = st.slider("Minimum Model Confidence (%)", 50, 95, 75)
            include_weekends = st.checkbox("Include Weekends in Prediction", False, 
                                         help="Include weekend days in prediction calculation")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Main content
    try:
        # Get stock data
        with st.spinner(f"📊 Loading {stock} data..."):
            data = get_stock_data(stock, period=chart_period)
        
        if data.empty:
            st.error(f"❌ Unable to fetch data for {stock}. Please try another stock.")
            return
        
        # Current metrics
        current_price_val = data["Close"].iloc[-1]
        prev_price_val = data["Close"].iloc[-2] if len(data) > 1 else current_price_val
        volume_val = data["Volume"].iloc[-1]
        high_52w_val = data["Close"].max()
        low_52w_val = data["Close"].min()
        
        # Ensure scalar values
        if hasattr(current_price_val, 'item'):
            current_price_val = current_price_val.item()
        if hasattr(prev_price_val, 'item'):
            prev_price_val = prev_price_val.item()
        if hasattr(volume_val, 'item'):
            volume_val = volume_val.item()
        if hasattr(high_52w_val, 'item'):
            high_52w_val = high_52w_val.item()
        if hasattr(low_52w_val, 'item'):
            low_52w_val = low_52w_val.item()
        
        current_price = float(current_price_val)
        prev_price = float(prev_price_val)
        volume = int(volume_val)
        high_52w = float(high_52w_val)
        low_52w = float(low_52w_val)
        change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "#00ff88" if change_pct >= 0 else "#ff4757"
            arrow = "📈" if change_pct >= 0 else "📉"
            st.markdown(f'''
            <div class="metric-box">
                <h3>💰 Current Price</h3>
                <h2>₹{current_price:.2f}</h2>
                <p style="color: {color};">{arrow} {change_pct:+.2f}%</p>
            </div>
            ''', unsafe_allow_html=True)

        with col2:
            st.markdown(f'''
            <div class="metric-box">
                <h3>📈 52W High</h3>
                <h2>₹{high_52w:.2f}</h2>
                <p style="color: #8892b0;">Maximum this year</p>
            </div>
            ''', unsafe_allow_html=True)

        with col3:
            st.markdown(f'''
            <div class="metric-box">
                <h3>📉 52W Low</h3>
                <h2>₹{low_52w:.2f}</h2>
                <p style="color: #8892b0;">Minimum this year</p>
            </div>
            ''', unsafe_allow_html=True)

        with col4:
            st.markdown(f'''
            <div class="metric-box">
                <h3>📊 Volume</h3>
                <h2>{volume:,}</h2>
                <p style="color: #8892b0;">Today's trading</p>
            </div>
            ''', unsafe_allow_html=True)

        # Technical Analysis Section
        if show_technical:
            st.markdown("---")
            st.markdown('<h2 class="tech-header">🔍 Technical Analysis Dashboard</h2>', unsafe_allow_html=True)
            
            # Calculate indicators
            try:
                rsi = calculate_rsi(data["Close"], rsi_period)
                macd_line, signal_line, histogram = calculate_macd(data["Close"])
                ma_5, ma_10, ma_20, ma_50, ema_12, ema_26 = calculate_moving_averages(data["Close"])
            except Exception as e:
                st.error(f"Error calculating technical indicators: {str(e)}")
                return

            # Price chart with moving averages (candlestick overlay)
            fig_price = go.Figure()

            # Add moving averages first
            colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#feca57"]
            mas = [("MA5", ma_5), ("MA20", ma_20), ("MA50", ma_50), ("EMA12", ema_12), ("EMA26", ema_26)]
            
            for i, (name, ma_data) in enumerate(mas):
                if not ma_data.empty:
                    fig_price.add_trace(go.Scatter(
                        x=data["Date"],
                        y=ma_data,
                        name=name,
                        line=dict(color=colors[i % len(colors)], width=1.6),
                        opacity=0.6,
                        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra>" + name + "</extra>",
                    ))

            def _column_to_series(column):
                if isinstance(column, pd.DataFrame):
                    column = column.iloc[:, 0]
                return pd.to_numeric(column, errors="coerce")

            open_series = _column_to_series(data["Open"])
            high_series = _column_to_series(data["High"])
            low_series = _column_to_series(data["Low"])
            close_series = _column_to_series(data["Close"])

            hover_text = [
                f"Open: ₹{o:.2f}<br>High: ₹{h:.2f}<br>Low: ₹{l:.2f}<br>Close: ₹{c:.2f}"
                for o, h, l, c in zip(open_series, high_series, low_series, close_series)
            ]

            # Candlestick chart (added last so it sits on top)
            fig_price.add_trace(go.Candlestick(
                x=data["Date"],
                open=open_series,
                high=high_series,
                low=low_series,
                close=close_series,
                name="Price 🕯️",
                increasing=dict(
                    line=dict(color="#00ff88", width=1.8),
                    fillcolor="rgba(0, 255, 136, 0.35)",
                ),
                decreasing=dict(
                    line=dict(color="#ff4757", width=1.8),
                    fillcolor="rgba(255, 71, 87, 0.35)",
                ),
                whiskerwidth=0.3,
                opacity=0.92,
                hoverinfo="x+text+name",
                hovertext=hover_text,
            ))
            
            fig_price.update_layout(
                title=f"📊 {stock} - Price Chart with Moving Averages",
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1a1a2e"),
                legend=dict(traceorder="reversed"),
            )
            fig_price.update_xaxes(type="date")
            
            st.plotly_chart(fig_price, config=PLOTLY_CONFIG)
            
            # RSI and MACD Analysis
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # RSI Chart
                fig_rsi = go.Figure()
                
                fig_rsi.add_trace(go.Scatter(
                    x=data["Date"], 
                    y=rsi, 
                    name="RSI", 
                    line=dict(color="#e74c3c", width=3),
                    fill='tonexty'
                ))
                
                # RSI levels
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff4757", 
                                annotation_text="Overbought (70)", annotation_position="top right")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00ff88", 
                                annotation_text="Oversold (30)", annotation_position="bottom right")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="#8892b0", 
                                annotation_text="Neutral (50)")
                
                fig_rsi.update_layout(
                    title=f"📈 RSI ({rsi_period} Period) - Momentum Indicator",
                    template="plotly_dark",
                    height=400,
                    yaxis=dict(range=[0, 100]),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Date",
                    yaxis_title="RSI"
                )
                st.plotly_chart(fig_rsi, config=PLOTLY_CONFIG)
            
            with col2:
                # RSI Signal Analysis
                try:
                    current_rsi = float(rsi.iloc[-1])
                    if pd.isna(current_rsi):
                        current_rsi = 50
                except:
                    current_rsi = 50
                
                if current_rsi > 70:
                    signal_class = "signal-bearish"
                    signal_text = "🔴 OVERBOUGHT"
                    signal_desc = "Consider selling - stock may be overvalued"
                    recommendation = "SELL"
                elif current_rsi < 30:
                    signal_class = "signal-bullish"
                    signal_text = "🟢 OVERSOLD"
                    signal_desc = "Consider buying - stock may be undervalued"
                    recommendation = "BUY"
                else:
                    signal_class = "signal-neutral"
                    signal_text = "🟡 NEUTRAL"
                    signal_desc = "Hold position - wait for clearer signals"
                    recommendation = "HOLD"
                
                st.markdown(f'''
                <div class="glass-card" style="text-align: center;">
                    <h3>🎯 RSI Signal</h3>
                    <h1 style="margin: 1rem 0;">{current_rsi:.1f}</h1>
                    <div class="{signal_class}" style="margin: 1rem 0;">{signal_text}</div>
                    <h4 style="color: #4facfe; margin: 1rem 0;">{recommendation}</h4>
                    <p style="color: #8892b0; font-size: 0.9rem;">{signal_desc}</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Additional technical info
                try:
                    # Safely calculate volume comparison
                    volume_mean_val = data['Volume'].mean()
                    if hasattr(volume_mean_val, 'item'):
                        volume_mean_val = volume_mean_val.item()
                    volume_mean = float(volume_mean_val)
                    
                    # Safely calculate price difference from MA20
                    ma_20_last_val = ma_20.iloc[-1]
                    if hasattr(ma_20_last_val, 'item'):
                        ma_20_last_val = ma_20_last_val.item()
                    ma_20_last = float(ma_20_last_val)
                    
                    # Calculate price difference percentage
                    price_diff_pct = ((current_price - ma_20_last) / ma_20_last * 100)
                    
                    # Calculate volatility
                    volatility_val = ((data['High'] - data['Low']) / data['Close'] * 100).iloc[-1]
                    if hasattr(volatility_val, 'item'):
                        volatility_val = volatility_val.item()
                    volatility = float(volatility_val)
                    
                    volume_trend = "📈 High" if volume > volume_mean else "📉 Low"
                    
                except Exception as e:
                    # Fallback values
                    volatility = 0.0
                    price_diff_pct = 0.0
                    volume_trend = "📊 Normal"
                
                st.markdown(f'''
                <div class="glass-card" style="text-align: center; margin-top: 1rem;">
                    <h4>📊 Quick Stats</h4>
                    <p><strong>Volatility:</strong> {volatility:.1f}%</p>
                    <p><strong>Volume Trend:</strong> {volume_trend}</p>
                    <p><strong>Price from MA20:</strong> {price_diff_pct:.1f}%</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # MACD Chart
            fig_macd = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('MACD Line & Signal Line', 'MACD Histogram'),
                row_heights=[0.7, 0.3]
            )
            
            # MACD lines
            fig_macd.add_trace(go.Scatter(
                x=data["Date"], y=macd_line, 
                name="MACD", line=dict(color="#00f2fe", width=2)
            ), row=1, col=1)
            
            fig_macd.add_trace(go.Scatter(
                x=data["Date"], y=signal_line, 
                name="Signal", line=dict(color="#ff6b6b", width=2)
            ), row=1, col=1)
            
            # MACD histogram
            try:
                histogram_values = [float(h) if not pd.isna(h) else 0 for h in histogram]
                colors = ['#00ff88' if h >= 0 else '#ff4757' for h in histogram_values]
                
                fig_macd.add_trace(go.Bar(
                    x=data["Date"], y=histogram_values, 
                    name="Histogram", marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)
            except:
                pass
            
            fig_macd.update_layout(
                title="📊 MACD - Trend Following Momentum Indicator",
                template="plotly_dark",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_macd, config=PLOTLY_CONFIG)

        # AI/ML Predictions Section
        if show_ml:
            st.markdown("---")
            
            # Pass the prediction parameters to the ML function
            show_ml_predictions(data, stock, days_ahead, prediction_date, model_confidence)

        if show_ai_chat:
            show_ai_insights(stock, data)

        # Market Overview Section
        if show_overview:
            st.markdown("---")
            st.markdown('<h2 class="tech-header">📊 Market Overview</h2>', unsafe_allow_html=True)
            
            try:
                overview_df = get_market_overview(PENNY_STOCKS)
                
                if not overview_df.empty:
                    # Create market overview cards
                    cols = st.columns(3)
                    
                    for idx, row in overview_df.iterrows():
                        col_idx = idx % 3
                        color = "#00ff88" if row["Change_Val"] >= 0 else "#ff4757"
                        arrow = "📈" if row["Change_Val"] >= 0 else "📉"
                        
                        with cols[col_idx]:
                            st.markdown(f'''
                            <div class="glass-card" style="text-align: center;">
                                <h3>{arrow} {row["Stock"]}</h3>
                                <h2>{row["Price"]}</h2>
                                <p style="color: {color}; font-weight: 600; font-size: 1.1rem;">{row["Change"]}</p>
                                <p style="color: #8892b0; font-size: 0.9rem;">Volume: {row["Volume"]}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                else:
                    st.info("📊 Market overview data temporarily unavailable")
            except Exception:
                st.info("📊 Market overview temporarily unavailable")

        # Educational Section
        st.markdown("---")
        st.markdown('<h2 class="tech-header">🎓 Learn About Technical Analysis</h2>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📈 RSI", "📊 MACD", "🤖 AI Models"])
        
        with tab1:
            st.markdown("""
            ### 📈 Relative Strength Index (RSI)
            
            **What it measures:** Price momentum and overbought/oversold conditions
            
            **How to read it:**
            - **RSI > 70**: Stock is overbought (potentially overvalued)
            - **RSI < 30**: Stock is oversold (potentially undervalued)
            - **RSI around 50**: Neutral momentum
            
            **Trading Strategy:**
            - Look for RSI divergences with price
            - Use 70/30 levels for entry/exit signals
            - Combine with other indicators for confirmation
            """)
        
        with tab2:
            st.markdown("""
            ### 📊 MACD (Moving Average Convergence Divergence)
            
            **What it measures:** Trend changes and momentum
            
            **Components:**
            - **MACD Line**: 12-period EMA minus 26-period EMA
            - **Signal Line**: 9-period EMA of MACD line
            - **Histogram**: MACD line minus Signal line
            
            **Trading Signals:**
            - **Bullish**: MACD crosses above Signal line
            - **Bearish**: MACD crosses below Signal line
            - **Divergence**: Price and MACD move in opposite directions
            """)
        
        with tab3:
            explain_ml_models()

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #8892b0; padding: 2rem;">
            <p style="font-size: 1.1rem;">🚀 Powered by Advanced AI & Real-time Market Data</p>
            <p style="font-size: 0.9rem;">Built with Python • Streamlit • Machine Learning • Real-time APIs</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">
                ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Always do your own research 
                and consult with a financial advisor before making investment decisions. Past performance does not 
                guarantee future results.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try refreshing the page or selecting a different stock.")

# ========================
# Main Application Entry Point
# ========================
def main():
    """Main application entry point"""
    
    # Check authentication
    if not is_authenticated():
        # Show login page
        if show_login_page():
            st.rerun()  # Refresh to show main app
    else:
        # Show main application
        main_app()

# ========================
# Run the Application
# ========================
if __name__ == "__main__":
    main()