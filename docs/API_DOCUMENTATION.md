# 📋 API Documentation

## Advanced Stock Predictor AI - API Reference

### 🔧 **Core Components**

---

## 📊 **main.py - Main Application Module**

### **Functions**

#### `apply_main_styles()`
Applies the custom CSS styling for the Streamlit application.

**Returns:** None

**Purpose:** 
- Sets up dark theme with gradient background
- Configures glass-morphism UI elements
- Defines color schemes for trading signals

---

#### `calculate_rsi(prices, window=14)`
Calculates the Relative Strength Index for momentum analysis.

**Parameters:**
- `prices` (pd.Series | np.array): Price data series
- `window` (int): Period for RSI calculation (default: 14)

**Returns:** 
- `pd.Series`: RSI values (0-100 scale)

**Example:**
```python
rsi_values = calculate_rsi(data['Close'], window=14)
```

**Formula:**
```
RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss
```

---

#### `calculate_macd(prices, fast=12, slow=26, signal=9)`
Calculates MACD (Moving Average Convergence Divergence) indicator.

**Parameters:**
- `prices` (pd.Series): Price data
- `fast` (int): Fast EMA period (default: 12)
- `slow` (int): Slow EMA period (default: 26)
- `signal` (int): Signal line EMA period (default: 9)

**Returns:**
- `tuple`: (macd_line, signal_line, histogram)

**Trading Signals:**
- Bullish: MACD crosses above signal line
- Bearish: MACD crosses below signal line

---

#### `calculate_moving_averages(prices)`
Calculates multiple moving averages for trend analysis.

**Parameters:**
- `prices` (pd.Series): Price data

**Returns:**
- `tuple`: (ma_5, ma_10, ma_20, ma_50, ema_12, ema_26)

**Usage:**
```python
ma_5, ma_10, ma_20, ma_50, ema_12, ema_26 = calculate_moving_averages(data['Close'])
```

---

#### `get_stock_data(ticker, period="1y")`
Fetches stock data from Yahoo Finance with caching.

**Parameters:**
- `ticker` (str): Stock symbol (e.g., "YESBANK.NS")
- `period` (str): Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")

**Returns:**
- `pd.DataFrame`: Stock data with columns [Date, Open, High, Low, Close, Volume]

**Caching:** Uses `@st.cache_data` for performance optimization

---

#### `get_market_overview(tickers)`
Generates market overview dashboard for multiple stocks.

**Parameters:**
- `tickers` (list): List of stock symbols

**Returns:**
- `pd.DataFrame`: Market overview with columns [Stock, Price, Change, Volume, Change_Val]

**Example:**
```python
overview = get_market_overview(["YESBANK.NS", "SUZLON.NS"])
```

---

## 🤖 **ml_predictor.py - Machine Learning Module**

### **Classes**

#### `StockPredictor`
Main class for machine learning predictions and model management.

**Attributes:**
- `models` (dict): Dictionary of ML models
- `trained_models` (dict): Trained model instances
- `scalers` (dict): Feature scaling objects
- `feature_importance` (dict): Feature importance scores

**Methods:**

##### `__init__()`
Initializes the StockPredictor with default models.

**Models:**
- Random Forest Regressor (200 estimators)
- Gradient Boosting Regressor (200 estimators)
- Linear Regression

---

##### `prepare_data(features_df, target_col='close_price', test_size=0.2)`
Prepares data for model training.

**Parameters:**
- `features_df` (pd.DataFrame): Feature matrix
- `target_col` (str): Target column name
- `test_size` (float): Proportion for test set

**Returns:**
- `tuple`: (X_train, X_test, y_train, y_test, scaler, feature_columns)

**Data Processing:**
- Feature scaling using StandardScaler
- Train/test split with temporal ordering
- Missing value handling

---

##### `train_models(X_train, y_train, X_test, y_test, scaler, feature_columns)`
Trains all machine learning models.

**Parameters:**
- Training and testing data arrays
- Scaler object and feature column names

**Returns:**
- `dict`: Model results with metrics

**Metrics Calculated:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of Determination)
- Feature importance (for tree-based models)

---

##### `predict_future(features_df, days_ahead=7)`
Generates future price predictions.

**Parameters:**
- `features_df` (pd.DataFrame): Historical features
- `days_ahead` (int): Number of days to predict

**Returns:**
- `dict`: Predictions by model with confidence scores

---

### **Feature Engineering Functions**

#### `create_features(data)`
Creates comprehensive feature set for ML models.

**Input Features:**
- Price data (Open, High, Low, Close, Volume)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Moving averages (SMA, EMA)
- Time-based features (day of week, month)
- Lag features (1, 2, 3, 5 periods)

**Output:**
- `pd.DataFrame`: 20+ engineered features

**Feature Categories:**
1. **Price Features:**
   - `close_price`: Current closing price
   - `high_low_ratio`: Daily high/low ratio
   - `volume`: Trading volume

2. **Technical Indicators:**
   - `rsi`: Relative Strength Index
   - `bb_width`: Bollinger Band width
   - `bb_position`: Position within bands
   - `volatility`: Price volatility
   - `momentum`: Price momentum

3. **Moving Averages:**
   - `price_to_ma_5/10/20/50`: Price to MA ratios
   - `price_to_ema_12/26`: Price to EMA ratios

4. **MACD Features:**
   - `macd`: MACD line
   - `macd_signal`: Signal line
   - `macd_histogram`: MACD histogram

5. **Time Features:**
   - `day_of_week`: Weekday (0-6)
   - `month`: Month (1-12)

6. **Lag Features:**
   - `close_lag_1/2/3/5`: Lagged prices
   - `volume_lag_1/2/3/5`: Lagged volume

---

### **Technical Indicator Functions**

#### `calculate_bollinger_bands(prices, window=20, num_std=2)`
Calculates Bollinger Bands for volatility analysis.

**Parameters:**
- `prices` (array-like): Price series
- `window` (int): Moving average period
- `num_std` (float): Standard deviation multiplier

**Returns:**
- `tuple`: (bb_width, bb_position)

**Interpretation:**
- `bb_width > 0.1`: High volatility
- `bb_position > 0.8`: Near upper band (overbought)
- `bb_position < 0.2`: Near lower band (oversold)

---

#### `calculate_volatility(prices, window=14)`
Calculates rolling price volatility.

**Parameters:**
- `prices` (array-like): Price series
- `window` (int): Rolling window size

**Returns:**
- `np.array`: Volatility values

---

#### `calculate_momentum(prices, window=10)`
Calculates price momentum indicator.

**Parameters:**
- `prices` (array-like): Price series
- `window` (int): Momentum period

**Returns:**
- `np.array`: Momentum values (percentage change)

---

## 🔐 **login.py - Authentication Module**

### **Functions**

#### `show_login_page()`
Displays the login interface with animated styling.

**Returns:**
- `bool`: True if login successful

**Features:**
- Demo user access (email: demo@example.com)
- Registration form for new users
- Session state management
- Animated UI with glassmorphism design

---

#### `is_authenticated()`
Checks if user is currently authenticated.

**Returns:**
- `bool`: Authentication status

---

#### `get_current_user()`
Retrieves current user information.

**Returns:**
- `dict`: User information with keys ['email', 'type']

---

#### `logout()`
Logs out the current user and clears session state.

**Returns:** None

---

## 📊 **Visualization Functions**

### **Plotly Chart Functions**

#### `plot_model_comparison(models_results)`
Creates model performance comparison chart.

**Parameters:**
- `models_results` (dict): Model results with metrics

**Returns:**
- `plotly.graph_objects.Figure`: Bar chart comparing RMSE scores

---

#### `plot_predictions_vs_actual(models_results)`
Plots predicted vs actual values for model validation.

**Parameters:**
- `models_results` (dict): Model results with predictions

**Returns:**
- `plotly.graph_objects.Figure`: Subplots comparing predictions

---

#### `plot_feature_importance(models_results)`
Visualizes feature importance for tree-based models.

**Parameters:**
- `models_results` (dict): Model results with feature importance

**Returns:**
- `plotly.graph_objects.Figure`: Horizontal bar charts

---

#### `plot_future_predictions(current_price, future_predictions, days_ahead)`
Plots future price predictions from multiple models.

**Parameters:**
- `current_price` (float): Current stock price
- `future_predictions` (dict): Predictions by model
- `days_ahead` (int): Prediction horizon

**Returns:**
- `plotly.graph_objects.Figure`: Line chart with predictions

---

## 🎯 **Configuration & Constants**

### **Supported Stocks**
```python
PENNY_STOCKS = [
    "YESBANK.NS",    # Yes Bank
    "SUZLON.NS",     # Suzlon Energy
    "PNB.NS",        # Punjab National Bank
    "IDEA.NS",       # Vodafone Idea
    "RPOWER.NS",     # Reliance Power
    "JPPOWER.NS",    # Jaiprakash Power
    "IRFC.NS",       # Indian Railway Finance
    "ONGC.NS",       # Oil and Natural Gas
    "IOB.NS",        # Indian Overseas Bank
    "TATAPOWER.NS"   # Tata Power
]
```

### **Color Schemes**
```python
SIGNAL_COLORS = {
    'bullish': '#00ff88',    # Green
    'bearish': '#ff4757',    # Red
    'neutral': '#ffa726',    # Orange
    'info': '#4facfe',       # Blue
    'purple': '#667eea',     # Purple
    'cyan': '#00f2fe'        # Cyan
}
```

### **Model Parameters**
```python
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 10,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'linear_regression': {}
}
```

---

## 🚀 **Usage Examples**

### **Basic Stock Analysis**
```python
import streamlit as st
from main import get_stock_data, calculate_rsi, calculate_macd

# Get stock data
data = get_stock_data("YESBANK.NS", period="1y")

# Calculate technical indicators
rsi = calculate_rsi(data['Close'])
macd_line, signal_line, histogram = calculate_macd(data['Close'])

# Display results
st.line_chart(data['Close'])
st.line_chart(rsi)
```

### **ML Prediction Workflow**
```python
from ml_predictor import StockPredictor, create_features

# Prepare data
features_df = create_features(data)
predictor = StockPredictor()

# Train models
X_train, X_test, y_train, y_test, scaler, features = predictor.prepare_data(features_df)
results = predictor.train_models(X_train, y_train, X_test, y_test, scaler, features)

# Make predictions
future_predictions = predictor.predict_future(features_df, days_ahead=7)
```

### **Custom Visualization**
```python
import plotly.graph_objects as go

# Create custom chart
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
))

# Add moving average
fig.add_trace(go.Scatter(
    x=data['Date'],
    y=calculate_moving_averages(data['Close'])[2],  # MA20
    name='MA20'
))

st.plotly_chart(fig)
```

---

## ⚠️ **Error Handling**

### **Common Issues & Solutions**

#### `Series.format Error`
**Cause:** Pandas Series passed to f-string formatting
**Solution:** Convert to scalar using `.item()` or `float()`

```python
# Wrong
st.write(f"Price: {series_value:.2f}")

# Correct
st.write(f"Price: {float(series_value):.2f}")
```

#### `Import Errors`
**Cause:** Missing dependencies or incorrect Python interpreter
**Solution:** Install requirements and set correct interpreter

```bash
pip install -r requirements.txt
```

#### `Data Loading Issues`
**Cause:** Network connectivity or invalid ticker symbols
**Solution:** Check internet connection and verify ticker format

---

## 📈 **Performance Optimization**

### **Caching Strategies**
- `@st.cache_data` for data fetching functions
- `@st.cache_resource` for model loading
- Session state for user preferences

### **Memory Management**
- Limit historical data periods
- Use efficient data types (float32 vs float64)
- Clear unused variables in large datasets

### **UI Responsiveness**
- Use `st.spinner()` for long operations
- Progressive loading with status updates
- Async operations where possible

---

*For detailed implementation examples, see the source code and usage documentation.*
