# 🛠️ Developer Guide & Technical Specifications

## Advanced Stock Predictor AI - Developer Documentation

### 🏗️ **System Architecture**

---

## 📋 **Technical Stack Overview**

### **Core Technologies**

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Frontend** | Streamlit | 1.32.0 | Web application framework |
| **Backend** | Python | 3.12.4 | Core application logic |
| **Data Processing** | Pandas | 2.2.2 | Data manipulation & analysis |
| **Numerical Computing** | NumPy | 1.26.4 | Mathematical operations |
| **Machine Learning** | Scikit-learn | 1.4.2 | ML models & algorithms |
| **Data Visualization** | Plotly | 5.21.0 | Interactive charts |
| **Market Data** | yfinance | 0.2.37 | Yahoo Finance API |
| **Styling** | CSS3 | - | Custom UI components |

### **Architecture Patterns**

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                   │
├─────────────────────────────────────────────────────────┤
│  Streamlit Frontend  │  Custom CSS  │  Plotly Charts   │
├─────────────────────────────────────────────────────────┤
│                     BUSINESS LOGIC                      │
├─────────────────────────────────────────────────────────┤
│  Authentication  │  Technical Analysis  │  ML Predictor │
├─────────────────────────────────────────────────────────┤
│                     DATA ACCESS LAYER                   │
├─────────────────────────────────────────────────────────┤
│  Yahoo Finance API  │  Data Caching  │  Feature Store  │
├─────────────────────────────────────────────────────────┤
│                     STORAGE LAYER                       │
└─────────────────────────────────────────────────────────┘
│  Session State  │  Model Persistence  │  Cache Storage  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 **Development Environment Setup**

### **Prerequisites**

1. **Python Environment:**
   ```bash
   # Required Python version
   Python 3.10+ (Recommended: 3.12.4)
   
   # Package managers
   pip 23.0+
   conda (optional, for environment management)
   ```

2. **Development Tools:**
   ```bash
   # Code editor
   VS Code (with Python extension)
   PyCharm Professional
   
   # Version control
   Git 2.40+
   
   # Virtual environment
   venv, conda, or pipenv
   ```

### **Environment Configuration**

#### **1. Clone & Setup**

```bash
# Clone repository
git clone <repository-url>
cd advanced-stock-predictor

# Create virtual environment
python -m venv venv
# OR
conda create -n stock-predictor python=3.12

# Activate environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
# Conda
conda activate stock-predictor

# Install dependencies
pip install -r requirements.txt
```

#### **2. VS Code Configuration**

**`.vscode/settings.json`:**
```json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": [
        "--multi-line=3",
        "--trailing-comma",
        "--force-grid-wrap=0",
        "--combine-as",
        "--line-width=88"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/.pytest_cache": true
    }
}
```

**`.vscode/launch.json`:**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit App",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/venv/Scripts/streamlit.exe",
            "args": ["run", "main.py"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Debug ML Module",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/ml_predictor.py",
            "console": "integratedTerminal"
        }
    ]
}
```

#### **3. Environment Variables**

**`.env` file:**
```bash
# Development settings
DEBUG=True
STREAMLIT_THEME=dark
CACHE_TTL=3600

# API configurations
YAHOO_FINANCE_TIMEOUT=30
MAX_DATA_POINTS=1000

# ML model settings
MODEL_CACHE_DIR=./models
FEATURE_CACHE_TTL=1800

# Security settings
SESSION_SECRET_KEY=your-secret-key-here
```

---

## 📁 **Project Structure & Organization**

### **Directory Layout**

```
advanced-stock-predictor/
├── 📄 main.py                    # Main Streamlit application
├── 🤖 ml_predictor.py           # ML models and predictions
├── 🔐 login.py                  # Authentication system
├── 📋 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
├── 📄 LICENSE                   # MIT license
├── 📄 .gitignore               # Git ignore rules
├── 🗂️ docs/                     # Documentation files
│   ├── 📚 API_DOCUMENTATION.md
│   ├── 📖 USER_GUIDE.md
│   ├── 🛠️ DEVELOPER_GUIDE.md
│   └── 📸 SCREENSHOT_GUIDE.md
├── 🗂️ models/                   # Trained model storage
│   ├── 🧠 random_forest.joblib
│   ├── 🧠 gradient_boosting.joblib
│   └── 📊 scaler.joblib
├── 🗂️ data/                     # Data cache and storage
│   ├── 📈 stock_cache/
│   └── 📊 feature_cache/
├── 🗂️ tests/                    # Unit and integration tests
│   ├── 🧪 test_main.py
│   ├── 🧪 test_ml_predictor.py
│   └── 🧪 test_login.py
├── 🗂️ assets/                   # Static assets
│   ├── 🎨 css/
│   ├── 🖼️ images/
│   └── 📊 charts/
├── 🗂️ scripts/                  # Utility scripts
│   ├── 📸 take_screenshots.py
│   └── 🔧 data_validation.py
└── 🗂️ .vscode/                  # VS Code configuration
    ├── ⚙️ settings.json
    └── 🐛 launch.json
```

### **Module Dependencies**

```mermaid
graph TD
    A[main.py] --> B[ml_predictor.py]
    A --> C[login.py]
    B --> D[sklearn models]
    B --> E[pandas/numpy]
    A --> F[streamlit]
    A --> G[plotly]
    A --> H[yfinance]
    B --> H
    C --> I[session management]
```

---

## 💾 **Data Architecture**

### **Data Flow Pipeline**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Yahoo Finance │───▶│  Data Ingestion │───▶│     Caching     │
│       API       │    │     Module      │    │     Layer       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Predictor  │◀───│ Feature Engine  │◀───│ Data Processing │
│     Models      │    │     Module      │    │     Module      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│   Predictions   │    │  Technical      │
│    & Charts     │    │  Indicators     │
└─────────────────┘    └─────────────────┘
```

### **Data Models**

#### **Stock Data Schema**

```python
class StockData:
    """
    Core stock data structure
    """
    date: datetime          # Trading date
    open: float            # Opening price
    high: float            # Daily high
    low: float             # Daily low
    close: float           # Closing price
    volume: int            # Trading volume
    adjusted_close: float  # Dividend-adjusted close
    
    # Derived fields
    ticker: str            # Stock symbol
    market: str            # Exchange (NSE, BSE)
    sector: str            # Industry sector
    market_cap: float      # Market capitalization
```

#### **Feature Schema**

```python
class TechnicalFeatures:
    """
    Engineered features for ML models
    """
    # Price features
    close_price: float
    high_low_ratio: float
    price_change: float
    price_change_pct: float
    
    # Volume features
    volume: int
    volume_sma: float
    volume_ratio: float
    
    # Technical indicators
    rsi: float             # Relative Strength Index
    macd: float            # MACD line
    macd_signal: float     # MACD signal line
    macd_histogram: float  # MACD histogram
    bb_width: float        # Bollinger Band width
    bb_position: float     # BB position (0-1)
    
    # Moving averages
    sma_5: float
    sma_10: float
    sma_20: float
    sma_50: float
    ema_12: float
    ema_26: float
    
    # Price ratios
    price_to_sma_5: float
    price_to_sma_10: float
    price_to_sma_20: float
    price_to_sma_50: float
    
    # Lag features
    close_lag_1: float
    close_lag_2: float
    close_lag_3: float
    close_lag_5: float
    
    # Time features
    day_of_week: int       # 0-6 (Monday-Sunday)
    month: int             # 1-12
    quarter: int           # 1-4
    
    # Volatility
    volatility: float      # Rolling volatility
    momentum: float        # Price momentum
```

### **Caching Strategy**

#### **Multi-Level Caching**

```python
# Level 1: Streamlit built-in caching
@st.cache_data(ttl=3600)  # 1 hour TTL
def get_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """Cache raw stock data"""
    pass

@st.cache_data(ttl=1800)  # 30 minutes TTL
def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Cache feature engineering results"""
    pass

# Level 2: File-based caching
class DataCache:
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data from disk"""
        cache_file = self.cache_dir / f"{key}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None
    
    def set_cached_data(self, key: str, data: pd.DataFrame) -> None:
        """Store data to disk cache"""
        cache_file = self.cache_dir / f"{key}.parquet"
        data.to_parquet(cache_file)
```

---

## 🤖 **Machine Learning Architecture**

### **Model Pipeline Design**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Stock Data │───▶│ Feature Engine  │───▶│ Feature Matrix  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Trained Models │◀───│   Model Train   │◀───│ Data Splitting  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Predictions   │───▶│   Evaluation    │───▶│    Results      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Model Specifications**

#### **Random Forest Configuration**

```python
RandomForestRegressor(
    n_estimators=200,        # Number of trees
    max_depth=10,           # Maximum tree depth
    min_samples_split=5,    # Min samples for split
    min_samples_leaf=2,     # Min samples in leaf
    max_features='sqrt',    # Features per split
    bootstrap=True,         # Bootstrap sampling
    random_state=42,        # Reproducibility
    n_jobs=-1              # Parallel processing
)
```

**Hyperparameter Rationale:**
- `n_estimators=200`: Balance between accuracy and speed
- `max_depth=10`: Prevent overfitting while capturing complexity
- `min_samples_split=5`: Ensure statistical significance
- `max_features='sqrt'`: Optimal for regression tasks

#### **Gradient Boosting Configuration**

```python
GradientBoostingRegressor(
    n_estimators=200,       # Number of boosting stages
    learning_rate=0.1,      # Shrinkage parameter
    max_depth=6,           # Maximum tree depth
    min_samples_split=5,   # Min samples for split
    min_samples_leaf=2,    # Min samples in leaf
    subsample=0.8,         # Fraction of samples
    loss='squared_error',   # Loss function
    random_state=42        # Reproducibility
)
```

**Hyperparameter Rationale:**
- `learning_rate=0.1`: Conservative learning for stability
- `max_depth=6`: Shallower trees for ensemble method
- `subsample=0.8`: Stochastic gradient boosting

#### **Linear Regression Configuration**

```python
LinearRegression(
    fit_intercept=True,     # Include intercept term
    normalize=False,        # Features pre-normalized
    copy_X=True,           # Copy input arrays
    n_jobs=-1              # Parallel processing
)
```

### **Feature Engineering Pipeline**

#### **Technical Indicator Calculations**

```python
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive technical indicator calculation
    """
    indicators = pd.DataFrame(index=data.index)
    
    # Price-based features
    indicators['close_price'] = data['Close']
    indicators['high_low_ratio'] = data['High'] / data['Low']
    indicators['price_change'] = data['Close'].diff()
    indicators['price_change_pct'] = data['Close'].pct_change()
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        indicators[f'sma_{window}'] = data['Close'].rolling(window).mean()
        indicators[f'price_to_sma_{window}'] = (
            data['Close'] / indicators[f'sma_{window}']
        )
    
    # Exponential moving averages
    indicators['ema_12'] = data['Close'].ewm(span=12).mean()
    indicators['ema_26'] = data['Close'].ewm(span=26).mean()
    
    # RSI calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD calculation
    indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
    indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
    indicators['macd_histogram'] = (
        indicators['macd'] - indicators['macd_signal']
    )
    
    # Bollinger Bands
    bb_sma = data['Close'].rolling(20).mean()
    bb_std = data['Close'].rolling(20).std()
    bb_upper = bb_sma + (bb_std * 2)
    bb_lower = bb_sma - (bb_std * 2)
    indicators['bb_width'] = (bb_upper - bb_lower) / bb_sma
    indicators['bb_position'] = (
        (data['Close'] - bb_lower) / (bb_upper - bb_lower)
    )
    
    # Volatility and momentum
    indicators['volatility'] = data['Close'].rolling(14).std()
    indicators['momentum'] = data['Close'].pct_change(10)
    
    # Volume features
    indicators['volume'] = data['Volume']
    indicators['volume_sma'] = data['Volume'].rolling(20).mean()
    indicators['volume_ratio'] = data['Volume'] / indicators['volume_sma']
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        indicators[f'close_lag_{lag}'] = data['Close'].shift(lag)
        indicators[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
    
    # Time-based features
    indicators['day_of_week'] = data.index.dayofweek
    indicators['month'] = data.index.month
    indicators['quarter'] = data.index.quarter
    
    return indicators.dropna()
```

### **Model Training Process**

#### **Data Preparation**

```python
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(
            score_func=f_regression, 
            k=15
        )
    
    def prepare_training_data(
        self, 
        features_df: pd.DataFrame,
        target_col: str = 'close_price',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for model training
        """
        # Separate features and target
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')
        
        # Split data (time series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(
            X_train_scaled, y_train
        )
        X_test_selected = self.feature_selector.transform(
            X_test_scaled
        )
        
        return (
            X_train_selected, X_test_selected,
            y_train.values, y_test.values,
            self.scaler, X.columns.tolist()
        )
```

#### **Model Evaluation Metrics**

```python
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive model evaluation
    """
    metrics = {}
    
    # Regression metrics
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Financial metrics
    metrics['directional_accuracy'] = calculate_directional_accuracy(
        y_true, y_pred
    )
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(y_true, y_pred)
    
    return metrics

def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate percentage of correct direction predictions
    """
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    return np.mean(true_direction == pred_direction) * 100

def calculate_sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Sharpe ratio based on prediction returns
    """
    returns = np.diff(y_pred) / y_pred[:-1]
    excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
```

---

## 🎨 **Frontend Architecture**

### **Streamlit Component Structure**

```python
class StreamlitApp:
    def __init__(self):
        self.setup_page_config()
        self.apply_custom_styling()
        self.initialize_session_state()
    
    def setup_page_config(self) -> None:
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Advanced Stock Predictor AI",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/username/repo/issues',
                'Report a bug': 'https://github.com/username/repo/issues',
                'About': "Advanced AI-powered stock prediction platform"
            }
        )
    
    def apply_custom_styling(self) -> None:
        """Apply custom CSS styling"""
        st.markdown(
            """
            <style>
            /* Custom CSS styles */
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 30px;
            }
            
            .metric-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                margin-bottom: 20px;
            }
            
            .signal-bullish {
                color: #00ff88;
                font-weight: bold;
            }
            
            .signal-bearish {
                color: #ff4757;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
```

### **Component Architecture**

```python
class UIComponents:
    @staticmethod
    def render_sidebar() -> Dict[str, Any]:
        """Render sidebar with stock selection and parameters"""
        with st.sidebar:
            st.title("🎯 Stock Analysis")
            
            # Stock selection
            ticker = st.selectbox(
                "Select Stock",
                options=PENNY_STOCKS,
                index=0,
                format_func=lambda x: f"{x} - {STOCK_NAMES.get(x, x)}"
            )
            
            # Time period selection
            period = st.selectbox(
                "Time Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                index=5
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                ml_days = st.slider("Prediction Days", 1, 30, 7)
                show_confidence = st.checkbox("Show Confidence Intervals", True)
            
            return {
                'ticker': ticker,
                'period': period,
                'ml_days': ml_days,
                'show_confidence': show_confidence
            }
    
    @staticmethod
    def render_market_overview(overview_data: pd.DataFrame) -> None:
        """Render market overview section"""
        st.markdown("## 🌟 Market Overview")
        
        cols = st.columns(len(overview_data))
        for idx, (_, row) in enumerate(overview_data.iterrows()):
            with cols[idx]:
                change_color = "🟢" if row['Change'] > 0 else "🔴"
                st.metric(
                    label=f"{change_color} {row['Stock']}",
                    value=f"₹{row['Price']:.2f}",
                    delta=f"{row['Change']:.2f}%"
                )
    
    @staticmethod
    def render_technical_analysis(data: pd.DataFrame, indicators: Dict) -> None:
        """Render technical analysis section"""
        st.markdown("## 📊 Technical Analysis")
        
        # Price chart with moving averages
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))
        
        # Moving averages
        colors = ['red', 'orange', 'yellow', 'blue']
        for i, ma in enumerate([5, 10, 20, 50]):
            if f'MA{ma}' in indicators:
                fig.add_trace(go.Scatter(
                    x=data['Date'],
                    y=indicators[f'MA{ma}'],
                    mode='lines',
                    name=f'MA{ma}',
                    line=dict(color=colors[i], width=2)
                ))
        
        fig.update_layout(
            title="Price Chart with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
```

---

## 🧪 **Testing Strategy**

### **Test Architecture**

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='D'
    )
    
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    return pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(len(dates)) * 0.1,
        'High': prices + np.abs(np.random.randn(len(dates))) * 0.5,
        'Low': prices - np.abs(np.random.randn(len(dates))) * 0.5,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

@pytest.fixture
def ml_predictor():
    """Initialize ML predictor for testing"""
    from ml_predictor import StockPredictor
    return StockPredictor()
```

### **Unit Tests**

```python
# tests/test_ml_predictor.py
import pytest
import numpy as np
import pandas as pd
from ml_predictor import StockPredictor, create_features

class TestStockPredictor:
    def test_predictor_initialization(self, ml_predictor):
        """Test predictor initialization"""
        assert len(ml_predictor.models) == 3
        assert 'random_forest' in ml_predictor.models
        assert 'gradient_boosting' in ml_predictor.models
        assert 'linear_regression' in ml_predictor.models
    
    def test_feature_creation(self, sample_stock_data):
        """Test feature engineering"""
        features = create_features(sample_stock_data)
        
        # Check required columns exist
        required_features = [
            'close_price', 'rsi', 'macd', 'bb_width',
            'price_to_sma_20', 'volatility', 'momentum'
        ]
        for feature in required_features:
            assert feature in features.columns
        
        # Check data types
        assert features['close_price'].dtype in [np.float64, np.float32]
        assert features['day_of_week'].dtype in [np.int64, np.int32]
        
        # Check value ranges
        assert features['rsi'].min() >= 0
        assert features['rsi'].max() <= 100
        assert features['day_of_week'].min() >= 0
        assert features['day_of_week'].max() <= 6
    
    def test_model_training(self, ml_predictor, sample_stock_data):
        """Test model training process"""
        features = create_features(sample_stock_data)
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler, feature_cols = (
            ml_predictor.prepare_data(features)
        )
        
        # Train models
        results = ml_predictor.train_models(
            X_train, y_train, X_test, y_test, scaler, feature_cols
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        for model_name in ml_predictor.models.keys():
            assert model_name in results
            assert 'rmse' in results[model_name]
            assert 'mae' in results[model_name]
            assert 'r2' in results[model_name]
            
            # Check metric ranges
            assert results[model_name]['rmse'] > 0
            assert results[model_name]['mae'] > 0
            assert results[model_name]['r2'] <= 1.0
    
    def test_prediction_generation(self, ml_predictor, sample_stock_data):
        """Test future prediction generation"""
        features = create_features(sample_stock_data)
        
        # Train models first
        X_train, X_test, y_train, y_test, scaler, feature_cols = (
            ml_predictor.prepare_data(features)
        )
        ml_predictor.train_models(
            X_train, y_train, X_test, y_test, scaler, feature_cols
        )
        
        # Generate predictions
        predictions = ml_predictor.predict_future(features, days_ahead=7)
        
        # Verify prediction structure
        assert isinstance(predictions, dict)
        assert len(predictions) == 7  # 7 days
        
        for day_pred in predictions.values():
            assert isinstance(day_pred, dict)
            for model_name in ml_predictor.models.keys():
                assert model_name in day_pred
                assert isinstance(day_pred[model_name], (int, float))
```

### **Integration Tests**

```python
# tests/test_integration.py
import pytest
import streamlit as st
from unittest.mock import patch, MagicMock

class TestStreamlitIntegration:
    @patch('streamlit.cache_data')
    @patch('yfinance.download')
    def test_data_loading_workflow(self, mock_yf_download, mock_cache, sample_stock_data):
        """Test complete data loading workflow"""
        # Mock yfinance response
        mock_yf_download.return_value = sample_stock_data.set_index('Date')
        
        from main import get_stock_data
        
        # Test data loading
        data = get_stock_data("YESBANK.NS", "1y")
        
        assert not data.empty
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_ml_prediction_workflow(self, sample_stock_data):
        """Test complete ML prediction workflow"""
        from ml_predictor import StockPredictor, create_features
        
        # Create features
        features = create_features(sample_stock_data)
        
        # Initialize and train predictor
        predictor = StockPredictor()
        X_train, X_test, y_train, y_test, scaler, feature_cols = (
            predictor.prepare_data(features)
        )
        
        results = predictor.train_models(
            X_train, y_train, X_test, y_test, scaler, feature_cols
        )
        
        # Generate predictions
        predictions = predictor.predict_future(features, days_ahead=7)
        
        # Verify complete workflow
        assert len(results) == 3  # Three models
        assert len(predictions) == 7  # Seven days
        assert all(isinstance(p, dict) for p in predictions.values())
```

### **Performance Tests**

```python
# tests/test_performance.py
import time
import pytest
import pandas as pd
from ml_predictor import create_features, StockPredictor

class TestPerformance:
    def test_feature_creation_performance(self, sample_stock_data):
        """Test feature creation performance"""
        start_time = time.time()
        features = create_features(sample_stock_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within 5 seconds for 1 year of data
        assert execution_time < 5.0
        assert not features.empty
    
    def test_model_training_performance(self, sample_stock_data):
        """Test model training performance"""
        features = create_features(sample_stock_data)
        predictor = StockPredictor()
        
        X_train, X_test, y_train, y_test, scaler, feature_cols = (
            predictor.prepare_data(features)
        )
        
        start_time = time.time()
        results = predictor.train_models(
            X_train, y_train, X_test, y_test, scaler, feature_cols
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Training should complete within 30 seconds
        assert execution_time < 30.0
        assert len(results) == 3
    
    @pytest.mark.parametrize("data_size", [100, 500, 1000])
    def test_scalability(self, data_size):
        """Test performance with different data sizes"""
        # Generate data of specified size
        dates = pd.date_range(start='2020-01-01', periods=data_size, freq='D')
        data = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.random.randn(data_size),
            'High': 102 + np.random.randn(data_size),
            'Low': 98 + np.random.randn(data_size),
            'Close': 100 + np.random.randn(data_size),
            'Volume': np.random.randint(1000000, 10000000, data_size)
        })
        
        start_time = time.time()
        features = create_features(data)
        end_time = time.time()
        
        # Performance should scale linearly
        time_per_record = (end_time - start_time) / data_size
        assert time_per_record < 0.01  # Less than 10ms per record
```

---

## 🚀 **Deployment & Production**

### **Production Configuration**

#### **Environment Variables**

```bash
# Production environment file
ENVIRONMENT=production
DEBUG=False

# Security settings
SECRET_KEY=your-production-secret-key
SESSION_TIMEOUT=3600

# Performance settings
CACHE_TTL=1800
MAX_WORKERS=4
MEMORY_LIMIT=2GB

# External APIs
YAHOO_FINANCE_TIMEOUT=10
RATE_LIMIT_PER_MINUTE=60

# Monitoring
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn
```

#### **Docker Configuration**

**`Dockerfile`:**
```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**`docker-compose.yml`:**
```yaml
version: '3.8'

services:
  stock-predictor:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - CACHE_TTL=1800
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - stock-predictor
    restart: unless-stopped
```

### **Performance Optimization**

#### **Caching Strategy**

```python
# config/cache_config.py
import redis
from functools import wraps
import pickle

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
    
    def cache_result(self, expiration=3600):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)
                
                # Compute and cache result
                result = func(*args, **kwargs)
                self.redis_client.setex(
                    cache_key, 
                    expiration, 
                    pickle.dumps(result)
                )
                
                return result
            return wrapper
        return decorator

# Usage
cache = RedisCache()

@cache.cache_result(expiration=1800)
def get_stock_data_cached(ticker, period):
    # Expensive data fetching operation
    return fetch_stock_data(ticker, period)
```

#### **Database Integration**

```python
# config/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
class PredictionResult(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(20), nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    model_name = Column(String(50), nullable=False)
    predicted_price = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=True)

# Database connection
DATABASE_URL = "postgresql://user:password@localhost/stockpredict"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### **Monitoring & Logging**

#### **Application Monitoring**

```python
# utils/monitoring.py
import logging
import time
from functools import wraps
from typing import Dict, Any
import psutil
import streamlit as st

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    def track_function_performance(self, func_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    result = func(*args, **kwargs)
                    status = 'success'
                except Exception as e:
                    result = None
                    status = 'error'
                    logging.error(f"Error in {func_name}: {str(e)}")
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Log performance metrics
                self.log_performance({
                    'function': func_name,
                    'execution_time': end_time - start_time,
                    'memory_usage': end_memory - start_memory,
                    'status': status,
                    'timestamp': time.time()
                })
                
                return result
            return wrapper
        return decorator
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        logging.info(f"Performance: {metrics}")
        
        # Store in session state for dashboard
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
        
        st.session_state.performance_metrics.append(metrics)

# Usage
monitor = PerformanceMonitor()

@monitor.track_function_performance('ml_prediction')
def train_and_predict(data):
    # ML training and prediction code
    pass
```

#### **Error Handling & Alerting**

```python
# utils/error_handling.py
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
import smtplib
from email.mime.text import MIMEText

# Initialize Sentry for error tracking
sentry_logging = LoggingIntegration(
    level=logging.INFO,
    event_level=logging.ERROR
)

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[sentry_logging],
    traces_sample_rate=1.0,
    environment="production"
)

class AlertManager:
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_alert(self, subject: str, message: str, 
                   recipients: list):
        """Send email alert for critical errors"""
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
        except Exception as e:
            logging.error(f"Failed to send alert: {str(e)}")

def handle_critical_error(error: Exception, context: Dict[str, Any]):
    """Handle critical application errors"""
    # Log to Sentry
    sentry_sdk.capture_exception(error)
    
    # Send email alert
    alert_manager = AlertManager(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="alerts@yourcompany.com",
        password="your-password"
    )
    
    alert_manager.send_alert(
        subject=f"Critical Error in Stock Predictor: {type(error).__name__}",
        message=f"Error: {str(error)}\nContext: {context}",
        recipients=["admin@yourcompany.com"]
    )
```

---

## 📚 **API Integration Guidelines**

### **External API Management**

#### **Yahoo Finance Integration**

```python
# utils/data_sources.py
import yfinance as yf
import requests
from typing import Optional, Dict, Any
import time
import logging

class YahooFinanceClient:
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Set user agent to avoid rate limiting
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch stock data with retry logic and error handling
        """
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period=period, timeout=self.timeout)
                
                if data.empty:
                    logging.warning(f"No data returned for {ticker}")
                    return None
                
                # Reset index to get Date as column
                data.reset_index(inplace=True)
                return data
                
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout on attempt {attempt + 1} for {ticker}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error for {ticker}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                
            except Exception as e:
                logging.error(f"Unexpected error for {ticker}: {str(e)}")
                break
        
        return None
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get additional stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta')
            }
        except Exception as e:
            logging.error(f"Error fetching info for {ticker}: {str(e)}")
            return {}
```

### **Rate Limiting & Throttling**

```python
# utils/rate_limiter.py
import time
from collections import defaultdict, deque
from typing import Dict
import threading

class RateLimiter:
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
    
    def allow_request(self, identifier: str) -> bool:
        """
        Check if request is allowed under rate limit
        """
        with self.lock:
            now = time.time()
            request_times = self.requests[identifier]
            
            # Remove old requests outside time window
            while request_times and request_times[0] < now - self.time_window:
                request_times.popleft()
            
            # Check if under limit
            if len(request_times) < self.max_requests:
                request_times.append(now)
                return True
            
            return False
    
    def get_reset_time(self, identifier: str) -> float:
        """Get time when rate limit resets"""
        with self.lock:
            request_times = self.requests[identifier]
            if request_times:
                return request_times[0] + self.time_window
            return time.time()

# Usage in data fetching
rate_limiter = RateLimiter(max_requests=60, time_window=60)

def fetch_with_rate_limit(ticker: str):
    if not rate_limiter.allow_request('yahoo_finance'):
        reset_time = rate_limiter.get_reset_time('yahoo_finance')
        wait_time = reset_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)
    
    return yf_client.get_stock_data(ticker)
```

---

This comprehensive developer guide covers all aspects of the system architecture, from setup and configuration to deployment and monitoring. Use it as a reference for understanding the codebase structure, implementing new features, and maintaining the application in production.

---

*For specific implementation examples and code snippets, refer to the source files and API documentation.*
