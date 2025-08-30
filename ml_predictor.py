
# ml_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================
# Technical Indicators for Features (Fixed for 1D data)
# ========================
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    try:
        # Ensure 1D data
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        elif hasattr(prices, 'flatten'):
            prices = prices.flatten()
            
        prices = pd.Series(prices, dtype=float).dropna()
        
        if len(prices) < window:
            return np.array([50] * len(prices))
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    except Exception:
        return np.array([50] * len(prices))

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    try:
        # Ensure 1D data
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        elif hasattr(prices, 'flatten'):
            prices = prices.flatten()
            
        prices = pd.Series(prices, dtype=float).dropna()
        
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bb_width = (upper_band - lower_band) / rolling_mean.replace(0, np.inf)
        bb_position = (prices - lower_band) / (upper_band - lower_band).replace(0, np.inf)
        return bb_width.fillna(0).values, bb_position.fillna(0.5).values
    except Exception:
        return np.array([0] * len(prices)), np.array([0.5] * len(prices))

def calculate_volatility(prices, window=14):
    """Calculate price volatility"""
    try:
        # Ensure 1D data
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        elif hasattr(prices, 'flatten'):
            prices = prices.flatten()
            
        prices = pd.Series(prices, dtype=float).dropna()
        returns = prices.pct_change()
        volatility = returns.rolling(window=window, min_periods=1).std()
        return volatility.fillna(0).values
    except Exception:
        return np.array([0] * len(prices))

def calculate_momentum(prices, window=10):
    """Calculate price momentum"""
    try:
        # Ensure 1D data
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        elif hasattr(prices, 'flatten'):
            prices = prices.flatten()
            
        prices = pd.Series(prices, dtype=float).dropna()
        momentum = prices / prices.shift(window) - 1
        return momentum.fillna(0).values
    except Exception:
        return np.array([0] * len(prices))

# ========================
# Feature Engineering
# ========================
def create_features(data):
    """Create comprehensive feature set for ML models"""
    try:
        features_df = pd.DataFrame()
        
        # Ensure we have the required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Ensure we have the right data types and handle 1D arrays
        close_prices = data['Close'].values.flatten() if hasattr(data['Close'].values, 'flatten') else data['Close'].values
        high_prices = data['High'].values.flatten() if hasattr(data['High'].values, 'flatten') else data['High'].values
        low_prices = data['Low'].values.flatten() if hasattr(data['Low'].values, 'flatten') else data['Low'].values
        volume = data['Volume'].values.flatten() if hasattr(data['Volume'].values, 'flatten') else data['Volume'].values
        
        # Convert to float and handle any non-numeric values
        close_prices = pd.to_numeric(close_prices, errors='coerce')
        high_prices = pd.to_numeric(high_prices, errors='coerce')
        low_prices = pd.to_numeric(low_prices, errors='coerce')
        volume = pd.to_numeric(volume, errors='coerce')
        
        # Price-based features
        features_df['close_price'] = close_prices
        features_df['high_low_ratio'] = high_prices / np.maximum(low_prices, 0.001)  # Avoid division by zero
        features_df['volume'] = volume
        
        # Moving averages
        close_series = pd.Series(close_prices).dropna()
        if len(close_series) == 0:
            st.error("No valid price data available")
            return pd.DataFrame()
            
        for window in [5, 10, 20, 50]:
            try:
                ma = close_series.rolling(window=window, min_periods=1).mean()
                features_df[f'ma_{window}'] = ma
                # Avoid division by zero
                ma_nonzero = ma.replace(0, np.nan).fillna(close_series.mean())
                features_df[f'price_to_ma_{window}'] = close_series / ma_nonzero
            except Exception as e:
                st.warning(f"Error calculating MA{window}: {str(e)}")
                features_df[f'ma_{window}'] = close_series
                features_df[f'price_to_ma_{window}'] = 1.0
        
        # Exponential moving averages
        for span in [12, 26]:
            try:
                ema = close_series.ewm(span=span, adjust=False).mean()
                features_df[f'ema_{span}'] = ema
                # Avoid division by zero
                ema_nonzero = ema.replace(0, np.nan).fillna(close_series.mean())
                features_df[f'price_to_ema_{span}'] = close_series / ema_nonzero
            except Exception as e:
                st.warning(f"Error calculating EMA{span}: {str(e)}")
                features_df[f'ema_{span}'] = close_series
                features_df[f'price_to_ema_{span}'] = 1.0
        
        # Technical indicators
        features_df['rsi'] = calculate_rsi(close_prices)
        
        bb_width, bb_position = calculate_bollinger_bands(close_prices)
        features_df['bb_width'] = bb_width
        features_df['bb_position'] = bb_position
        
        features_df['volatility'] = calculate_volatility(close_prices)
        features_df['momentum'] = calculate_momentum(close_prices)
        
        # Volume indicators
        volume_series = pd.Series(volume).dropna()
        if len(volume_series) > 0:
            try:
                volume_ma = volume_series.rolling(window=10, min_periods=1).mean()
                features_df['volume_ma'] = volume_ma
                # Avoid division by zero
                volume_ma_nonzero = volume_ma.replace(0, np.nan).fillna(volume_series.mean())
                features_df['volume_ratio'] = volume_series / volume_ma_nonzero
            except Exception as e:
                st.warning(f"Error calculating volume indicators: {str(e)}")
                features_df['volume_ma'] = volume_series
                features_df['volume_ratio'] = 1.0
        else:
            features_df['volume_ma'] = 0
            features_df['volume_ratio'] = 1.0
        
        # MACD
        ema_12 = close_series.ewm(span=12, adjust=False).mean()
        ema_26 = close_series.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_histogram'] = macd - macd_signal
        
        # Time-based features
        if 'Date' in data.columns:
            dates = pd.to_datetime(data['Date'])
            features_df['day_of_week'] = dates.dt.dayofweek
            features_df['month'] = dates.dt.month
        else:
            features_df['day_of_week'] = 0
            features_df['month'] = 1
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = close_series.shift(lag)
            features_df[f'volume_lag_{lag}'] = volume_series.shift(lag)
        
        # Price patterns
        features_df['price_change_1d'] = close_series.pct_change(1)
        features_df['price_change_3d'] = close_series.pct_change(3)
        features_df['price_change_7d'] = close_series.pct_change(7)
        
        # Fill NaN values (updated for newer pandas versions)
        features_df = features_df.bfill().ffill().fillna(0)
        
        # Replace infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        return features_df
        
    except Exception as e:
        st.error(f"Error creating features: {str(e)}")
        return pd.DataFrame()

# ========================
# ML Models Class
# ========================
class StockPredictor:
    def __init__(self):
        self.models = {
            "🌲 Random Forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            "🚀 Gradient Boosting": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            "📈 Linear Regression": LinearRegression(),
        }
        
        self.scalers = {}
        self.trained_models = {}
        self.feature_importance = {}
        self.model_metrics = {}
    
    def prepare_data(self, features_df, target_col='close_price', test_size=0.2):
        """Prepare data for training"""
        try:
            # Remove target column from features
            feature_columns = [col for col in features_df.columns if col != target_col]
            X = features_df[feature_columns].values
            y = features_df[target_col].values
            
            # Split data
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None, None, None, None, None
    
    def train_models(self, X_train, y_train, X_test, y_test, scaler, feature_columns):
        """Train all models"""
        try:
            results = {}
            
            for name, model in self.models.items():
                with st.spinner(f"Training {name}..."):
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    test_mae = mean_absolute_error(y_test, test_pred)
                    test_r2 = r2_score(y_test, test_pred)
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'test_r2': test_r2,
                        'test_predictions': test_pred,
                        'test_actual': y_test
                    }
                    
                    # Feature importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': feature_columns,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        results[name]['feature_importance'] = importance_df
            
            self.trained_models = results
            self.scalers['main'] = scaler
            return results
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return {}
    
    def predict_future(self, features_df, days_ahead=7):
        """Make future predictions"""
        try:
            if not self.trained_models:
                return {}
            
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            feature_columns = [col for col in features_df.columns if col != 'close_price']
            
            predictions = {}
            
            for name, model_data in self.trained_models.items():
                model = model_data['model']
                scaler = self.scalers['main']
                
                # Scale features
                latest_scaled = scaler.transform(latest_features[feature_columns].values)
                
                # Predict future prices
                future_predictions = []
                current_features = latest_scaled[0].copy()
                
                for _ in range(days_ahead):
                    pred = model.predict([current_features])[0]
                    future_predictions.append(pred)
                    
                    # Update features (simplified approach)
                    # In practice, you'd want more sophisticated feature updating
                    current_features = np.roll(current_features, 1)
                    current_features[0] = pred
                
                predictions[name] = {
                    'predictions': future_predictions,
                    'rmse': float(model_data['test_rmse']),  # Ensure scalar
                    'r2': float(model_data['test_r2'])       # Ensure scalar
                }
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making future predictions: {str(e)}")
            return {}

# ========================
# Visualization Functions
# ========================
def plot_model_comparison(models_results):
    """Plot model performance comparison"""
    try:
        fig = go.Figure()
        
        models = list(models_results.keys())
        # Ensure all values are scalars
        rmse_scores = [float(models_results[model]['test_rmse']) for model in models]
        r2_scores = [float(models_results[model]['test_r2']) for model in models]
        
        # RMSE comparison
        fig.add_trace(go.Bar(
            name='RMSE (₹)',
            x=models,
            y=rmse_scores,
            marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
            yaxis='y',
            text=[f'₹{score:.2f}' for score in rmse_scores],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='🎯 Model Performance Comparison - RMSE (Lower is Better)',
            template='plotly_dark',
            height=400,
            xaxis_title='Models',
            yaxis_title='RMSE (₹)',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting model comparison: {str(e)}")
        return go.Figure()

def plot_predictions_vs_actual(models_results):
    """Plot predictions vs actual values"""
    try:
        fig = make_subplots(
            rows=len(models_results), cols=1,
            subplot_titles=[f"{name} - Predictions vs Actual" for name in models_results.keys()],
            vertical_spacing=0.1
        )
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        for i, (name, results) in enumerate(models_results.items()):
            actual = results['test_actual']
            predicted = results['test_predictions']
            
            # Actual values
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(actual))),
                    y=actual,
                    mode='lines',
                    name=f'{name} - Actual',
                    line=dict(color='white', width=2),
                    showlegend=i==0
                ),
                row=i+1, col=1
            )
            
            # Predicted values
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(predicted))),
                    y=predicted,
                    mode='lines',
                    name=f'{name} - Predicted',
                    line=dict(color=colors[i], width=2, dash='dash'),
                    showlegend=i==0
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title='📊 Model Predictions vs Actual Prices (Test Set)',
            template='plotly_dark',
            height=300 * len(models_results),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting predictions vs actual: {str(e)}")
        return go.Figure()

def plot_feature_importance(models_results):
    """Plot feature importance for tree-based models"""
    try:
        models_with_importance = {name: results for name, results in models_results.items() 
                                if 'feature_importance' in results}
        
        if not models_with_importance:
            return None
        
        fig = make_subplots(
            rows=1, cols=len(models_with_importance),
            subplot_titles=list(models_with_importance.keys()),
            specs=[[{"type": "bar"}] * len(models_with_importance)]
        )
        
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        for i, (name, results) in enumerate(models_with_importance.items()):
            importance_df = results['feature_importance'].head(10)  # Top 10 features
            
            fig.add_trace(
                go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color=colors[i],
                    name=name,
                    showlegend=False
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='🔍 Top 10 Most Important Features',
            template='plotly_dark',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")
        return go.Figure()

def plot_future_predictions(current_price, future_predictions, days_ahead):
    """Plot future price predictions"""
    try:
        fig = go.Figure()
        
        # Historical point (current price)
        fig.add_trace(go.Scatter(
            x=[0],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(size=10, color='white', symbol='star')
        ))
        
        # Future predictions
        future_days = list(range(1, days_ahead + 1))
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        for i, (name, pred_data) in enumerate(future_predictions.items()):
            predictions = pred_data['predictions']
            confidence = max(0, min(100, 100 - (pred_data['rmse'] / current_price * 100)))
            
            fig.add_trace(go.Scatter(
                x=future_days,
                y=predictions,
                mode='lines+markers',
                name=f'{name} (Confidence: {confidence:.1f}%)',
                line=dict(color=colors[i], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f'🔮 Future Price Predictions - Next {days_ahead} Days',
            template='plotly_dark',
            height=400,
            xaxis_title='Days Ahead',
            yaxis_title='Predicted Price (₹)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting future predictions: {str(e)}")
        return go.Figure()

# ========================
# ML Analysis Dashboard
# ========================
def show_ml_predictions(data, stock_symbol, days_ahead=7, prediction_date=None, min_confidence=75):
    """Main ML predictions dashboard with date picker support"""
    
    st.markdown('<h2 style="color: #4facfe; text-align: center; margin: 2rem 0;">🤖 AI-Powered Stock Predictions</h2>', 
                unsafe_allow_html=True)
    
    if data.empty or len(data) < 50:
        st.warning("⚠️ Insufficient data for machine learning analysis. Need at least 50 data points.")
        return
    
    # Add comprehensive error tracking
    try:
        # Display prediction target information
        if prediction_date:
            st.info(f"""
            🎯 **Prediction Target**
            - **Date**: {prediction_date.strftime('%A, %B %d, %Y')}
            - **Days Ahead**: {days_ahead}
            - **Minimum Confidence**: {min_confidence}%
            """)
        
        # Sidebar controls for ML
        with st.sidebar:
            st.markdown("### 🤖 Advanced ML Settings")
            
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            
            # Advanced settings
            with st.expander("⚙️ Model Configuration"):
                include_volume = st.checkbox("Include Volume Features", True)
                include_technical = st.checkbox("Include Technical Indicators", True)
                normalize_features = st.checkbox("Normalize Features", True)
                ensemble_prediction = st.checkbox("Use Ensemble Prediction", True, 
                                                help="Combine all model predictions for better accuracy")
        
        # Create features
        with st.spinner("🔄 Engineering features..."):
            features_df = create_features(data)
            
        if features_df.empty:
            st.error("❌ Failed to create features. Please check your data.")
            return
        
        # Initialize predictor
        predictor = StockPredictor()
        
        # Prepare data
        with st.spinner("📊 Preparing training data..."):
            X_train, X_test, y_train, y_test, scaler, feature_columns = predictor.prepare_data(
                features_df, test_size=test_size
            )
            
        if X_train is None:
            st.error("❌ Failed to prepare training data.")
            return
        
        # Train models
        st.markdown("### 🏋️ Training AI Models")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        models_results = predictor.train_models(X_train, y_train, X_test, y_test, scaler, feature_columns)
        
        progress_bar.progress(100)
        status_text.text("✅ All models trained successfully!")
        
        if not models_results:
            st.error("❌ Model training failed.")
            return
        
        # Filter models by confidence threshold - ROBUST VERSION
        qualified_models = {}
        current_price = float(data['Close'].iloc[-1])
        
        for name, results in models_results.items():
            # Create a completely new dictionary with guaranteed scalar values
            new_result = {}
            
            # Extract and convert all values to basic Python types
            try:
                new_result['test_rmse'] = float(results['test_rmse'])
                new_result['test_mae'] = float(results['test_mae']) 
                new_result['test_r2'] = float(results['test_r2'])
                new_result['test_predictions'] = results['test_predictions']  # Keep as numpy array
                new_result['test_actual'] = results['test_actual']  # Keep as numpy array
                new_result['model'] = results['model']  # Keep model object
                
                # Calculate confidence as basic float
                rmse_val = new_result['test_rmse']
                confidence_val = max(0.0, min(100.0, 100.0 - (rmse_val / current_price * 100.0)))
                new_result['confidence'] = confidence_val
                
                # Copy feature importance if it exists
                if 'feature_importance' in results:
                    new_result['feature_importance'] = results['feature_importance']
                
                # Only add if meets confidence threshold
                if confidence_val >= min_confidence:
                    qualified_models[name] = new_result
                    
            except Exception as e:
                st.warning(f"Error processing model {name}: {str(e)}")
                continue
        
        if not qualified_models:
            st.warning(f"⚠️ No models meet the {min_confidence}% confidence threshold. Showing all results anyway.")
            qualified_models = {}
            
            for name, results in models_results.items():
                # Create new dictionary with guaranteed scalar values
                new_result = {}
                try:
                    new_result['test_rmse'] = float(results['test_rmse'])
                    new_result['test_mae'] = float(results['test_mae'])
                    new_result['test_r2'] = float(results['test_r2'])
                    new_result['test_predictions'] = results['test_predictions']
                    new_result['test_actual'] = results['test_actual']
                    new_result['model'] = results['model']
                    
                    # Calculate confidence
                    rmse_val = new_result['test_rmse']
                    confidence_val = max(0.0, min(100.0, 100.0 - (rmse_val / current_price * 100.0)))
                    new_result['confidence'] = confidence_val
                    
                    # Copy feature importance if it exists
                    if 'feature_importance' in results:
                        new_result['feature_importance'] = results['feature_importance']
                    
                    qualified_models[name] = new_result
                    
                except Exception as e:
                    st.warning(f"Error processing model {name}: {str(e)}")
                    continue
        
        # Display results
        st.markdown("### 📊 Model Performance")
        
        try:
            # Performance metrics table - ULTRA SAFE VERSION
            metrics_data = []
            for name, results in qualified_models.items():
                try:
                    # Extract values with multiple safety checks
                    rmse_val = results.get('test_rmse', 0)
                    if hasattr(rmse_val, 'item'):  # If it's a numpy scalar
                        rmse_val = rmse_val.item()
                    rmse_val = float(rmse_val)
                    
                    mae_val = results.get('test_mae', 0)
                    if hasattr(mae_val, 'item'):
                        mae_val = mae_val.item()
                    mae_val = float(mae_val)
                    
                    r2_val = results.get('test_r2', 0)
                    if hasattr(r2_val, 'item'):
                        r2_val = r2_val.item()
                    r2_val = float(r2_val)
                    
                    confidence_val = results.get('confidence', 0)
                    if hasattr(confidence_val, 'item'):
                        confidence_val = confidence_val.item()
                    confidence_val = float(confidence_val)
                    
                    # Create safe formatted strings
                    rmse_str = f"{rmse_val:.2f}"
                    mae_str = f"{mae_val:.2f}"
                    r2_str = f"{r2_val:.3f}"
                    conf_str = f"{confidence_val:.1f}%"
                    status_str = "✅ Qualified" if confidence_val >= min_confidence else "⚠️ Low Confidence"
                    
                    metrics_data.append({
                        'Model': str(name),
                        'RMSE (₹)': rmse_str,
                        'MAE (₹)': mae_str,
                        'R² Score': r2_str,
                        'Confidence': conf_str,
                        'Status': status_str
                    })
                    
                except Exception as e:
                    st.warning(f"Error formatting metrics for {name}: {str(e)}")
                    # Add a fallback row
                    metrics_data.append({
                        'Model': str(name),
                        'RMSE (₹)': "Error",
                        'MAE (₹)': "Error", 
                        'R² Score': "Error",
                        'Confidence': "Error",
                        'Status': "❌ Error"
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.error("No valid metrics data to display")
                
        except Exception as e:
            st.error(f"Error in metrics table: {str(e)}")
            return
        
        try:
            # Model comparison chart
            fig_comparison = plot_model_comparison(qualified_models)
            st.plotly_chart(fig_comparison, use_container_width=True)
        except Exception as e:
            st.error(f"Error in model comparison chart: {str(e)}")
        
        try:
            # Predictions vs Actual
            st.markdown("### 📈 Predictions vs Actual Values")
            fig_predictions = plot_predictions_vs_actual(qualified_models)
            st.plotly_chart(fig_predictions, use_container_width=True)
        except Exception as e:
            st.error(f"Error in predictions vs actual chart: {str(e)}")
        
        try:
            # Feature importance
            fig_importance = plot_feature_importance(qualified_models)
            if fig_importance:
                st.markdown("### 🔍 Feature Importance Analysis")
                st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.error(f"Error in feature importance chart: {str(e)}")
        
        # Future predictions with date-specific logic
        st.markdown(f"### 🔮 Price Prediction for {prediction_date.strftime('%B %d, %Y') if prediction_date else f'{days_ahead} Days Ahead'}")
        
        current_price = float(data['Close'].iloc[-1])
        
        with st.spinner("🔮 Generating future predictions..."):
            future_predictions = predictor.predict_future(features_df, days_ahead)
        
        if future_predictions:
            # Plot future predictions
            fig_future = plot_future_predictions(current_price, future_predictions, days_ahead)
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Prediction summary with date information
            st.markdown("### 🎯 Prediction Summary")
            
            # Calculate ensemble prediction if enabled
            if ensemble_prediction and len(future_predictions) > 1:
                ensemble_pred = np.mean([float(pred_data['predictions'][-1]) for pred_data in future_predictions.values()])
                
                # Calculate ensemble confidence - now much simpler since confidence is already float
                ensemble_confidence = np.mean([qualified_models[name]['confidence'] for name in future_predictions.keys()])
                
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(0, 242, 254, 0.2));
                    border: 2px solid #4facfe;
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    margin: 2rem 0;
                ">
                    <h2>🏆 Ensemble Prediction</h2>
                    <h1 style="color: #4facfe;">₹{ensemble_pred:.2f}</h1>
                    <p style="color: #4facfe; font-weight: 600; font-size: 1.2rem;">
                        Combined AI Prediction
                    </p>
                    <p style="color: #8892b0;">
                        Average Confidence: {ensemble_confidence:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Individual model predictions
            cols = st.columns(len(future_predictions))
            
            for i, (name, pred_data) in enumerate(future_predictions.items()):
                try:
                    # ULTRA SAFE value extraction
                    predictions_list = pred_data.get('predictions', [])
                    if not predictions_list:
                        continue
                        
                    final_pred = predictions_list[-1]
                    if hasattr(final_pred, 'item'):
                        final_pred = final_pred.item()
                    final_prediction = float(final_pred)
                    
                    change_pct = ((final_prediction - current_price) / current_price) * 100
                    
                    # Get confidence with ultra-safe extraction
                    model_result = qualified_models.get(name, {})
                    conf_val = model_result.get('confidence', 0)
                    if hasattr(conf_val, 'item'):
                        conf_val = conf_val.item()
                    confidence_val = float(conf_val)
                    
                    # Get RMSE safely
                    rmse_val = pred_data.get('rmse', 0)
                    if hasattr(rmse_val, 'item'):
                        rmse_val = rmse_val.item()
                    rmse_val = float(rmse_val)
                    
                    color = "#00ff88" if change_pct >= 0 else "#ff4757"
                    arrow = "📈" if change_pct >= 0 else "📉"
                    
                    # Create safe formatted strings
                    pred_str = f"₹{final_prediction:.2f}"
                    change_str = f"{arrow} {change_pct:+.2f}%"
                    conf_str = f"Confidence: {confidence_val:.1f}%"
                    rmse_str = f"RMSE: ₹{rmse_val:.2f}"
                    
                    with cols[i]:
                        st.markdown(f'''
                        <div style="
                            background: rgba(255, 255, 255, 0.08);
                            border-radius: 15px;
                            padding: 1.5rem;
                            text-align: center;
                            border: 1px solid rgba(255, 255, 255, 0.1);
                            margin: 1rem 0;
                            transition: all 0.3s ease;
                        ">
                            <h3>{name}</h3>
                            <h2 style="color: {color};">{pred_str}</h2>
                            <p style="color: {color}; font-weight: 600;">
                                {change_str}
                            </p>
                            <p style="color: #8892b0;">
                                {conf_str}
                            </p>
                            <p style="color: #8892b0; font-size: 0.8rem;">
                                {rmse_str}
                            </p>
                        </div>
                        ''', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying {name} prediction: {str(e)}")
                    continue
            
            # Prediction export
            st.markdown("### 📥 Export Predictions")
            export_predictions(future_predictions, stock_symbol, days_ahead)
        
        # Best model recommendation
        st.markdown("### 🏆 Model Recommendation")
        
        best_model = min(qualified_models.items(), key=lambda x: x[1]['test_rmse'])
        best_name, best_results = best_model
        
        # Values are already properly typed
        best_rmse = float(best_results['test_rmse'])
        best_r2 = float(best_results['test_r2'])
        best_confidence = float(best_results['confidence'])  # Already stored as float
        best_prediction = float(future_predictions[best_name]['predictions'][-1])
        
        st.success(f"""
        🏆 **Best Performing Model: {best_name}**
        
        - **RMSE**: ₹{best_rmse:.2f}
        - **R² Score**: {best_r2:.3f}
        - **Confidence**: {best_confidence:.1f}%
        - **Prediction for {prediction_date.strftime('%B %d, %Y') if prediction_date else f'{days_ahead} days'}**: 
          ₹{best_prediction:.2f}
        
        This model shows the lowest prediction error and highest reliability for your selected timeframe.
        """)
        
        # Trading recommendation based on predictions
        st.markdown("### 💡 AI Trading Recommendation")
        
        if ensemble_prediction and len(future_predictions) > 1:
            pred_price = float(ensemble_pred)
            avg_confidence = float(ensemble_confidence)
        else:
            pred_price = float(future_predictions[best_name]['predictions'][-1])
            avg_confidence = float(best_results['confidence'])  # Already stored as float
        
        price_change_pct = ((pred_price - current_price) / current_price) * 100
        
        if price_change_pct > 5 and avg_confidence > 70:
            recommendation = "🟢 STRONG BUY"
            reason = f"Models predict {price_change_pct:+.1f}% increase with {avg_confidence:.1f}% confidence"
            rec_color = "#00ff88"
        elif price_change_pct > 2 and avg_confidence > 60:
            recommendation = "🟡 BUY"
            reason = f"Models predict {price_change_pct:+.1f}% increase with {avg_confidence:.1f}% confidence"
            rec_color = "#ffa726"
        elif price_change_pct < -5 and avg_confidence > 70:
            recommendation = "🔴 STRONG SELL"
            reason = f"Models predict {price_change_pct:+.1f}% decrease with {avg_confidence:.1f}% confidence"
            rec_color = "#ff4757"
        elif price_change_pct < -2 and avg_confidence > 60:
            recommendation = "🟡 SELL"
            reason = f"Models predict {price_change_pct:+.1f}% decrease with {avg_confidence:.1f}% confidence"
            rec_color = "#ff6b6b"
        else:
            recommendation = "🟡 HOLD"
            reason = f"Models predict {price_change_pct:+.1f}% change with moderate confidence"
            rec_color = "#ffa726"
        
        st.markdown(f'''
        <div style="
            background: rgba(255, 255, 255, 0.08);
            border: 2px solid {rec_color};
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            margin: 2rem 0;
        ">
            <h2 style="color: {rec_color};">{recommendation}</h2>
            <p style="font-size: 1.1rem; margin: 1rem 0;">{reason}</p>
            <p style="color: #8892b0; font-size: 0.9rem;">
                Target Date: {prediction_date.strftime('%B %d, %Y') if prediction_date else f'{days_ahead} days from now'}
            </p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Risk disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Important Disclaimer**: 
        - These predictions are based on historical data and technical indicators
        - Past performance does not guarantee future results
        - Market conditions can change rapidly and unpredictably
        - Always do your own research and consider multiple factors before making investment decisions
        - Consider consulting with a qualified financial advisor
        - Never invest more than you can afford to lose
        """)
        
    except Exception as e:
        st.error(f"❌ Error in ML analysis: {str(e)}")
        st.info("Please try refreshing the page or selecting a different stock.")
        # Add debug information
        with st.expander("🔍 Debug Information"):
            st.text(f"Error details: {str(e)}")
            st.text(f"Data shape: {data.shape}")
            st.text(f"Data columns: {list(data.columns)}")
            st.text(f"Days ahead: {days_ahead}")
            st.text(f"Prediction date: {prediction_date}")

# ========================
# Model Explanation
# ========================
def explain_ml_models():
    """Explain the ML models used"""
    
    st.markdown("### 🧠 Understanding Our AI Models")
    
    with st.expander("🌲 Random Forest - How it works"):
        st.markdown("""
        **Random Forest** combines multiple decision trees to make predictions:
        
        - **Strengths**: Handles non-linear relationships, resistant to overfitting
        - **Best for**: Complex patterns in stock data
        - **Features used**: Price movements, technical indicators, volume data
        
        Each tree votes on the prediction, and the average becomes the final result.
        """)
    
    with st.expander("🚀 Gradient Boosting - How it works"):
        st.markdown("""
        **Gradient Boosting** builds models sequentially, each correcting errors of the previous:
        
        - **Strengths**: High accuracy, learns from mistakes
        - **Best for**: Capturing subtle price trends
        - **Features used**: All available technical indicators
        
        Each new model focuses on the hardest-to-predict cases.
        """)
    
    with st.expander("📈 Linear Regression - How it works"):
        st.markdown("""
        **Linear Regression** finds the best straight line through the data:
        
        - **Strengths**: Simple, interpretable, fast
        - **Best for**: Identifying overall trends
        - **Features used**: Moving averages, price ratios
        
        Works well when relationships between features and price are linear.
        """)

# ========================
# Export Functions
# ========================
def export_predictions(future_predictions, stock_symbol, days_ahead):
    """Export predictions to downloadable format"""
    
    try:
        export_data = []
        
        for model_name, pred_data in future_predictions.items():
            for day, prediction in enumerate(pred_data['predictions'], 1):
                export_data.append({
                    'Stock': stock_symbol,
                    'Model': model_name,
                    'Day': day,
                    'Predicted_Price': round(prediction, 2),
                    'RMSE': round(pred_data['rmse'], 2),
                    'R2_Score': round(pred_data['r2'], 3),
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        export_df = pd.DataFrame(export_data)
        
        # Create CSV download
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"{stock_symbol}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting predictions: {str(e)}")