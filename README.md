# 📊 Advanced Stock Predictor AI

## 🚀 **Project Overview**

Advanced Stock Predictor AI is a sophisticated machine learning-powered web application built with Streamlit that provides real-time stock analysis, technical indicators, and AI-driven price predictions for Indian penny stocks. The application combines traditional technical analysis with cutting-edge machine learning algorithms to deliver comprehensive investment insights.

![Main Dashboard](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12.3-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ **Key Features**

### 🎯 **Core Functionality**
- **Real-time Stock Data**: Live market data integration using Yahoo Finance API
- **Technical Analysis**: Advanced indicators including RSI, MACD, Bollinger Bands, and Moving Averages
- **AI Predictions**: Multiple machine learning models for price forecasting
- **Interactive Charts**: Dynamic, responsive visualizations with Plotly
- **User Authentication**: Secure login system with demo and registered user support
- **Market Overview**: Real-time market snapshot with key metrics
- **REST API**: FastAPI endpoints for programmatic access to stock data and predictions
- **AI Insights**: Multi-provider AI integration (Ollama, OpenRouter, Gemini, OpenAI) for intelligent market analysis

### 🤖 **AI & Machine Learning**
- **Random Forest Regressor**: Ensemble learning for robust predictions
- **Gradient Boosting**: Sequential learning for capturing complex patterns
- **Linear Regression**: Baseline model for trend analysis
- **Feature Engineering**: 20+ technical indicators and derived features
- **Model Confidence Scoring**: Reliability metrics for each prediction
- **Ensemble Predictions**: Combined model outputs for enhanced accuracy

### 📈 **Technical Analysis Tools**
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **MACD**: Trend-following momentum indicator
- **Moving Averages**: SMA and EMA for trend identification
- **Bollinger Bands**: Volatility and price level analysis
- **Volume Analysis**: Trading volume patterns and trends
- **Price Patterns**: Support/resistance levels and trend detection

## 🛠️ **Technology Stack**

| Category | Technology | Version |
|----------|------------|---------|
| **Backend** | Python | 3.12.3 |
| **Web Framework** | Streamlit | 1.32.0 |
| **API Framework** | FastAPI | Latest |
| **Data Processing** | Pandas | 2.2.2 |
| **Numerical Computing** | NumPy | 1.26.4 |
| **Machine Learning** | Scikit-learn | 1.4.2 |
| **Data Visualization** | Plotly | 5.22.0 |
| **Market Data** | yfinance | 0.2.65 |
| **AI Integration** | OpenAI, Gemini, Ollama | Latest |
| **HTTP Client** | httpx | Latest |
| **PDF Generation** | ReportLab | 4.4.4 |
| **Documentation** | LaTeX (research paper) | - |
| **UI Styling** | Custom CSS | - |

## 📦 **Installation & Setup**

### **Prerequisites**
- Python 3.12+ 
- Anaconda or Miniconda (recommended)
- Git

### **Quick Start**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/sharath2004-tech/predictor.git
   cd predictor
   ```

2. **Create Virtual Environment**
   
   Using Conda:
   ```bash
   conda create -n stock-predictor python=3.12
   conda activate stock-predictor
   ```
   
   Or using venv (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run main.py
   ```

5. **Configure Environment (Optional)**
   ```bash
   # Create .env file for AI provider API keys
   OPENROUTER_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   OLLAMA_HOST=http://127.0.0.1:11434  # For local Ollama
   ```

6. **Access the App**
   Open your browser and navigate to `http://localhost:8501`

## 🏗️ **Project Structure**

```
📦 Advanced Stock Predictor AI
├── 📄 main.py                    # Main Streamlit application
├── 📄 ml_predictor.py            # Machine learning models and predictions
├── 📄 ml_predictor_fixed.py      # Optimized ML predictor implementation
├── 📄 login.py                   # User authentication system
├── 📄 api.py                     # FastAPI REST endpoints
├── 📄 take_screenshots.py        # Screenshot automation utility
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project documentation
├── 📄 LICENSE                    # MIT license
├── 📄 SECURITY.md                # Security policies
├── 📁 services/                  # AI provider integrations
│   ├── 📄 ollama_client.py      # Ollama local AI client
│   ├── 📄 openrouter_client.py  # OpenRouter API client
│   ├── 📄 gemini_client.py      # Google Gemini AI client
│   └── 📄 openai_client.py      # OpenAI API client
├── 📁 docs/                      # Documentation
│   ├── 📄 README.md             # Documentation index
│   ├── 📄 API_DOCUMENTATION.md   # API reference
│   ├── 📄 DEPLOYMENT_GUIDE.md    # Deployment instructions
│   ├── 📄 DEVELOPER_GUIDE.md     # Developer setup guide
│   ├── 📄 SECURITY_GUIDE.md      # Security best practices
│   ├── 📄 USER_GUIDE.md          # End-user manual
│   └── 📄 FAQ_and_Features.pdf   # Comprehensive Q&A and features PDF
├── 📁 research_paper/            # Academic research paper (LaTeX)
│   ├── 📄 main.tex              # LaTeX source document
│   ├── 📄 SASBEFRBDTSC.sty      # Custom LaTeX style file
│   ├── 📄 references.bib        # BibTeX bibliography
│   ├── 📄 references.ris        # RIS format references
│   └── 📄 README.md             # Compilation instructions
├── 📁 screenshots/               # Application screenshots
├── 📁 scripts/                   # Utility scripts
│   └── 📄 generate_faq_pdf.py   # PDF documentation generator
├── 📁 .vscode/                   # VS Code configuration
│   └── 📄 settings.json         # Python interpreter settings
├── 📁 .git/                      # Git repository
└── 📁 .venv/                     # Virtual environment
```

## 🎨 **Application Screenshots**

### 🏠 **Main Dashboard**
![Main Dashboard](screenshots/main-dashboard.png)
*Real-time stock metrics with animated gradient background and glass-morphism design*

### 📊 **Market Overview**
![Market Overview](screenshots/market-overview.png)
*Comprehensive market analysis with real-time data and key performance indicators*

### 📈 **Technical Analysis**
![Technical Analysis](screenshots/technical-analysis.png) 
*Interactive candlestick charts with moving averages, RSI, MACD, and volume analysis*

### 🤖 **AI Predictions**
![AI Predictions](screenshots/ai-predictions.png)
*Machine learning model performance comparison and future price predictions with confidence metrics*

## 🔧 **Configuration**

### **Supported Stocks**
The application currently supports these Indian penny stocks:
- YESBANK.NS (Yes Bank)
- SUZLON.NS (Suzlon Energy)
- PNB.NS (Punjab National Bank)
- IDEA.NS (Vodafone Idea)
- RPOWER.NS (Reliance Power)
- JPPOWER.NS (Jaiprakash Power)
- IRFC.NS (Indian Railway Finance)
- ONGC.NS (Oil and Natural Gas)
- IOB.NS (Indian Overseas Bank)
- TATAPOWER.NS (Tata Power)

### **Customization Options**
- **Technical Indicators**: Adjustable periods for RSI, MACD, and moving averages
- **Chart Timeframes**: 3 months to 2 years of historical data
- **ML Model Settings**: Confidence thresholds and ensemble options
- **Prediction Horizons**: 1 to 365 days ahead forecasting

## 🚀 **Usage Guide**

### **Getting Started**
1. **Launch the Application**: Run `streamlit run main.py`
2. **Login**: 
   - **Demo Mode**: Click "Try Demo Mode" for instant access (no registration required)
   - **Firebase Auth**: Register with email/password for personalized experience
3. **Select Stock**: Choose from the dropdown list of supported stocks
4. **Configure Analysis**: Set technical indicator parameters and prediction settings
5. **View AI Insights**: Ask questions about stocks using integrated AI providers
6. **View Results**: Analyze charts, technical indicators, and AI predictions

### **Key Features Walkthrough**

#### **📊 Technical Analysis**
- View real-time candlestick charts with volume
- Analyze RSI for momentum and overbought/oversold conditions
- Monitor MACD for trend changes and momentum shifts
- Track moving averages for trend identification

#### **🤖 AI Predictions**
- Train multiple ML models on historical data
- Compare model performance metrics (RMSE, R², confidence)
- Generate future price predictions with confidence intervals
- View feature importance for model interpretability
- Get trading recommendations based on AI analysis

#### **⚙️ Advanced Settings**
- Adjust technical indicator parameters
- Set prediction timeframes and confidence thresholds
- Enable/disable ensemble predictions
- Configure chart display options

## 📊 **Machine Learning Models**

### **Model Architecture**

| Model | Type | Features | Strengths |
|-------|------|----------|-----------|
| **Random Forest** | Ensemble | 20+ technical indicators | Handles non-linearity, robust to overfitting |
| **Gradient Boosting** | Sequential | Price patterns, volume data | High accuracy, learns from errors |
| **Linear Regression** | Linear | Moving averages, ratios | Fast, interpretable, good baseline |

### **Feature Engineering**
- **Price Features**: Open, High, Low, Close, Volume
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Momentum
- **Moving Averages**: SMA(5,10,20,50), EMA(12,26)
- **Derived Features**: Price ratios, volatility, lag features
- **Time Features**: Day of week, month seasonality

### **Model Evaluation**
- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAE**: Mean Absolute Error for average prediction deviation  
- **R² Score**: Coefficient of determination for model fit quality
- **Confidence Score**: Custom metric based on prediction reliability

## 🎨 **Design Philosophy**

### **User Experience**
- **Modern UI**: Glass-morphism design with animated gradients
- **Responsive Layout**: Optimized for desktop and mobile viewing
- **Interactive Elements**: Real-time updates and smooth transitions
- **Color Psychology**: Strategic use of colors for different signal types
- **Accessibility**: High contrast ratios and clear visual hierarchy

### **Visual Design Elements**
- **Gradient Backgrounds**: Animated color transitions
- **Glass Cards**: Translucent containers with backdrop blur
- **Signal Colors**: Green (bullish), Red (bearish), Orange (neutral)
- **Typography**: Poppins font for modern, readable text
- **Icons**: Contextual emojis for intuitive navigation

## ⚠️ **Disclaimer & Risk Warning**

> **IMPORTANT**: This application is designed for educational and research purposes only. 
> 
> **Investment Risks:**
> - Past performance does not guarantee future results
> - Stock market investments carry inherent risks
> - AI predictions are based on historical patterns and may not account for unforeseen events
> - Always conduct your own research and consult with qualified financial advisors
> - Never invest more than you can afford to lose
> 
> **Model Limitations:**
> - Predictions are probabilistic, not certainties
> - Market volatility can exceed model expectations
> - External factors (news, events, policy changes) may impact accuracy
> - Model performance may vary across different market conditions

## 🤝 **Contributing**

We welcome contributions to improve the Advanced Stock Predictor AI! Here's how you can help:

### **Development Setup**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

### **Areas for Contribution**
- **New ML Models**: Implement LSTM, Prophet, or other time series models
- **Additional Indicators**: Add new technical analysis tools
- **Market Expansion**: Support for international markets
- **Performance Optimization**: Improve computational efficiency
- **UI/UX Enhancements**: Design improvements and new features
- **Testing**: Unit tests and integration tests
- **Documentation**: Improve code documentation and user guides

### **Code Style**
- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Comment complex algorithms and business logic

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Advanced Stock Predictor AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📚 **Documentation**

### **Available Documentation**
- **[API Documentation](docs/API_DOCUMENTATION.md)**: REST API endpoints and usage
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Production deployment instructions
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Setup and development workflow
- **[Security Guide](docs/SECURITY_GUIDE.md)**: Security best practices
- **[User Guide](docs/USER_GUIDE.md)**: End-user manual and tutorials
- **[FAQ & Features PDF](docs/FAQ_and_Features.pdf)**: Comprehensive Q&A and unique features
- **[Research Paper](research_paper/)**: Academic paper with LaTeX source

### **Research Paper**
A comprehensive academic research paper is available in the `research_paper/` directory, documenting:
- System architecture and methodology
- Machine learning model evaluation and results
- Feature engineering and technical analysis approach
- Performance metrics and validation
- Complete LaTeX source with BibTeX references

Compile the paper:
```powershell
cd research_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 📞 **Support & Contact**

- **Issues**: Report bugs or request features via [GitHub Issues](https://github.com/sharath2004-tech/predictor/issues)
- **Discussions**: Join the community discussions on [GitHub Discussions](https://github.com/sharath2004-tech/predictor/discussions)
- **Documentation**: Explore comprehensive guides in the `docs/` folder

## 🔮 **Roadmap**

### **Version 2.0 (Planned)**
- [ ] Real-time WebSocket data feeds
- [ ] Advanced LSTM neural networks
- [ ] Sentiment analysis from news/social media
- [ ] Portfolio optimization tools
- [ ] Risk management dashboard
- [ ] Mobile app development
- [ ] Enhanced AI conversation history and context retention
- [ ] Multi-stock portfolio comparison dashboard

### **Version 2.5 (Future)**
- [ ] Multi-asset support (crypto, forex, commodities)
- [ ] Algorithmic trading integration
- [ ] Advanced backtesting framework
- [ ] Cloud deployment options
- [ ] API for external integrations

## 🙏 **Acknowledgments**

- **Yahoo Finance**: For providing free market data API
- **Streamlit Team**: For the amazing web app framework
- **scikit-learn**: For robust machine learning tools
- **Plotly**: For interactive visualization capabilities
- **Open Source Community**: For the incredible tools and libraries that make this project possible

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**🍴 Fork it to create your own version!**

**🐛 Report issues to help us improve!**

</div>

---

## 🔑 **API Integration**

The application includes a FastAPI backend (`api.py`) with the following endpoints:

- **GET /health**: Health check endpoint
- **GET /stock/{symbol}**: Retrieve historical stock data
- **GET /trends**: Get market trend analysis with AI insights
- **GET /trends/simple**: Simplified trend snapshot
- **POST /ask**: AI-powered query endpoint

Run the API server:
```bash
uvicorn api:app --reload --port 8000
```

Access API docs at: `http://localhost:8000/docs`

**Note**: The FastAPI server can run alongside the Streamlit app for programmatic access to stock data and predictions.

## 🤖 **AI Provider Configuration**

The system supports multiple AI providers for intelligent market analysis:

### **Supported Providers**
1. **Ollama** (Local): Free, privacy-focused, runs on your machine
   - Install: [ollama.ai](https://ollama.ai)
   - Run: `ollama run llama2` or your preferred model
   
2. **OpenRouter**: Access to multiple AI models via API
   - Get API key: [openrouter.ai](https://openrouter.ai)
   
3. **Google Gemini**: Google's advanced AI models
   - Get API key: [Google AI Studio](https://makersuite.google.com/app/apikey)
   
4. **OpenAI**: GPT models for analysis
   - Get API key: [OpenAI Platform](https://platform.openai.com)

Configure via `.env` file or Streamlit secrets.

---

*Last updated: November 14, 2025*
