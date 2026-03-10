# GenAI Trading Brain

A comprehensive AI-powered trading platform with manual and automated trading capabilities, featuring machine learning-driven signals and GenAI integration for intelligent trading decisions.

## 🚀 Features

- **AI-Driven Trading Signals**: Machine learning models for predicting market movements
- **Manual Trading Interface**: Interactive Streamlit dashboard for manual trade execution
- **Automated Trading**: Background automation for hands-free trading
- **Real-Time Market Data**: Integration with market data sources for NIFTY50 stocks
- **Technical Indicators**: Comprehensive set of technical analysis indicators
- **Portfolio Management**: Track portfolio statistics and trading history
- **GenAI Integration**: Leverage generative AI for trading insights and analysis
- **Paper Trading**: Safe testing environment with simulated trades

## 🏗️ Architecture

- **Backend**: FastAPI server providing REST API endpoints for all trading operations
- **Frontend**: Streamlit web application for user-friendly trading interface
- **Data Layer**: Market data downloading and universe management
- **Features Engine**: Technical indicators calculation and analysis
- **ML Engine**: Model training, prediction, and signal generation
- **Execution Engine**: Paper trading and automated execution systems

## 📋 Prerequisites

- Python 3.8+
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Trading_brain
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Start the FastAPI backend server:**
   ```bash
   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

3. **Access the application:**
   - **Dashboard**: Open `http://localhost:8501` in your browser
   - **API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs

## 📁 Project Structure

```
Trading_brain/
├── api_server.py          # FastAPI backend server
├── app.py                  # Streamlit frontend application
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── brain/                 # GenAI integration
│   ├── genai.py
│   └── __init__.py
├── data/                  # Market data management
│   ├── downloader.py
│   ├── universe.py
│   └── __init__.py
├── execution/             # Trading execution
│   ├── automation.py
│   ├── paper_trader.py
│   └── __init__.py
├── features/              # Technical indicators
│   ├── indicators.py
│   └── __init__.py
└── ml/                    # Machine learning
    ├── model.json
    ├── predict.py
    ├── train.py
    └── __init__.py
```

## 🔧 Configuration

Edit `config.py` to customize:
- API settings
- Market data sources
- Model parameters
- Trading thresholds

## 📊 API Endpoints

The FastAPI server provides endpoints for:
- Health checks and status
- Manual trading operations
- Portfolio statistics
- Automation control
- Model training and prediction
- GenAI queries

## 🤖 Machine Learning

- **Training**: Use `ml/train.py` to train models on historical data
- **Prediction**: Real-time signal generation for trading decisions
- **Model Storage**: Models saved in JSON format for portability

## ⚠️ Disclaimer

This is a trading simulation platform for educational and research purposes. Not intended for real financial trading. Always consult with financial advisors before making investment decisions.

## 📄 License

[Add your license information here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For questions or issues, please open a GitHub issue or contact the maintainers.