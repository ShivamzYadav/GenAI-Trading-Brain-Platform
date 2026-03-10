"""
FastAPI Backend Server for GenAI Trading Brain
Handles all trading operations and provides HTTP API endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.universe import NIFTY50
from data.downloader import fetch
from features.indicators import add_indicators
from ml.train import train, get_model_accuracy, get_model_metrics
from ml.predict import signal
from brain.genai import ask_llm
from execution.paper_trader import trade, get_portfolio_stats, trade_history
from execution.automation import start_automation, stop_automation, get_automation_status, get_execution_history, clear_execution_history
from utils import is_market_hours, is_weekday

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GenAI Trading Brain API",
    description="API for manual AI-driven trading",
    version="1.0.0"
)

# Pydantic models for request/response
class TradeRequest(BaseModel):
    stock: str
    train_mode: Optional[bool] = False

class AutomationRequest(BaseModel):
    stocks: list[str]
    interval_minutes: Optional[int] = 5

# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """API root endpoint"""
    return {
        "name": "GenAI Trading Brain API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Check API health and market status"""
    try:
        return {
            "status": "healthy",
            "market_hours": is_market_hours() and is_weekday()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/accuracy", tags=["Model"])
async def get_accuracy_endpoint():
    """Get current model accuracy and metrics"""
    try:
        accuracy = get_model_accuracy()
        metrics = get_model_metrics()
        
        if accuracy is None:
            return {
                "accuracy": None,
                "precision": None,
                "confusion_matrix": None,
                "status": "Model not trained yet",
                "message": "Train the model first by calling /analyze with train_mode=true"
            }
        return {
            "accuracy": float(accuracy),
            "accuracy_percentage": f"{accuracy*100:.2f}%",
            "precision": metrics.get("precision"),
            "precision_percentage": f"{metrics.get('precision')*100:.2f}%" if metrics.get("precision") else None,
            "confusion_matrix": metrics.get("confusion_matrix"),
            "tn": metrics.get("tn"),
            "fp": metrics.get("fp"),
            "fn": metrics.get("fn"),
            "tp": metrics.get("tp"),
            "status": "Model trained"
        }
    except Exception as e:
        logger.error(f"Error getting model accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Stock & Portfolio Endpoints
# ============================================================================

@app.get("/stocks", tags=["Stocks"])
async def get_stocks():
    """Get list of available stocks (NIFTY-50)"""
    try:
        return {"stocks": NIFTY50}
    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio", tags=["Portfolio"])
async def get_portfolio(current_price: Optional[float] = None):
    """Get current portfolio statistics"""
    try:
        return get_portfolio_stats(current_price)
    except Exception as e:
        logger.error(f"Error getting portfolio stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades", tags=["Portfolio"])
async def get_trades():
    """Get trade history"""
    try:
        return trade_history
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Manual Trading Endpoints
# ============================================================================

@app.post("/analyze", tags=["Manual Trading"])
async def analyze_stock(request: TradeRequest):
    """Analyze a single stock and return trading decision"""
    try:
        stock = request.stock
        logger.info(f"Analyzing stock: {stock}")
        
        # Fetch data
        df = fetch(stock)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"No data available for {stock}")
        
        # Add indicators
        df = add_indicators(df)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail=f"Failed to add indicators for {stock}")
        
        # Train model only if requested
        if request.train_mode:
            train(df)
            logger.info(f"Model trained for {stock}")
        else:
            logger.info(f"Using saved model for {stock}")
        #print(df.head())
        # Get signal
        sig = signal(df)
        last = df.iloc[-1]

        
        # Get AI decision
        ai_decision = ask_llm(sig, last["rsi"], last["volatility"])
        
        # Execute trade
        trade_action = trade(last["Close"], ai_decision)
        
        return {
            "stock": stock,
            "signal": sig,
            "rsi": float(last["rsi"].iloc[0]) if hasattr(last["rsi"], 'iloc') else float(last["rsi"]),
            "volatility": float(last["volatility"].iloc[0]) if hasattr(last["volatility"], 'iloc') else float(last["volatility"]),
            "current_price": float(last["Close"].iloc[0]) if hasattr(last["Close"], 'iloc') else float(last["Close"]),
            "ai_decision": ai_decision,
            "trade_action": trade_action,
            "train_mode": request.train_mode
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing stock {request.stock}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Automated Trading Endpoints
# ============================================================================

@app.post("/automation/start", tags=["Automation"])
async def start_automation_trading(request: AutomationRequest):
    """Start automated trading for multiple stocks"""
    try:
        result = start_automation(request.stocks, request.interval_minutes)
        logger.info(f"Automation started for {request.stocks}")
        return result
    except Exception as e:
        logger.error(f"Error starting automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/automation/stop", tags=["Automation"])
async def stop_automation_endpoint():
    """Stop automated trading"""
    try:
        result = stop_automation()
        logger.info("Automation stopped")
        return result
    except Exception as e:
        logger.error(f"Error stopping automation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automation/status", tags=["Automation"])
async def get_automation_status_endpoint():
    """Get automation status"""
    try:
        return get_automation_status()
    except Exception as e:
        logger.error(f"Error getting automation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/automation/history", tags=["Automation"])
async def get_automation_history_endpoint():
    """Get automation execution history"""
    try:
        return {"executions": get_execution_history()}
    except Exception as e:
        logger.error(f"Error getting automation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/automation/clear", tags=["Automation"])
async def clear_automation_history_endpoint():
    """Clear automation execution history"""
    try:
        result = clear_execution_history()
        logger.info("Automation history cleared")
        return result
    except Exception as e:
        logger.error(f"Error clearing automation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
