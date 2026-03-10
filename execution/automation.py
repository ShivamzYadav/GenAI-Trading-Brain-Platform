"""
Automated Trading Module
Handles scheduled trading based on predefined parameters
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logger = logging.getLogger(__name__)

# Global automation state
automation_state = {
    "enabled": False,
    "stocks": [],
    "interval_minutes": 5,
    "is_running": False,
    "last_run": None,
    "executions": [],
    "next_run": None,
    "thread": None
}

def start_automation(stocks: List[str], interval_minutes: int = 5, callback=None):
    """Start automated trading"""
    global automation_state
    
    try:
        automation_state["enabled"] = True
        automation_state["stocks"] = stocks
        automation_state["interval_minutes"] = interval_minutes
        automation_state["is_running"] = True
        automation_state["executions"] = []
        
        logger.info(f"Starting automation for stocks: {stocks} every {interval_minutes} minutes")
        
        # Start background thread
        if automation_state["thread"] is None or not automation_state["thread"].is_alive():
            automation_state["thread"] = threading.Thread(
                target=_automation_loop,
                args=(stocks, interval_minutes, callback),
                daemon=True
            )
            automation_state["thread"].start()
        
        return {
            "status": "started",
            "stocks": stocks,
            "interval_minutes": interval_minutes,
            "message": f"Automation started for {len(stocks)} stocks"
        }
    except Exception as e:
        logger.error(f"Error starting automation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def stop_automation():
    """Stop automated trading"""
    global automation_state
    
    try:
        automation_state["enabled"] = False
        automation_state["is_running"] = False
        logger.info("Stopping automation")
        
        return {
            "status": "stopped",
            "message": "Automation stopped",
            "total_executions": len(automation_state["executions"])
        }
    except Exception as e:
        logger.error(f"Error stopping automation: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_automation_status():
    """Get current automation status"""
    return {
        "enabled": automation_state["enabled"],
        "is_running": automation_state["is_running"],
        "stocks": automation_state["stocks"],
        "interval_minutes": automation_state["interval_minutes"],
        "last_run": automation_state["last_run"],
        "next_run": automation_state["next_run"],
        "total_executions": len(automation_state["executions"]),
        "executions": automation_state["executions"][-10:]  # Last 10 executions
    }

def get_execution_history():
    """Get automation execution history"""
    return automation_state["executions"]

def _automation_loop(stocks: List[str], interval_minutes: int, callback=None):
    """Background loop for automated trading"""
    from data.downloader import fetch
    from features.indicators import add_indicators
    from ml.predict import signal
    from brain.genai import ask_llm
    from execution.paper_trader import trade
    
    while automation_state["enabled"]:
        try:
            if automation_state["is_running"]:
                execution_time = datetime.now()
                automation_state["last_run"] = execution_time.isoformat()
                
                # Calculate next run time
                next_run = execution_time + timedelta(minutes=interval_minutes)
                automation_state["next_run"] = next_run.isoformat()
                
                logger.info(f"Running automation at {execution_time}")
                
                # Execute analysis for each stock
                for stock in stocks:
                    try:
                        # Fetch and analyze
                        df = fetch(stock)
                        if df is None or df.empty:
                            continue
                        
                        df = add_indicators(df)
                        if df is None or df.empty:
                            continue
                        
                        # Get signal
                        sig = signal(df)
                        last = df.iloc[-1]
                        
                        # Get AI decision
                        ai_decision = ask_llm(sig, last["rsi"], last["volatility"])
                        
                        # Execute trade
                        trade_action = trade(last["Close"], ai_decision)
                        
                        execution = {
                            "timestamp": execution_time.isoformat(),
                            "stock": stock,
                            "signal": sig,
                            "rsi": float(last["rsi"].iloc[0]) if hasattr(last["rsi"], 'iloc') else float(last["rsi"]),
                            "volatility": float(last["volatility"].iloc[0]) if hasattr(last["volatility"], 'iloc') else float(last["volatility"]),
                            "current_price": float(last["Close"].iloc[0]) if hasattr(last["Close"], 'iloc') else float(last["Close"]),
                            "ai_decision": ai_decision,
                            "trade_action": trade_action,
                            "status": "success"
                        }
                        automation_state["executions"].append(execution)
                        logger.info(f"Automated trade executed for {stock}: {trade_action}")
                        
                        if callback:
                            callback(stock, execution)
                            
                    except Exception as e:
                        logger.error(f"Error processing {stock}: {e}")
                        error_execution = {
                            "timestamp": execution_time.isoformat(),
                            "stock": stock,
                            "status": "error",
                            "message": str(e)
                        }
                        automation_state["executions"].append(error_execution)
                
                # Keep only last 100 executions in memory
                if len(automation_state["executions"]) > 100:
                    automation_state["executions"] = automation_state["executions"][-100:]
            
            # Sleep for the specified interval
            time.sleep(interval_minutes * 60)
            
        except Exception as e:
            logger.error(f"Error in automation loop: {e}")
            time.sleep(interval_minutes * 60)

def clear_execution_history():
    """Clear execution history"""
    automation_state["executions"] = []
    return {"status": "cleared", "message": "Execution history cleared"}
