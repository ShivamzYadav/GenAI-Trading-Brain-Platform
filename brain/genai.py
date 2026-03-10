import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Gemini
try:
    import google.genai as genai
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        model = "gemini-2.0-flash"
        logger.info("Gemini API initialized successfully")
    else:
        logger.warning("GEMINI_API_KEY not set. Using rule-based decisions.")
        logger.info("Set GEMINI_API_KEY in .env file: https://aistudio.google.com/app/apikey")
        model = None
except ImportError as e:
    logger.warning(f"google-genai not installed: {e}")
    logger.info("Install with: pip install google-genai")
    model = None
except Exception as e:
    logger.warning(f"Gemini initialization failed: {e}. Using rule-based decisions.")
    model = None

def ask_llm(signal, rsi, volatility):
    """Get trading decision from Google Gemini LLM"""
    # Extract values
    signal_val = signal
    rsi_val = float(rsi.iloc[0]) if hasattr(rsi, 'iloc') else float(rsi)
    volatility_val = float(volatility.iloc[0]) if hasattr(volatility, 'iloc') else float(volatility)
    
    # If no API key, use rule-based fallback
    if model is None:
        return _rule_based_decision(signal_val, rsi_val, volatility_val)
    
    try:
        prompt = f"""You are a professional stock trader AI. Analyze the following market indicators and provide a trading decision.

MARKET DATA:
- ML Signal: {signal_val}
- RSI (Relative Strength Index): {rsi_val:.2f}
- Volatility: {volatility_val:.4f}

Based on these indicators, provide your trading decision in this EXACT format:
Decision: [BUY/SELL/HOLD]
Confidence: [HIGH/MEDIUM/LOW]
Reason: [Brief explanation]

Consider:
- ML Signal accuracy
- RSI overbought (>70) / oversold (<30) conditions
- Volatility risk levels
- Risk/reward ratio"""
        
        response = genai.models.generate_content(model=model, contents=prompt)
        decision_text = response.text.strip()
        logger.info(f"Gemini Decision: {decision_text}")
        return decision_text
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return _rule_based_decision(signal_val, rsi_val, volatility_val)

def _rule_based_decision(signal_val, rsi_val, volatility_val):
    """Fallback rule-based decision system"""
    if signal_val == "BUY" and rsi_val < 70 and volatility_val < 0.05:
        decision = "BUY"
        confidence = "HIGH"
        reason = f"BUY signal with healthy RSI ({rsi_val:.2f}) and low volatility"
    elif signal_val == "SELL" and rsi_val > 30:
        decision = "SELL"
        confidence = "MEDIUM"
        reason = f"SELL signal confirmed by RSI ({rsi_val:.2f})"
    else:
        decision = "HOLD"
        confidence = "LOW"
        reason = f"Conditions not favorable: Signal={signal_val}, RSI={rsi_val:.2f}, Vol={volatility_val:.4f}"
    
    return f"Decision: {decision}\nConfidence: {confidence}\nReason: {reason}"
