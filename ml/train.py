import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import json

logger = logging.getLogger(__name__)
model_accuracy = None
model_metrics = {
    "accuracy": None,
    "precision": None,
    "confusion_matrix": None,
    "tn": None,
    "fp": None,
    "fn": None,
    "tp": None
}

def train(df):
    """Train XGBoost model and save"""
    global model_accuracy, model_metrics
    try:
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        X = df[["ema", "rsi", "volatility"]]
        y = df["target"]
        
        # Split data for accuracy calculation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        model_metrics["accuracy"] = float(model_accuracy)
        model_metrics["precision"] = float(precision)
        model_metrics["confusion_matrix"] = cm.tolist()
        model_metrics["tn"] = int(cm[0, 0])
        model_metrics["fp"] = int(cm[0, 1])
        model_metrics["fn"] = int(cm[1, 0])
        model_metrics["tp"] = int(cm[1, 1])
        
        logger.info(f"Model trained with accuracy: {model_accuracy:.4f}, precision: {precision:.4f}")
        logger.info(f"Confusion Matrix:\nTN: {cm[0,0]}, FP: {cm[0,1]}\nFN: {cm[1,0]}, TP: {cm[1,1]}")

        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "model.json")
        model.get_booster().save_model(model_path)
        logger.info(f"Model trained and saved")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None

def get_model_accuracy():
    """Get current model accuracy"""
    return model_accuracy

def get_model_metrics():
    """Get all model metrics"""
    return model_metrics
