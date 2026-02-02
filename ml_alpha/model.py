import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from typing import Any

MODEL_PATH = os.path.join(os.path.dirname(__file__), "alpha_model.pkl")

def save_model(model: Any, path: str = MODEL_PATH):
    joblib.dump(model, path)

def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def create_default_model(**kwargs):
    return RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, **kwargs)