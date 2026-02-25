import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from typing import Any

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH_TEMPLATE = os.path.join(RESULT_DIR, "{target}_model.pkl")

def get_model_path(target: str = "alpha"):
    return MODEL_PATH_TEMPLATE.format(target=target)

def save_model(model: Any, path: str):
    joblib.dump(model, path)

def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def create_default_model(**kwargs):
    defaults = {'n_estimators': 200, 'max_depth': 8, 'random_state': 42}
    defaults.update(kwargs)
    return RandomForestRegressor(**defaults)