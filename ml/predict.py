from typing import Dict, Any
from .model import load_model, get_model_path
from .train import PRE_ALPHA_FEATURES
import pandas as pd


def predict_target(feature_row: Dict[str, Any], target: str = "alpha"):
    model_path = get_model_path(target)
    model = load_model(model_path)
    if model is None:
        raise RuntimeError(f"No trained model found for target '{target}'. Run ml.train.train_from_csv(target='{target}') first.")
    # Use pandas DataFrame to preserve feature names (matches training)
    x = pd.DataFrame([[feature_row.get(n, 0.0) for n in PRE_ALPHA_FEATURES]], columns=PRE_ALPHA_FEATURES)
    val = float(model.predict(x)[0])
    return val


def predict_alpha(feature_row: Dict[str, Any]):
    return predict_target(feature_row, target="alpha")


def predict_beta(feature_row: Dict[str, Any]):
    return predict_target(feature_row, target="beta")