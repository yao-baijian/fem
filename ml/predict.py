from typing import Dict, Any
from .model import load_model, get_model_path
from .dataset import get_feature_fieldnames
import pandas as pd


def predict_target(feature_row: Dict[str, Any], target: str = "alpha"):
    """
    Predict a target value (alpha or beta) from feature values.
    
    Args:
        feature_row: Dictionary with feature values
        target: Target variable ("alpha" or "beta")
    
    Returns:
        Predicted value for the target
    
    Raises:
        RuntimeError: If model for target has not been trained
    """
    model_path = get_model_path(target)
    model = load_model(model_path)
    if model is None:
        raise RuntimeError(f"No trained model found for target '{target}'. Run ml.train.train_from_csv(target='{target}') first.")
    
    # Use pandas DataFrame to preserve feature names (matches training)
    # Determine with_io based on target (beta requires IO features)
    with_io = (target == "beta")
    feature_names = get_feature_fieldnames(with_io=with_io)
    x = pd.DataFrame([[feature_row.get(n, 0.0) for n in feature_names]], columns=feature_names)
    val = float(model.predict(x)[0])
    return val


def predict_alpha(feature_row: Dict[str, Any]) -> float:
    """
    Predict alpha value (constraint weight for logic placement).
    
    Args:
        feature_row: Dictionary with feature values from placer
    
    Returns:
        Predicted alpha value
    """
    return predict_target(feature_row, target="alpha")


def predict_beta(feature_row: Dict[str, Any]) -> float:
    """
    Predict beta value (constraint weight for IO placement).
    
    Args:
        feature_row: Dictionary with feature values from placer
    
    Returns:
        Predicted beta value
    """
    return predict_target(feature_row, target="beta")