from typing import Dict, Any
from .model import load_model
from .dataset import FIELDNAMES
import numpy as np

_PRE_ALPHA_FEATURES = [f for f in FIELDNAMES if f not in ("alpha", "hpwl_after", "overlap_after", "instance")]

def predict_alpha(feature_row: Dict[str, Any]):
    model = load_model()
    if model is None:
        raise RuntimeError("No trained model found. Run ml_alpha.train.train_from_csv() first.")
    x = np.array([[feature_row.get(n, 0.0) for n in _PRE_ALPHA_FEATURES]])
    alpha = float(model.predict(x)[0])
    return alpha