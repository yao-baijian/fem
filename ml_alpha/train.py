import os
import pandas as pd
import sys
sys.path.append('.')
from FEM import FEM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from FEM.ml_alpha.model import create_default_model, save_model, MODEL_PATH
from FEM.ml_alpha.dataset import CSV_PATH, FIELDNAMES

PRE_ALPHA_FEATURES = [f for f in FIELDNAMES if f not in ("alpha", "hpwl_after", "overlap_after")]

def train_from_csv(csv_path: str = None, target: str = "alpha", test_size: float = 0.2):
    p = csv_path or CSV_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"No CSV at {p}")
    df = pd.read_csv(p)
    df = df.dropna()
    if target not in df.columns:
        raise ValueError("target column not present")
    X = df[PRE_ALPHA_FEATURES]
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    model = create_default_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    save_model(model, MODEL_PATH)
    return {"mse": mse, "model_path": MODEL_PATH}

if __name__ == "__main__":
    res = train_from_csv()
    print(res)