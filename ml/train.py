import os
import pandas as pd
import sys
sys.path.append('.')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ml.model import create_default_model, save_model, get_model_path
from ml.dataset import (
    CSV_PATH, 
    get_feature_fieldnames,
    LOGIC_FIELDNAMES,
    LOGIC_IO_FIELDNAMES
)

def detect_fieldnames_from_csv(csv_path: str):
    """
    Detect which fieldnames the CSV uses by reading the header.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        Appropriate fieldnames list
    """
    try:
        df = pd.read_csv(csv_path, nrows=0)  # Read only header
        columns = set(df.columns)
        
        # Check if it's IO fieldnames
        if "beta" in columns and "fixed_insts_num" in columns:
            return LOGIC_IO_FIELDNAMES
        else:
            return LOGIC_FIELDNAMES
    except Exception as e:
        print(f"Warning: Could not detect fieldnames: {e}, using default LOGIC_FIELDNAMES")
        return LOGIC_FIELDNAMES

def train_from_csv(csv_path: str = None, target: str = "alpha", test_size: float = 0.2):
    """
    Train model from CSV data.
    
    Args:
        csv_path: Path to CSV file
        target: Target column name ("alpha" or "beta")
        test_size: Fraction of data to use for validation
    
    Returns:
        Dictionary with MSE and model path
    """
    p = csv_path or CSV_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"No CSV at {p}")
    
    # Detect which fieldnames the CSV uses
    fieldnames = detect_fieldnames_from_csv(p)
    
    # Get feature fieldnames (exclude selection fields and target)
    with_io = "beta" in fieldnames
    feature_fieldnames = get_feature_fieldnames(with_io=with_io)
    
    # Read CSV
    df = pd.read_csv(p)
    df = df.dropna()
    
    # Validate target
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not present in CSV")
    
    # Validate features
    missing_features = [f for f in feature_fieldnames if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    # Prepare training data
    X = df[feature_fieldnames]
    y = df[target]
    
    print(f"Training with {len(X)} samples, {len(feature_fieldnames)} features")
    print(f"Features: {feature_fieldnames}")
    print(f"Target: {target}")
    
    # Split and train
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    model = create_default_model()
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    
    # Save
    model_path = get_model_path(target)
    save_model(model, model_path)
    
    print(f"MSE: {mse:.4f}")
    print(f"Model saved to: {model_path}")
    
    return {"mse": mse, "model_path": model_path}

if __name__ == "__main__":
    res = train_from_csv()
    print(res)