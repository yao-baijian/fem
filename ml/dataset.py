import csv
import os
from typing import Dict, Any, Optional, List

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")
os.makedirs(RESULT_DIR, exist_ok=True)

CSV_PATH_ALPHA = os.path.join(RESULT_DIR, "ml_data_alpha.csv")
CSV_PATH_ALPHA_BETA = os.path.join(RESULT_DIR, "ml_data_alpha_beta.csv")

LOGIC_FIELDNAMES = [
    "opti_insts_num", "avail_sites_num", "utilization",
    "logic_area_length", "logic_area_width", "net_count",
    "max_degree", "avg_degree", "logic_depth",
    "alpha"
]

LOGIC_IO_FIELDNAMES = [
    "opti_insts_num", "avail_sites_num", "fixed_insts_num", "utilization",
    "logic_area_length", "logic_area_width", "net_count",
    "max_degree", "avg_degree", "logic_depth",
    "alpha", "beta"
]

FIELDNAMES = LOGIC_FIELDNAMES

def set_fieldnames(with_io: bool = False):
    """Set the fieldnames based on whether IO is included."""
    global FIELDNAMES
    FIELDNAMES = LOGIC_IO_FIELDNAMES if with_io else LOGIC_FIELDNAMES

def get_feature_fieldnames(with_io: bool = False) -> List[str]:
    """Get only the feature fieldnames (excluding target and selection fields)."""
    fieldnames = LOGIC_IO_FIELDNAMES if with_io else LOGIC_FIELDNAMES
    # Remove target variables (alpha and beta)
    return [f for f in fieldnames if f not in ['alpha', 'beta']]

def extract_features_from_placer(
    placer,
    alpha: float = 0.0,
    beta: float = 0.0,
    with_io: bool = False
) -> Dict[str, Any]:
    """
    Extract features from placer for dataset.
    
    Args:
        placer: FpgaPlacer instance
        logic_coords, io_coords: Placement coordinates (for future use)
        hpwl_before, hpwl_after, overlap_after: Selection metrics
        instance: Instance name
        alpha, beta: Constraint weights
        with_io: Whether IO placement is used
    
    Returns:
        Dictionary with all features and selection fields
    """
    logic_grid = placer.get_grid("logic")
    utilization = placer.opti_insts_num / (logic_grid.area_length * logic_grid.area_width)

    max_degree, avg_degree = placer.net_manager.get_net_degrees()
    net_count=len(placer.net_manager.nets)
    logic_depth=placer.net_manager.logic_depth

    row = {
        # Feature fields (for training)
        "opti_insts_num": int(placer.opti_insts_num),
        "avail_sites_num": int(placer.avail_sites_num),
        "utilization": float(utilization),
        "logic_area_length": int(logic_grid.area_length),
        "logic_area_width": int(logic_grid.area_width),
        "net_count": int(net_count),
        "max_degree": int(max_degree),
        "avg_degree": float(avg_degree),
        "logic_depth": float(logic_depth),
        "alpha": float(alpha),
    }
    
    # Add IO-specific features if applicable
    if with_io:
        row["fixed_insts_num"] = int(placer.fixed_insts_num)
        row["beta"] = float(beta)
    
    return row

def get_csv_path(target: str = "alpha") -> str:
    if target == "beta":
        return CSV_PATH_ALPHA_BETA
    else:
        return CSV_PATH_ALPHA

def append_row(row: Dict[str, Any], path: Optional[str] = None, with_io: bool = False):
    """
    Append a row to the CSV file.
    
    Args:
        row: Dictionary with features
        path: Path to CSV file (defaults to CSV_PATH based on with_io)
        with_io: Whether to use IO fieldnames (determines default CSV path)
    """
    if path is None:
        path = CSV_PATH_ALPHA_BETA if with_io else CSV_PATH_ALPHA
    
    exists = os.path.exists(path)
    fieldnames = LOGIC_IO_FIELDNAMES if with_io else LOGIC_FIELDNAMES
    
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})