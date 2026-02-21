import csv
import os
from typing import Dict, Any, Optional

MODULE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(MODULE_DIR, "ml_data.csv")
FIELDNAMES = [
    "opti_insts_num", "avail_sites_num", "fixed_insts_num", "utilization",
    "logic_area_length", "logic_area_width", "io_height", "net_count",
    "hpwl_before", "hpwl_after", "overlap_after", "alpha", "beta"
]


def extract_features_from_placer(placer, logic_coords=None, io_coords=None, hpwl_before=None, hpwl_after=None, overlap_after=None, instance: str = '', alpha: float = 0.0, beta: float = 0.0) -> Dict[str, Any]:
    logic_grid = placer.get_grid("logic")
    io_grid = placer.get_grid("io")
    net_count = len(placer.net_manager.nets)
    utilization = placer.opti_insts_num / (logic_grid.area_length * logic_grid.area_width)

    row = {
        "opti_insts_num": int(placer.opti_insts_num),
        "avail_sites_num": int(placer.avail_sites_num),
        "fixed_insts_num": int(placer.fixed_insts_num),
        "utilization": float(utilization),
        "logic_area_length": int(logic_grid.area_length),
        "logic_area_width": int(logic_grid.area_width),
        "io_height": int(io_grid.area_width),
        "net_count": int(net_count),
        "hpwl_before": float(hpwl_before),
        "hpwl_after": float(hpwl_after),
        "overlap_after": int(overlap_after),
        "alpha": float(alpha),
        "beta": float(beta)
    }
    return row

def append_row(row: Dict[str, Any], path: Optional[str] = None):
    p = path or CSV_PATH
    exists = os.path.exists(p)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})