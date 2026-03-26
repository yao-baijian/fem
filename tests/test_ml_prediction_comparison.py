import torch
import sys
import numpy as np
import time
import os
import concurrent.futures
import warnings
sys.path.insert(0, '.')

from fem_placer import FpgaPlacer, FPGAPlacementOptimizer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import *
from ml.dataset import extract_features_from_placer
from ml.predict import predict_target
from placement_eval_logic import run_logic_placement
from sklearn.exceptions import InconsistentVersionWarning

SET_LEVEL('WARNING')

# Silence sklearn model version mismatch warnings when unpickling
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# The 5 circuits mentioned in the target table
instances = ['s13207', 's35932', 'LU8PEEng', 'sv_chip0_hierarchy_no_mem']

num_trials = 5
num_steps = 200
dev = 'cuda'
manual_grad = False
anneal = 'lin'
place_type = PlaceType.CENTERED
debug = False
overlap_allowed_max = 0.05 # 1% overlap allowed, as stated in the paper

DEFAULT_ALPHA = 10.0
MAX_WORKERS = 10

def evaluate_placement(fpga_placer, inst_num, alpha, beta=0.0):
    """Evaluate placement for a given alpha using shared logic helper.

    Uses placement_eval_logic.run_logic_placement and keeps original metrics.
    """
    res = run_logic_placement(
        fpga_placer=fpga_placer,
        alpha=alpha,
        beta=beta,
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev,
        manual_grad=manual_grad,
        anneal=anneal,
        place_type=place_type,
    )

    overlap = res['overlap']
    fem_hpwl = res['fem_hpwl_initial']
    elapsed = res['time']

    logic_inst_total = float(inst_num.get('logic_inst_num', 0))
    overlap_percent = float(overlap) / logic_inst_total if logic_inst_total > 0 else 0.0

    hpwl_key = 'hpwl_no_io' if place_type != PlaceType.IO else 'hpwl'

    return {
        'hpwl': fem_hpwl[hpwl_key],
        'overlap': overlap,
        'overlap_percent': overlap_percent,
        'time': elapsed,
    }

def run_grid_search(fpga_placer, inst_num):
    """Coarse alpha sweep following the multi-thread pattern in test_train_alpha_logic_io.

    Uses a ThreadPoolExecutor over alpha values with a thread-safe evaluation
    (no legalization, no grid mutation), then selects the best alpha.
    """
    # Reduced coarse sweep for testing (to save time, modify as needed)
    coarse_start = 0
    coarse_end = 100
    coarse_step = 5

    def evaluate_alpha_no_legal(used_alpha: float):
        optimizer = FPGAPlacementOptimizer(
            num_inst=fpga_placer.instances["logic"].num,
            num_fixed_inst=fpga_placer.instances["io"].num,
            num_site=fpga_placer.get_grid("logic").area,
            num_fixed_site=fpga_placer.get_grid("io").area,
            coupling_matrix=fpga_placer.net_manager.insts_matrix,
            site_coords_matrix=fpga_placer.logic_site_coords,
            io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
            io_site_coords=fpga_placer.io_site_coords,
            constraint_alpha=used_alpha,
            constraint_beta=0.0,
            num_trials=num_trials,
            num_steps=num_steps,
            dev=dev,
            betamin=0.01,
            betamax=0.5,
            anneal=anneal,
            optimizer="adam",
            learning_rate=0.1,
            h_factor=0.01,
            io_factor=1.0,
            seed=1,
            dtype=torch.float32,
            with_io=False,
            manual_grad=manual_grad,
        )

        config, result = optimizer.optimize()
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
        real_logic_coords = config[optimal_inds[0]]

        # Overlap before legalization (thread-safe, pure tensor logic)
        n_unique = len(torch.unique(real_logic_coords, dim=0))
        overlap = real_logic_coords.shape[0] - n_unique

        fem_hpwl_initial = fpga_placer.net_manager.analyze_solver_hpwl(
            real_logic_coords, None, False
        )
        hpwl = fem_hpwl_initial["hpwl_no_io"] if place_type != PlaceType.IO else fem_hpwl_initial["hpwl"]

        logic_inst_total = float(inst_num.get('logic_inst_num', 0))
        overlap_percent = float(overlap) / logic_inst_total if logic_inst_total > 0 else 0.0

        return {
            "alpha": used_alpha,
            "hpwl": hpwl,
            "overlap": overlap,
            "overlap_percent": overlap_percent,
        }

    best_res = None
    best_obj = float("inf")
    best_alpha = 0.0

    tasks = []
    coarse_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for used_alpha in range(coarse_start, coarse_end, coarse_step):
            tasks.append(executor.submit(evaluate_alpha_no_legal, float(used_alpha)))

        for future in concurrent.futures.as_completed(tasks):
            res = future.result()
            coarse_results.append(res)
            used_alpha = res["alpha"]
            print(
                f"  Testing alpha={used_alpha}: HPWL={res['hpwl']}, Overlap={res['overlap']}"
            )
            in_range = res["overlap_percent"] <= overlap_allowed_max
            obj = res["hpwl"] if in_range else res["hpwl"] + (res["overlap"] * 10)
            if obj < best_obj:
                best_obj = obj
                best_res = res
                best_alpha = used_alpha

    return best_alpha, best_res

def process_instance(instance: str):
    """Run grid search + ML/default comparison for a single instance.

    Designed to be executed in parallel across instances.
    """
    print(f"Processing instance {instance}...")
    fpga_placer = FpgaPlacer(place_orientation = place_type, 
                            grid_type = GridType.SQUARE,
                            place_mode = IoMode.NORMAL,
                            utilization_factor = 0.4,
                            debug = debug,
                            device = dev)
    fpga_placer.set_instance_name(instance)
    
    dcp_path = f'./vivado/output_dir/{instance}/post_impl.dcp'
    pl_path = f'./vivado/output_dir/{instance}/optimized_placement.pl'
    
    if not os.path.exists(dcp_path):
        print(f"Skipping {instance}: {dcp_path} not found.")
        return None

    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(dcp_path, pl_path)
    
    # 1. Grid Search Optimal
    print(f"  Running Grid Search...")
    t0_gs = time.time()
    best_alpha_gs, res_gs = run_grid_search(fpga_placer, inst_num)
    t1_gs = time.time()
    gs_time = t1_gs - t0_gs
    
    # 2. ML Predicted
    print(f"  Running ML Prediction...")
    t0_ml_pred = time.time()
    row = extract_features_from_placer(fpga_placer, alpha=0, beta=0, with_io=False)
    try:
        # Use generic multi-task prediction interface (target='alpha' here)
        pred_alpha = predict_target(row, target="alpha")
    except Exception as e:
        print(f"  Failed ML prediction ({e}), defaulting to {DEFAULT_ALPHA}")
        pred_alpha = DEFAULT_ALPHA
    t1_ml_pred = time.time()
    
    res_ml = evaluate_placement(fpga_placer, inst_num, pred_alpha, 0)
    ml_inf_time = (t1_ml_pred - t0_ml_pred) * 1000 # ms
    
    # 3. Default
    print(f"  Running Default...")
    res_default = evaluate_placement(fpga_placer, inst_num, DEFAULT_ALPHA, 0)
    
    return {
        'Circuit': instance,
        'GS_HPWL': res_gs['hpwl'],
        'GS_Overlap': res_gs['overlap'],
        'ML_HPWL': res_ml['hpwl'],
        'ML_Overlap': res_ml['overlap'],
        'Default_HPWL': res_default['hpwl'],
        'GS_Time_s': gs_time,
        'ML_Inf_Time_ms': ml_inf_time
    }


if __name__ == "__main__":
    results = []

    # Follow test_train_alpha_logic_io.py pattern: process instances sequentially,
    # and use multithreading only inside the coarse alpha sweep.
    for inst in instances:
        try:
            res = process_instance(inst)
        except Exception as e:
            print(f"Instance {inst} failed with error: {e}")
            continue
        if res is not None:
            results.append(res)

    # Print Table
    if results:
        print("\\n" + "="*80)
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Comparison of HPWL and overlap using different coefficient selection methods.}")
        print("\\label{tab:ml-tuning-results}")
        print("\\begin{tabular}{l|c|c|c|c|c}")
        print("\\toprule")
        print("\\textbf{Circuit} & \\multicolumn{2}{c|}{\\textbf{Grid Search Optimal}} & \\multicolumn{2}{c|}{\\textbf{ML-Predicted}} & \\textbf{Default} \\\\")
        print("& HPWL & Overlap & HPWL & Overlap & HPWL \\\\")
        print("\\midrule")
        for r in results:
            print(f"{r['Circuit']} & {int(r['GS_HPWL'])} & {int(r['GS_Overlap'])} & {int(r['ML_HPWL'])} & {int(r['ML_Overlap'])} & {int(r['Default_HPWL'])} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")

        print("\\n--- Timing Info ---")
        for r in results:
            print(f"{r['Circuit']}: Grid Search took {r['GS_Time_s']:.1f}s, ML Prediction took {r['ML_Inf_Time_ms']:.2f}ms")
    else:
        print("No results generated.")