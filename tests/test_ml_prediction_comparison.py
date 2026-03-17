import torch
import sys
import numpy as np
import time
import os
sys.path.insert(0, '.')

from fem_placer import FpgaPlacer, Legalizer, FPGAPlacementOptimizer
from fem_placer.logger import SET_LEVEL
from fem_placer.config import *
from ml.dataset import extract_features_from_placer
from ml.predict import predict_alpha

SET_LEVEL('WARNING')

# The 5 circuits mentioned in the target table
instances = ['c6288', 's13207', 'LU8PEEng']

num_trials = 5
num_steps = 200
dev = 'cpu'
manual_grad = False
anneal = 'inverse'
place_type = PlaceType.CENTERED
debug = False
overlap_allowed_max = 0.01 # 1% overlap allowed, as stated in the paper

DEFAULT_ALPHA = 10.0

def evaluate_placement(fpga_placer, site_num, alpha, beta=0.0):
    # clear grid state
    fpga_placer.grids['logic'].clear_all()
    if place_type == PlaceType.IO:
        fpga_placer.grids['io'].clear_all()
        
    fpga_placer.set_alpha(alpha)
    if place_type == PlaceType.IO:
        fpga_placer.set_beta(beta)

    optimizer = FPGAPlacementOptimizer(
        num_inst=fpga_placer.instances['logic'].num,
        num_fixed_inst=fpga_placer.instances['io'].num,
        num_site=fpga_placer.get_grid('logic').area,
        num_fixed_site=fpga_placer.get_grid('io').area,
        logic_grid_width = fpga_placer.get_grid('logic').area_width,
        coupling_matrix=fpga_placer.net_manager.insts_matrix,
        site_coords_matrix=fpga_placer.logic_site_coords,
        io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
        io_site_coords=fpga_placer.io_site_coords,
        constraint_alpha=fpga_placer.constraint_alpha,
        constraint_beta=fpga_placer.constraint_alpha,
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev,
        betamin=0.01,
        betamax=0.5,
        anneal=anneal,
        optimizer='adam',
        learning_rate=0.1,
        h_factor=0.01,
        seed=1,
        dtype=torch.float32,
        with_io=(place_type == PlaceType.IO),
        manual_grad=manual_grad
    )

    t0 = time.time()
    config, result = optimizer.optimize()
    optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
    legalizer = Legalizer(placer=fpga_placer, device=dev)
    logic_ids, io_ids = fpga_placer.get_ids()

    if place_type == PlaceType.IO:
        real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[0][optimal_inds[0]])
        real_io_coords = fpga_placer.get_grid('io').to_real_coords_tensor(config[1][optimal_inds[0]])
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids, real_io_coords, io_ids, include_io=True)
    else:
        real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[optimal_inds[0]])
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids)

    t1 = time.time()
    
    try:
        overlap_percent = float(overlap) / float(site_num) if site_num > 0 else 0.0
    except Exception:
        overlap_percent = 0.0

    return {
        'hpwl': fem_hpwl_final['hpwl_no_io'] if place_type != PlaceType.IO else fem_hpwl_final['hpwl'],
        'overlap': overlap,
        'overlap_percent': overlap_percent,
        'time': t1 - t0
    }

def run_grid_search(fpga_placer, site_num):
    # Reduced coarse sweep for testing (to save time, modify as needed)
    coarse_start = 0
    coarse_end = 100
    coarse_step = 5
    best_res = None
    best_obj = float('inf')
    best_alpha = 0
    
    for used_alpha in range(coarse_start, coarse_end, coarse_step):
        res = evaluate_placement(fpga_placer, site_num, used_alpha, 0)
        in_range = res['overlap_percent'] <= overlap_allowed_max
        obj = res['hpwl'] if in_range else res['hpwl'] + (res['overlap'] * 10)
        if obj < best_obj:
            best_obj = obj
            best_res = res
            best_alpha = used_alpha

    return best_alpha, best_res

if __name__ == "__main__":
    results = []

    for instance in instances:
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
            continue

        vivado_hpwl, site_num, net_num = fpga_placer.init_placement(dcp_path, pl_path)
        
        # 1. Grid Search Optimal
        print(f"  Running Grid Search...")
        t0_gs = time.time()
        best_alpha_gs, res_gs = run_grid_search(fpga_placer, site_num)
        t1_gs = time.time()
        gs_time = t1_gs - t0_gs
        
        # 2. ML Predicted
        print(f"  Running ML Prediction...")
        t0_ml_pred = time.time()
        row = extract_features_from_placer(fpga_placer, alpha=0, beta=0, with_io=False)
        try:
            pred_alpha = predict_alpha(row)
        except Exception as e:
            print(f"  Failed ML prediction ({e}), defaulting to {DEFAULT_ALPHA}")
            pred_alpha = DEFAULT_ALPHA
        t1_ml_pred = time.time()
        
        res_ml = evaluate_placement(fpga_placer, site_num, pred_alpha, 0)
        ml_inf_time = (t1_ml_pred - t0_ml_pred) * 1000 # ms
        
        # 3. Default
        print(f"  Running Default...")
        res_default = evaluate_placement(fpga_placer, site_num, DEFAULT_ALPHA, 0)
        
        results.append({
            'Circuit': instance,
            'GS_HPWL': res_gs['hpwl'],
            'GS_Overlap': res_gs['overlap'],
            'ML_HPWL': res_ml['hpwl'],
            'ML_Overlap': res_ml['overlap'],
            'Default_HPWL': res_default['hpwl'],
            'GS_Time_s': gs_time,
            'ML_Inf_Time_ms': ml_inf_time
        })


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