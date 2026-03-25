import torch
import sys
import numpy as np
import json
import os
import time
sys.path.insert(0, '.')

from fem_placer import (
    FpgaPlacer,
    Legalizer,
    FPGAPlacementOptimizer
)

from fem_placer.logger import *
from fem_placer.config import *
from ml.dataset import *
from placement_eval_logic import run_logic_placement

SET_LEVEL('WARNING')

num_trials = 5
num_steps = 200
dev = 'cpu'
manual_grad = False
anneal='inverse'
case_type = 'fpga_placement'

# Configuration
RESULT_DIR = './result'
COARSE_CACHE_FILE = os.path.join(RESULT_DIR, 'coarse_results.json')
USE_COARSE_CACHE = False  # Set to True to skip coarse sweep and use cached results

instances = ['c1355', 'c2670', 'c5315', 'c6288', 'c7552',
             's1238', 's1488', 's5378', 's9234', 's15850', 'FPGA-example1']

# instances = ['FPGA-example1']
# instances = ['c5315']

# instances = ['c6288', s]

# Load coarse cache if available
coarse_cache = {}
if USE_COARSE_CACHE and os.path.exists(COARSE_CACHE_FILE):
    try:
        with open(COARSE_CACHE_FILE, 'r') as f:
            coarse_cache = json.load(f)
        print(f"Loaded coarse cache from {COARSE_CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not load coarse cache: {e}")

# Clear the dataset before starting a new run
clear_dataset(with_io=False)
clear_dataset(with_io=True)

print(f"{'Instance':<15} | {'Best HPWL':<10} | {'Overlap':<10} | {'Overlap %':<10} | {'In Range':<10} | {'Alpha':<10} | {'Beta':<10} | {'Time (s)':<10}")
print("-" * 102)

for instance in instances:
    start_time = time.time()
    place_type = PlaceType.CENTERED
    debug = False
    fpga_placer = FpgaPlacer(place_orientation = place_type, 
                            grid_type = GridType.SQUARE,
                            place_mode = IoMode.NORMAL,
                            utilization_factor = 0.4,
                            debug = debug,
                            device = dev)
    
    vivado_hpwl, site_num, net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')    
    # Coarse sweep parameters
    coarse_start = 0
    coarse_end = 100
    coarse_step = 5
    
    # Fine sweep parameters
    fine_radius = 2.0
    fine_step = 0.2
    top_k = 5
    
    # Allowed overlap range
    overlap_allowed_min = 0
    overlap_allowed_max = 0.08

    def evaluate_placement(alpha, beta=0):
        """Evaluate placement for given alpha (and optionally beta for IO placement).

        Uses shared logic from placement_eval_logic.run_logic_placement.
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
        fem_hpwl_initial = res['fem_hpwl_initial']
        fem_hpwl_final = res['fem_hpwl_final']

        overlap_percent = float(overlap) / float(site_num['logic_inst_num'])
        in_allowed_range = (overlap_percent >= overlap_allowed_min and overlap_percent <= overlap_allowed_max)
        obj = fem_hpwl_final['hpwl_no_io'] if in_allowed_range else (fem_hpwl_final['hpwl_no_io'] + (overlap * 10))

        return {
            'alpha': alpha,
            'beta': beta,
            'obj': obj,
            'hpwl_initial': fem_hpwl_initial['hpwl_no_io'],
            'hpwl_final': fem_hpwl_final['hpwl_no_io'],
            'overlap': overlap,
            'overlap_percent': overlap_percent,
            'in_allowed_range': in_allowed_range,
        }

    # ==================== COARSE SWEEP ====================
    coarse_results = []
    
    # Check if this instance is in cache
    cache_key = instance
    if USE_COARSE_CACHE and cache_key in coarse_cache:
        cached_a1_info = coarse_cache[cache_key]
        a1 = cached_a1_info['a1']
        b1 = cached_a1_info.get('b1', 0.0)  # Default to 0 if not in cache
    else:
        if place_type == PlaceType.IO:
            # 2D grid search for IO placement
            beta_start = 0
            beta_end = 100
            beta_step = 5
            
            for used_alpha in range(coarse_start, coarse_end, coarse_step):
                for used_beta in range(beta_start, beta_end, beta_step):
                    res = evaluate_placement(used_alpha, used_beta)
                    coarse_results.append(res)
            
            # Find best alpha and beta from coarse sweep
            in_range_candidates = [r for r in coarse_results if r['in_allowed_range']]
            if len(in_range_candidates) > 0:
                best = min(in_range_candidates, key=lambda r: r['hpwl_final'])
            else:
                best = min(coarse_results, key=lambda r: r['obj'])
            
            a1 = best['alpha']
            b1 = best['beta']
        else:
            # 1D sweep for CENTER placement (only alpha)
            for used_alpha in range(coarse_start, coarse_end, coarse_step):
                res = evaluate_placement(used_alpha, 0)
                coarse_results.append(res)
            
            # Find best alpha from coarse sweep
            in_range_candidates = [r for r in coarse_results if r['in_allowed_range']]
            if len(in_range_candidates) > 0:
                best = min(in_range_candidates, key=lambda r: r['hpwl_final'])
            else:
                best = min(coarse_results, key=lambda r: r['obj'])
            
            a1 = best['alpha']
            b1 = 0.0
        
        # Cache coarse result
        coarse_cache[cache_key] = {'a1': a1, 'b1': b1}
        with open(COARSE_CACHE_FILE, 'w') as f:
            json.dump(coarse_cache, f, indent=2)

    # ==================== FINE SWEEP ====================
    fine_results = []
    seen_alphas = set(r['alpha'] for r in coarse_results)
    seen_betas = set(r['beta'] for r in coarse_results)

    if place_type == PlaceType.IO:
        # 2D fine sweep
        fine_alpha_low = a1 - fine_radius
        fine_alpha_high = a1 + fine_radius
        fine_beta_low = b1 - fine_radius
        fine_beta_high = b1 + fine_radius
        
        fine_alphas = np.arange(fine_alpha_low, fine_alpha_high + fine_step/2, fine_step)
        fine_betas = np.arange(fine_beta_low, fine_beta_high + fine_step/2, fine_step)
        
        for used_alpha in fine_alphas:
            for used_beta in fine_betas:
                used_alpha = float(used_alpha)
                used_beta = float(used_beta)
                
                if any(abs(sa - used_alpha) < 0.01 for sa in seen_alphas) and \
                   any(abs(sb - used_beta) < 0.01 for sb in seen_betas):
                    fine_results.append(next(r for r in coarse_results 
                                            if abs(r['alpha'] - used_alpha) < 0.01 and 
                                               abs(r['beta'] - used_beta) < 0.01))
                else:
                    res = evaluate_placement(used_alpha, used_beta)
                    fine_results.append(res)
    else:
        # 1D fine sweep
        fine_alpha_low = a1 - fine_radius
        fine_alpha_high = a1 + fine_radius
        fine_alphas = np.arange(fine_alpha_low, fine_alpha_high + fine_step/2, fine_step)
        
        for used_alpha in fine_alphas:
            used_alpha = float(used_alpha)
            
            if any(abs(sa - used_alpha) < 0.01 for sa in seen_alphas):
                fine_results.append(next(r for r in coarse_results 
                                        if abs(r['alpha'] - used_alpha) < 0.01))
            else:
                res = evaluate_placement(used_alpha, 0)
                fine_results.append(res)

    # ==================== RANK AND COLLECT TOP-K ====================
    all_results = coarse_results + fine_results
    unique_results = []
    seen = set()
    for r in all_results:
        key = (r['alpha'], r['beta'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
            
    fine_results_sorted = sorted(unique_results, key=lambda r: (0 if r['in_allowed_range'] else 1, r['hpwl_final'], r['overlap']))
    top_results = fine_results_sorted[:top_k]
    
    best = top_results[0]
    total_time = time.time() - start_time
    
    # User specifically asked for 5% threshold in prompt
    is_in_range_5pct = best['overlap_percent'] <= 0.05
    in_range_str = "Yes" if is_in_range_5pct else "No"
    
    best_beta_str = f"{best['beta']:.1f}" if place_type == PlaceType.IO else "N/A"
    print(f"{instance:<15} | {best['hpwl_final']:<10.2f} | {best['overlap']:<10} | {best['overlap_percent']*100:<9.2f}% | {in_range_str:<10} | {best['alpha']:<10.1f} | {best_beta_str:<10} | {total_time:<10.2f}")

    for idx, r in enumerate(top_results, start=1):
        # Append to CSV
        beta_val = r['beta'] if place_type == PlaceType.IO else 0.0
        row = extract_features_from_placer(
            fpga_placer,
            alpha=r['alpha'],
            beta=beta_val,
            with_io=(place_type == PlaceType.IO)
        )
        append_row(row, with_io=(place_type == PlaceType.IO))
    