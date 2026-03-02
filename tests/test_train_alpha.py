import torch
import sys
import numpy as np
import json
import os
sys.path.insert(0, '.')

from fem_placer import (
    FpgaPlacer,
    Legalizer,
    FPGAPlacementOptimizer
)

from fem_placer.logger import *
from fem_placer.config import *
from ml.dataset import *

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

instances = ['c880', 'c1355', 'c2670', 'c5315', 'c6288', 'c7552',
             's713', 's1238', 's1488', 's5378', 's9234', 's15850']

# instances = ['FPGA-example1']
# instances = ['c5315']

# Load coarse cache if available
coarse_cache = {}
if USE_COARSE_CACHE and os.path.exists(COARSE_CACHE_FILE):
    try:
        with open(COARSE_CACHE_FILE, 'r') as f:
            coarse_cache = json.load(f)
        print(f"Loaded coarse cache from {COARSE_CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not load coarse cache: {e}")

for instance in instances:
    place_type = PlaceType.CENTERED
    debug = False
    fpga_placer = FpgaPlacer(place_type, 
                            GridType.RECTAN,
                            0.4,
                            debug,
                            device=dev)
    
    vivado_hpwl, site_num, site_net_num, total_net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    area_size = fpga_placer.grids['logic'].area
    
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
        """Evaluate placement for given alpha (and optionally beta for IO placement)."""
        # Clear grid state before each run
        fpga_placer.grids['logic'].clear_all()
        if place_type == PlaceType.IO:
            fpga_placer.grids['io'].clear_all()

        fpga_placer.set_alpha(alpha)
        if place_type == PlaceType.IO:
            fpga_placer.set_beta(beta)

        optimizer = FPGAPlacementOptimizer(
            num_inst=fpga_placer.opti_insts_num,
            num_fixed_inst=fpga_placer.fixed_insts_num,
            num_site=fpga_placer.get_grid('logic').area,
            logic_grid_width = fpga_placer.get_grid('logic').area_width,
            coupling_matrix=fpga_placer.net_manager.insts_matrix,
            site_coords_matrix=fpga_placer.logic_site_coords,
            io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
            io_site_coords=fpga_placer.io_site_coords,
            constraint_alpha=fpga_placer.constraint_alpha,
            constraint_beta=fpga_placer.constraint_alpha,  # For IO placements, beta is set separately
            num_trials=num_trials,
            num_steps=num_steps,
            dev=dev,
            betamin=0.01,
            betamax=0.5,
            anneal='inverse',
            optimizer='adam',
            learning_rate=0.1,
            h_factor=0.01,
            seed=1,
            dtype=torch.float32,
            with_io=(place_type == PlaceType.IO),
            manual_grad=manual_grad
        )

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

        try:
            overlap_percent = float(overlap) / float(site_num) if site_num > 0 else 0.0
        except Exception:
            overlap_percent = 0.0

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
    print(f"\n{'='*80}")
    print(f"Instance: {instance}")
    print(f"Place Type: {place_type.name}")
    print(f"{'='*80}")
    
    coarse_results = []
    
    # Check if this instance is in cache
    cache_key = instance
    if USE_COARSE_CACHE and cache_key in coarse_cache:
        print(f"Using cached coarse results for {instance}")
        cached_a1_info = coarse_cache[cache_key]
        a1 = cached_a1_info['a1']
        b1 = cached_a1_info.get('b1', 0.0)  # Default to 0 if not in cache
        print(f"Cached a1={a1}, b1={b1}")
    else:
        print("Running coarse sweep...")
        
        if place_type == PlaceType.IO:
            # 2D grid search for IO placement
            beta_start = 0
            beta_end = 100
            beta_step = 5
            
            print(f"2D Coarse sweep: alpha=[{coarse_start}..{coarse_end}], beta=[{beta_start}..{beta_end}]")
            
            for used_alpha in range(coarse_start, coarse_end, coarse_step):
                for used_beta in range(beta_start, beta_end, beta_step):
                    res = evaluate_placement(used_alpha, used_beta)
                    coarse_results.append(res)
                    print(f"  alpha={used_alpha:<3.0f} beta={used_beta:<3.0f} | "
                          f"hpwl={res['hpwl_final']:<10.2f} overlap={res['overlap_percent']:<6.3f} "
                          f"in_range={res['in_allowed_range']}")
            
            # Find best alpha and beta from coarse sweep
            in_range_candidates = [r for r in coarse_results if r['in_allowed_range']]
            if len(in_range_candidates) > 0:
                best = min(in_range_candidates, key=lambda r: r['hpwl_final'])
            else:
                best = min(coarse_results, key=lambda r: r['obj'])
            
            a1 = best['alpha']
            b1 = best['beta']
            print(f"Best from coarse: a1={a1}, b1={b1}, hpwl={best['hpwl_final']:.2f}")
        else:
            # 1D sweep for CENTER placement (only alpha)
            print(f"1D Coarse sweep: alpha=[{coarse_start}..{coarse_end}]")
            
            for used_alpha in range(coarse_start, coarse_end, coarse_step):
                res = evaluate_placement(used_alpha, 0)
                coarse_results.append(res)
                print(f"  alpha={used_alpha:<3.0f} | hpwl={res['hpwl_final']:<10.2f} "
                      f"overlap={res['overlap_percent']:<6.3f} in_range={res['in_allowed_range']}")
            
            # Find best alpha from coarse sweep
            in_range_candidates = [r for r in coarse_results if r['in_allowed_range']]
            if len(in_range_candidates) > 0:
                best = min(in_range_candidates, key=lambda r: r['hpwl_final'])
            else:
                best = min(coarse_results, key=lambda r: r['obj'])
            
            a1 = best['alpha']
            b1 = 0.0
            print(f"Best from coarse: a1={a1}, hpwl={best['hpwl_final']:.2f}")
        
        # Cache coarse result
        coarse_cache[cache_key] = {'a1': a1, 'b1': b1}
        with open(COARSE_CACHE_FILE, 'w') as f:
            json.dump(coarse_cache, f, indent=2)
        print(f"Saved coarse results to {COARSE_CACHE_FILE}")

    # ==================== FINE SWEEP ====================
    print(f"\nRunning fine sweep around a1={a1}, b1={b1}...")
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
    fine_results_sorted = sorted(fine_results, key=lambda r: (0 if r['in_allowed_range'] else 1, r['hpwl_final'], r['overlap']))
    top_results = fine_results_sorted[:top_k]

    print(f"\nTop {top_k} results:")
    for idx, r in enumerate(top_results, start=1):
        print(f"Top{idx}: alpha={r['alpha']:.1f} beta={r['beta']:.1f} hpwl={r['hpwl_final']:.2f} overlap={r['overlap_percent']:.3f}")
        
        # Append to CSV
        beta_val = r['beta'] if place_type == PlaceType.IO else 0.0
        row = extract_features_from_placer(
            fpga_placer,
            alpha=r['alpha'],
            beta=beta_val,
            with_io=(place_type == PlaceType.IO)
        )
        append_row(row, with_io=(place_type == PlaceType.IO))
    