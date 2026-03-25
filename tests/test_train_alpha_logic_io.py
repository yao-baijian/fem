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

SET_LEVEL('WARNING')

num_trials = 5
num_steps = 200
dev = 'cuda'
manual_grad = False
anneal='inverse'
case_type = 'fpga_placement'
io_factor = 100.0

# Configuration
RESULT_DIR = './result'
COARSE_CACHE_FILE = os.path.join(RESULT_DIR, 'coarse_results_io.json')
USE_COARSE_CACHE = True  # Set to True to skip coarse sweep and use cached results
VERBOSE_EVAL = False       # Set to False to disable printing each run
ENABLE_FINE_SWEEP = False  # Set to False to stop after coarse sweep
MAX_WORKERS = 16           # Number of parallel workers for coarse sweep

instances = ['c2670_boundary', 'c5315_boundary', 'c6288_boundary', 'c7552_boundary',
             's1488_boundary', 's5378_boundary', 's9234_boundary', 's15850_boundary']

# instances = ['bgm_boundary', 'sha1_boundary', 'RLE_BlobMerging_boundary']
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

# Clear the dataset before starting a new run
clear_dataset(with_io=False)
clear_dataset(with_io=True)

print(f"{'Instance':<15} | {'Best HPWL':<10} | {'OvL':<6} | {'OvIO':<6} | {'OvL %':<8} | {'OvIO %':<8} | {'In Range':<10} | {'Alpha':<10} | {'Beta':<10} | {'Time (s)':<10}")
print("-" * 128)

for instance in instances:
    start_time = time.time()
    place_type = PlaceType.IO
    debug = False
    fpga_placer = FpgaPlacer(place_orientation = place_type, 
                            grid_type = GridType.SQUARE,
                            place_mode = IoMode.VIRTUAL_NODE,
                            utilization_factor = 0.4,
                            debug = debug,
                            device = dev)
    
    fpga_placer.set_instance_name(instance)
    
    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    
    # Coarse sweep parameters
    coarse_start = 0
    coarse_end = 50
    coarse_step = 5
    
    # Fine sweep parameters
    fine_radius = 5.0
    fine_step = 0.2
    top_k = 5
    
    # Allowed overlap range
    overlap_allowed_min = 0
    overlap_allowed_max = 0.08

    def evaluate_placement(alpha, beta=0, skip_legalization=False):
        """Evaluate placement for given alpha (and optionally beta for IO placement).

        Returns overlap metrics separately for logic and IO instances.
        """
        if VERBOSE_EVAL:
            print(f"[{instance}] Evaluating alpha={alpha:.2f}, beta={beta:.2f} (skip_legalization={skip_legalization})...")
            
        optimizer = FPGAPlacementOptimizer(
            num_inst=fpga_placer.instances['logic'].num,
            num_fixed_inst=fpga_placer.instances['io'].num,
            num_site=fpga_placer.get_grid('logic').area,
            num_fixed_site=fpga_placer.get_grid('io').area,
            coupling_matrix=fpga_placer.net_manager.insts_matrix,
            site_coords_matrix=fpga_placer.logic_site_coords,
            io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
            io_site_coords=fpga_placer.io_site_coords,
            constraint_alpha=alpha,
            constraint_beta=beta,  # For IO placements, beta is set separately
            num_trials=num_trials,
            num_steps=num_steps,
            dev=dev,
            betamin=0.01,
            betamax=0.5,
            anneal='inverse',
            optimizer='adam',
            learning_rate=0.1,
            h_factor=0.01,
            io_factor=io_factor,
            seed=1,
            dtype=torch.float32,
            with_io=(place_type == PlaceType.IO),
            manual_grad=manual_grad
        )

        if VERBOSE_EVAL:
            print(f"  -> Optimizer starting...")
        
        config, result = optimizer.optimize()
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
        legalizer = Legalizer(placer=fpga_placer, device=dev)
        logic_ids, io_ids = fpga_placer.get_ids()

        # Default overlap counts
        overlap_logic = 0
        overlap_io = 0

        if VERBOSE_EVAL:
            print(f"  -> Optimizer finished. Legalizer starting...")

        if skip_legalization:
            if place_type == PlaceType.IO:
                real_logic_coords = config[0][optimal_inds[0]]
                real_io_coords = config[1][optimal_inds[0]]

                # Check overlap before legalization using pure PyTorch logic (thread-safe)
                n_unique_logic = len(torch.unique(real_logic_coords, dim=0))
                overlap_logic = real_logic_coords.shape[0] - n_unique_logic

                n_unique_io = len(torch.unique(real_io_coords, dim=0))
                overlap_io = real_io_coords.shape[0] - n_unique_io

                fem_hpwl_initial = fpga_placer.net_manager.analyze_solver_hpwl(real_logic_coords, real_io_coords, True)
                fem_hpwl_final = fem_hpwl_initial  # Same since skipped
                overlap = overlap_logic + overlap_io
            else:
                real_logic_coords = config[optimal_inds[0]]

                # Overlap before legalization
                n_unique = len(torch.unique(real_logic_coords, dim=0))
                overlap_logic = real_logic_coords.shape[0] - n_unique
                overlap = overlap_logic

                fem_hpwl_initial = fpga_placer.net_manager.analyze_solver_hpwl(real_logic_coords, None, False)
                fem_hpwl_final = fem_hpwl_initial
            placement_legalized = [real_logic_coords, real_io_coords if place_type == PlaceType.IO else None]
        else:
            # Recreate local grids in single-thread fine sweep strictly to avoid global mutation
            fpga_placer.grids['logic'].clear_all()
            if place_type == PlaceType.IO:
                fpga_placer.grids['io'].clear_all()

            if place_type == PlaceType.IO:
                real_logic_coords = config[0][optimal_inds[0]]
                real_io_coords = config[1][optimal_inds[0]]

                # Compute overlaps before legalization (this is what we want to report)
                n_unique_logic = len(torch.unique(real_logic_coords, dim=0))
                overlap_logic = real_logic_coords.shape[0] - n_unique_logic

                n_unique_io = len(torch.unique(real_io_coords, dim=0))
                overlap_io = real_io_coords.shape[0] - n_unique_io
                overlap = overlap_logic + overlap_io

                # Now run legalization to get final HPWL
                placement_legalized, _, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
                    real_logic_coords, logic_ids, real_io_coords, io_ids, include_io=True
                )
            else:
                real_logic_coords = config[optimal_inds[0]]

                # Compute overlap before legalization
                n_unique_logic = len(torch.unique(real_logic_coords, dim=0))
                overlap_logic = real_logic_coords.shape[0] - n_unique_logic
                overlap = overlap_logic

                # Legalize to obtain final HPWL
                placement_legalized, _, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids)

        if VERBOSE_EVAL:
            print(f"  -> Legalizer finished.")

        # Compute overlap percentages separately for logic and IO
        logic_inst_total = float(inst_num['logic_inst_num']) if inst_num['logic_inst_num'] > 0 else 1.0
        io_inst_total = float(inst_num['io_inst_num']) if inst_num.get('io_inst_num', 0) > 0 else 1.0

        overlap_percent = float(overlap_logic) / logic_inst_total
        io_overlap_percent = float(overlap_io) / io_inst_total if inst_num.get('io_inst_num', 0) > 0 else 0.0
        in_allowed_range = (overlap_percent >= overlap_allowed_min and overlap_percent <= overlap_allowed_max)
        obj = fem_hpwl_final['hpwl'] if in_allowed_range else (fem_hpwl_final['hpwl'] + (overlap * 10))

        return {
            'alpha': alpha,
            'beta': beta,
            'obj': obj,
            'hpwl_initial': fem_hpwl_initial['hpwl'],
            'hpwl_final': fem_hpwl_final['hpwl'],
            'overlap': overlap,
            'overlap_logic': overlap_logic,
            'overlap_io': overlap_io,
            'overlap_percent': overlap_percent,
            'io_overlap_percent': io_overlap_percent,
            'in_allowed_range': in_allowed_range,
        }

    # ==================== COARSE SWEEP ====================
    import concurrent.futures
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
            beta_end = 50
            beta_step = 5
            
            tasks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for used_alpha in range(coarse_start, coarse_end, coarse_step):
                    for used_beta in range(beta_start, beta_end, beta_step):
                        tasks.append(executor.submit(evaluate_placement, used_alpha, used_beta, True))
                        
                for future in concurrent.futures.as_completed(tasks):
                    res = future.result()
                    coarse_results.append(res)
                    if VERBOSE_EVAL:
                        print(f"[{instance}] Finished Coarse: alpha={res['alpha']:.2f}, beta={res['beta']:.2f} | obj={res['obj']:.2f}")
            
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
            tasks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for used_alpha in range(coarse_start, coarse_end, coarse_step):
                    tasks.append(executor.submit(evaluate_placement, used_alpha, 0, True))
                    
                for future in concurrent.futures.as_completed(tasks):
                    res = future.result()
                    coarse_results.append(res)
                    if VERBOSE_EVAL:
                        print(f"[{instance}] Finished Coarse: alpha={res['alpha']:.2f} | obj={res['obj']:.2f}")
            
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

    if not ENABLE_FINE_SWEEP:
        # User specified not to continue on fine sweep. Wrap up using best known configuration.
        # Fallback evaluation just to generate final output using full legalization:
        res = evaluate_placement(a1, b1, skip_legalization=False)
        total_time = time.time() - start_time
        is_in_range_5pct = (res['overlap_percent'] <= 0.05) and (res['io_overlap_percent'] <= 0.05)
        in_range_str = "Yes" if is_in_range_5pct else "No"
        best_beta_str = f"{res['beta']:.1f}" if place_type == PlaceType.IO else "N/A"
        print(
            f"{instance:<15} | "
            f"{res['hpwl_final']:<10.2f} | "
            f"{res['overlap_logic']:<6} | "
            f"{res['overlap_io']:<6} | "
            f"{res['overlap_percent']*100:<8.2f}% | "
            f"{res['io_overlap_percent']*100:<8.2f}% | "
            f"{in_range_str:<10} | "
            f"{res['alpha']:<10.1f} | "
            f"{best_beta_str:<10} | "
            f"{total_time:<10.2f}"
        )

        # Append to CSV
        beta_val = res['beta'] if place_type == PlaceType.IO else 0.0
        row = extract_features_from_placer(
            fpga_placer,
            alpha=res['alpha'],
            beta=beta_val,
            with_io=(place_type == PlaceType.IO)
        )
        append_row(row, with_io=(place_type == PlaceType.IO))
        continue

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
    is_in_range_5pct = (best['overlap_percent'] <= 0.05) and (best['io_overlap_percent'] <= 0.05)
    in_range_str = "Yes" if is_in_range_5pct else "No"
    
    best_beta_str = f"{best['beta']:.1f}" if place_type == PlaceType.IO else "N/A"
    print(
        f"{instance:<15} | "
        f"{best['hpwl_final']:<10.2f} | "
        f"{best['overlap_logic']:<6} | "
        f"{best['overlap_io']:<6} | "
        f"{best['overlap_percent']*100:<8.2f}% | "
        f"{best['io_overlap_percent']*100:<8.2f}% | "
        f"{in_range_str:<10} | "
        f"{best['alpha']:<10.1f} | "
        f"{best_beta_str:<10} | "
        f"{total_time:<10.2f}"
    )

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
    