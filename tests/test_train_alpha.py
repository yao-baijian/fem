import torch
import sys
import numpy as np
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

# instances = ['c880', 'c1355', 'c2670', 'c5315', 'c6288', 'c7552',
#              's713', 's1238', 's1488', 's5378', 's9234', 's15850', 'FPGA-example1']

instances = ['c5315']

for instance in instances:
    place_type = PlaceType.CENTERED
    debug = False
    fpga_placer = FpgaPlacer(place_type, 
                            GridType.SQUARE,
                            0.4,
                            debug,
                            device=dev)
    
    vivado_hpwl, site_num, site_net_num, total_net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    area_size = fpga_placer.grids['logic'].area
    
    # Two-step alpha search:
    # 1) coarse sweep to find approximate best alpha a1
    # 2) fine sweep around a1 and collect top-k results (allowing a small overlap range)
    best_obj = float('inf')
    best_alpha = -1
    best_hpwl_initial = -1
    best_hpwl_final = -1

    # Coarse sweep parameters
    coarse_start = 0
    coarse_end = 50
    coarse_step = 5

    # Fine sweep parameters
    fine_radius = 1.0  # ±1.0 around a1
    fine_step = 0.1    # 0.1 level granularity
    top_k = 5

    # Allowed overlap as fraction (e.g., 0.01 = 1%) - prefer solutions whose relative overlap lies inside this range
    overlap_allowed_min = 0
    overlap_allowed_max = 0.1

    def evaluate_alpha(alpha, beta=0):
        fpga_placer.set_alpha(alpha)
        # For IO placements, set beta separately; for CENTERED, beta remains 0
        if place_type == PlaceType.IO:
            fpga_placer.set_beta(beta)
        
        optimizer = FPGAPlacementOptimizer(
            num_inst=fpga_placer.opti_insts_num,
            num_fixed_inst=fpga_placer.fixed_insts_num,
            num_site=fpga_placer.get_grid('logic').area,
            coupling_matrix=fpga_placer.net_manager.insts_matrix,
            site_coords_matrix=fpga_placer.logic_site_coords,
            io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
            io_site_coords=fpga_placer.io_site_coords,
            bbox_length = fpga_placer.grids['logic'].area_length,
            constraint_alpha=fpga_placer.constraint_alpha,
            constraint_beta=fpga_placer.constraint_alpha,
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
        optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
        legalizer = Legalizer(placer=fpga_placer, device=dev)
        logic_ids, io_ids = fpga_placer.get_ids()

        if place_type == PlaceType.IO:
            real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[0][optimal_inds[0]])
            real_io_coords = fpga_placer.get_grid('io').to_real_coords_tensor(config[1][optimal_inds[0]])
            placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids, real_io_coords, io_ids, include_io = True)
        else:
            real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[optimal_inds[0]])
            placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids)

        # Compute relative overlap if possible
        try:
            overlap_percent = float(overlap) / float(site_num) if site_num > 0 else 0.0
        except Exception:
            overlap_percent = 0.0

        # Primary ranking: prefer solutions with overlap inside allowed range; secondary by hpwl
        in_allowed_range = (overlap_percent >= overlap_allowed_min and overlap_percent <= overlap_allowed_max)

        # Fallback objective (keeps previous penalty semantics when not in allowed range)
        obj = fem_hpwl_final['hpwl_no_io'] if in_allowed_range else (fem_hpwl_final['hpwl_no_io'] + (overlap * 10))

        print(f"{'Benchmarks':<12} {instance:<10} {site_num:<6} {f'{site_net_num}/{total_net_num}':<14} {overlap:<8} "
                f"{fem_hpwl_initial['hpwl_no_io']:<18.2f} {fem_hpwl_final['hpwl_no_io']:<16.2f} {vivado_hpwl:<12.2f} {alpha:<6.2f} {in_allowed_range}")

        return {
            'alpha': alpha,
            'beta': alpha if place_type == PlaceType.IO else 0.0,
            'obj': obj,
            'hpwl_initial': fem_hpwl_initial['hpwl_no_io'],
            'hpwl_final': fem_hpwl_final['hpwl_no_io'],
            'overlap': overlap,
            'overlap_percent': overlap_percent,
            'in_allowed_range': in_allowed_range,
        }

    # --- Coarse sweep ---
    coarse_results = []
    for used_alpha in range(coarse_start, coarse_end, coarse_step):
        res = evaluate_alpha(used_alpha)
        coarse_results.append(res)

    # Choose approximate alpha a1: prefer entries inside allowed overlap range
    in_range_candidates = [r for r in coarse_results if r['in_allowed_range']]
    if len(in_range_candidates) > 0:
        a1 = min(in_range_candidates, key=lambda r: r['hpwl_final'])['alpha']
    else:
        a1 = min(coarse_results, key=lambda r: r['obj'])['alpha']

    # --- Fine sweep around a1 ---
    fine_low = a1 - fine_radius
    fine_high = a1 + fine_radius
    fine_alphas = np.arange(fine_low, fine_high + fine_step/2, fine_step)
    fine_results = []
    seen_alphas = set(r['alpha'] for r in coarse_results)
    for used_alpha in fine_alphas:
        used_alpha = float(used_alpha)
        # Check if already evaluated in coarse sweep (approximately, within 0.01 tolerance)
        if any(abs(sa - used_alpha) < 0.01 for sa in seen_alphas):
            fine_results.append(next(r for r in coarse_results if abs(r['alpha'] - used_alpha) < 0.01))
        else:
            fine_results.append(evaluate_alpha(used_alpha))

    # Rank: prefer in_allowed_range, then lower hpwl_final
    fine_results_sorted = sorted(fine_results, key=lambda r: (0 if r['in_allowed_range'] else 1, r['hpwl_final'], r['overlap']))
    top_results = fine_results_sorted[:top_k]

    # Report top-k and append rows for each
    for idx, r in enumerate(top_results, start=1):
        print(f"Top{idx}: alpha={r['alpha']} hpwl_before={r['hpwl_initial']:.2f} hpwl_after={r['hpwl_final']:.2f} overlap={r['overlap']} ({r['overlap_percent']:.3f})")
        # For designs with IO, record beta separately (use same constraint value as beta)
        beta_val = r['alpha'] if place_type == PlaceType.IO else 0.0
        row = extract_features_from_placer(fpga_placer, hpwl_before=r['hpwl_initial'], hpwl_after=r['hpwl_final'], overlap_after=r['overlap'], instance=instance, alpha=r['alpha'], beta=beta_val)
        append_row(row)
    