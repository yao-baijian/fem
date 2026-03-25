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

SET_LEVEL('WARNING')

num_trials = 5
num_steps = 200
dev = 'cuda'
manual_grad = False
anneal = 'inverse'
case_type = 'fpga_placement'

# Configuration
RESULT_DIR = './result'
# Reuse the same coarse cache produced by test_train_alpha_logic_io.py
COARSE_CACHE_FILE = os.path.join(RESULT_DIR, 'coarse_results_io.json')
USE_COARSE_CACHE = True

# Candidate io_factor sweep configuration (start, end, step)
IO_FACTOR_START = 10.0
IO_FACTOR_END = 600.0
IO_FACTOR_STEP = 30.0

instances = ['c2670_boundary', 'c5315_boundary', 'c6288_boundary', 'c7552_boundary',
             's1488_boundary', 's5378_boundary', 's9234_boundary', 's15850_boundary']

# Load coarse cache if available
coarse_cache = {}
if USE_COARSE_CACHE and os.path.exists(COARSE_CACHE_FILE):
    try:
        with open(COARSE_CACHE_FILE, 'r') as f:
            coarse_cache = json.load(f)
        print(f"Loaded coarse cache from {COARSE_CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not load coarse cache: {e}")

print(f"{'Instance':<15} | {'Best HPWL':<10} | {'OvL':<6} | {'OvIO':<6} | {'OvL %':<8} | {'OvIO %':<8} | {'Alpha':<10} | {'Beta':<10} | {'IoFac':<10} | {'Time (s)':<10}")
print("-" * 128)

for instance in instances:
    start_time = time.time()
    place_type = PlaceType.IO
    debug = False

    fpga_placer = FpgaPlacer(
        place_orientation=place_type,
        grid_type=GridType.SQUARE,
        place_mode=IoMode.VIRTUAL_NODE,
        utilization_factor=0.4,
        debug=debug,
        device=dev,
    )

    fpga_placer.set_instance_name(instance)

    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(
        f'./vivado/output_dir/{instance}/post_impl.dcp',
        f'./vivado/output_dir/{instance}/optimized_placement.pl',
    )

    # Get alpha/beta from coarse cache (same as in test_train_alpha_logic_io.py)
    cache_key = instance
    if cache_key not in coarse_cache:
        print(f"[WARN] No coarse cache entry for instance {instance}, skipping.")
        continue

    a1 = coarse_cache[cache_key]['a1']
    b1 = coarse_cache[cache_key].get('b1', 0.0)

    # Allowed overlap range (same as logic+IO training script)
    overlap_allowed_min = 0
    overlap_allowed_max = 0.08

    def evaluate_placement(io_factor: float, alpha: float, beta: float):
        """Evaluate placement for given io_factor while keeping alpha/beta fixed.

        This mirrors the evaluate_placement in test_train_alpha_logic_io.py
        but sweeps io_factor instead of alpha/beta.
        """
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
            constraint_beta=beta,
            num_trials=num_trials,
            num_steps=num_steps,
            dev=dev,
            betamin=0.01,
            betamax=0.5,
            anneal=anneal,
            optimizer='adam',
            learning_rate=0.1,
            h_factor=0.01,
            io_factor=io_factor,
            seed=1,
            dtype=torch.float32,
            with_io=(place_type == PlaceType.IO),
            manual_grad=manual_grad,
        )

        config, result = optimizer.optimize()
        optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
        legalizer = Legalizer(placer=fpga_placer, device=dev)
        logic_ids, io_ids = fpga_placer.get_ids()

        # Default overlap counts
        overlap_logic = 0
        overlap_io = 0

        # Always run full legalization here for stable comparison, but
        # compute overlap using pre-legalization coordinates.
        fpga_placer.grids['logic'].clear_all()
        if place_type == PlaceType.IO:
            fpga_placer.grids['io'].clear_all()

        if place_type == PlaceType.IO:
            real_logic_coords = config[0][optimal_inds[0]]
            real_io_coords = config[1][optimal_inds[0]]

            # Overlap before legalization
            n_unique_logic = len(torch.unique(real_logic_coords, dim=0))
            overlap_logic = real_logic_coords.shape[0] - n_unique_logic

            n_unique_io = len(torch.unique(real_io_coords, dim=0))
            overlap_io = real_io_coords.shape[0] - n_unique_io
            overlap = overlap_logic + overlap_io

            # Legalize to obtain final HPWL
            placement_legalized, _, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
                real_logic_coords,
                logic_ids,
                real_io_coords,
                io_ids,
                include_io=True,
            )
        else:
            real_logic_coords = config[optimal_inds[0]]

            # Overlap before legalization
            n_unique_logic = len(torch.unique(real_logic_coords, dim=0))
            overlap_logic = real_logic_coords.shape[0] - n_unique_logic
            overlap = overlap_logic

            # Legalize to obtain final HPWL
            placement_legalized, _, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
                real_logic_coords,
                logic_ids,
            )

        # Compute overlap percentages separately for logic and IO
        logic_inst_total = float(inst_num['logic_inst_num']) if inst_num['logic_inst_num'] > 0 else 1.0
        io_inst_total = float(inst_num['io_inst_num']) if inst_num.get('io_inst_num', 0) > 0 else 1.0

        overlap_percent = float(overlap_logic) / logic_inst_total
        io_overlap_percent = (
            float(overlap_io) / io_inst_total if inst_num.get('io_inst_num', 0) > 0 else 0.0
        )

        in_allowed_range = (
            overlap_percent >= overlap_allowed_min and overlap_percent <= overlap_allowed_max
        )
        obj = (
            fem_hpwl_final['hpwl']
            if in_allowed_range
            else (fem_hpwl_final['hpwl'] + (overlap * 10))
        )

        return {
            'alpha': alpha,
            'beta': beta,
            'io_factor': io_factor,
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

    # Sweep io_factor values for this instance using step-length search
    results = []
    io_fac = IO_FACTOR_START
    while io_fac <= IO_FACTOR_END + 1e-9:
        res = evaluate_placement(io_fac, a1, b1)
        results.append(res)
        io_fac += IO_FACTOR_STEP

    # Rank by allowed range, then HPWL, then total overlap
    in_range_candidates = [r for r in results if r['in_allowed_range']]
    if len(in_range_candidates) > 0:
        best = min(in_range_candidates, key=lambda r: r['hpwl_final'])
    else:
        best = min(results, key=lambda r: r['obj'])

    total_time = time.time() - start_time

    print(
        f"{instance:<15} | "
        f"{best['hpwl_final']:<10.2f} | "
        f"{best['overlap_logic']:<6} | "
        f"{best['overlap_io']:<6} | "
        f"{best['overlap_percent']*100:<8.2f}% | "
        f"{best['io_overlap_percent']*100:<8.2f}% | "
        f"{best['alpha']:<10.1f} | "
        f"{best['beta']:<10.1f} | "
        f"{best['io_factor']:<10.1f} | "
        f"{total_time:<10.2f}"
    )
