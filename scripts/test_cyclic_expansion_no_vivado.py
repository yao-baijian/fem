import io
import sys
import os
import contextlib
import json
import time
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FPGAPlacementOptimizer,
    solve_placement_cyclic
)
from fem_placer.logger import *
from scripts.qubo_utils import reconstruct_logic_site_coords
SET_LEVEL('WARNING')

# Configuration
# instances = ['c2670', 'FPGA-example1']
instances = ['c2670', 'c2670', 'c5315', 'c6288', 'c7552',
             's1488', 's5378', 's9234', 's15850', 'bgm', 'sha1', 'RLE_BlobMerging', 'FPGA-example1']

max_iters = 200
k = 60          # instances per iteration
k_u = 30        # unbound sites per iteration
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Overlap':<8} {'QUBO Energy':<12} {'Iters':<6} {'QUBO Mem(MB)':<12} {'Time (s)':<10}")

for instance in instances:
    try:
        optimizer = FPGAPlacementOptimizer.from_saved_params(
            f'result/{instance}/init_params.json',
            num_trials=1,
            num_steps=1,
            dev=dev
        )
    except Exception as e:
        print(f"Skipping {instance} (init_params.json not found or error loading): {e}")
        continue

    J = optimizer.coupling_matrix.cpu()
    n_sites = optimizer.num_site
    num_inst = optimizer.num_inst
    
    # Recreate coordinates strictly to match constraint expectations
    logic_site_coords = reconstruct_logic_site_coords(n_sites)

    # Essentially, Cyclic Expansion maximum sub QUBO dim is len(cycles) <= k / 2 + k_u
    # Typically k=60, k_u=30 -> s <= 60
    # Memory footprint is essentially bounded by distance matrices (N x N) and flow matrices (M x M)
    # The sub-QUBO is O(s^2). But let's report the expected SBM QUBO size equivalent for fair comparison.
    qubo_dim_equiv = num_inst * n_sites + n_sites
    SBM_qubo_memory_mb = qubo_dim_equiv ** 2 * 4 / 1024 ** 2
    
    # Actually, the overall cyclic solver maintains the flow (F, M x M) and distance (D, N x N)
    # matrices in addition to permutations. Memory footprint is heavily populated by:
    # `D`: [N, N], `J`: [M, M], `perm`: [M], etc.
    # We will compute the memory for matrices (dominated by D and F). They are persistent.
    
    num_elements = (num_inst * num_inst) + (n_sites * n_sites)
    cyclic_mem_mb = (num_elements * 4) / (1024 ** 2)

    start_time = time.time()
    # with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    site_indices, grid_coords, energy, meta = solve_placement_cyclic(
        J, logic_site_coords,
        k=k, k_u=k_u, max_iters=max_iters,
        seed=42, verbose=False,
    )
    elapsed_time = time.time() - start_time
    
    # Check feasibility
    n_unique = len(torch.unique(site_indices))
    overlap = num_inst - n_unique
    iterations = meta.get('iterations', max_iters)
    
    print(f"{'Benchmarks':<12} {instance:<10} {num_inst:<6} {overlap:<8} {energy:<12.2f} {iterations:<6} {cyclic_mem_mb:<12.2f} {elapsed_time:<10.2f}")

