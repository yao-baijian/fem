"""
Test script for FPGA placement using CyclicExpansion.

This script solves the placement problem using the CyclicExpansion algorithm
(arXiv:2312.15467) which iteratively solves small 2-cycle swap sub-problems
instead of one large QUBO.

2-cycles naturally preserve permutation feasibility — no constraint weights needed.

Uses the master branch FpgaPlacer API (net_manager for coupling matrix).
"""

import sys
import time
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    Legalizer,
    Router,
    PlacementDrawer,
    solve_placement_cyclic,
)
from fem_placer.logger import *
SET_LEVEL('INFO')

# instances = ['c1355', 'c2670', 'c5315', 'c6288', 'c7552',
#              's1238', 's1488', 's5378', 's9234', 's15850']

# instances = ['bgm', 'sha1', 'diffeq_f_systemC', 'FPGA-example1']
# instances = ['RLE_BlobMerging']
instances = ['c2670', 'c2670', 'c5315', 'c6288', 'c7552',
             's1488', 's5378', 's9234', 's15850', 'bgm', 'sha1', 'RLE_BlobMerging', 'FPGA-example1']
# Configuration
max_iters = 200
k = 60          # instances per iteration
k_u = 30        # unbound sites per iteration
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Overlap':<8} "
      f"{'HPWL Init':<14} {'HPWL Final':<14} {'QAP Cost':<12} {'Iters':<6} {'Time (s)':<10}")

for instance in instances:
    dcp_file = f'./vivado/output_dir/{instance}/post_impl.dcp'
    output_file = f'optimized_placement_cyclic_{instance}.dcp'
    fpga_placer = FpgaPlacer(utilization_factor=0.3)

    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(
        dcp_file=dcp_file,
        dcp_output=output_file
    )

    # Get coupling matrix from net_manager
    J = fpga_placer.net_manager.insts_matrix
    num_inst = fpga_placer.instances['logic'].num

    # Get grid information
    logic_grid = fpga_placer.get_grid('logic')
    logic_site_coords = torch.cartesian_prod(
        torch.arange(logic_grid.width, dtype=torch.float32),
        torch.arange(logic_grid.height, dtype=torch.float32)
    )
    INFO(f"CyclicExpansion params: k={k}, k_u={k_u}, max_iters={max_iters}")

    # Solve with CyclicExpansion
    INFO("Solving placement with CyclicExpansion...")
    start_time = time.time()
    site_indices, grid_coords, energy, meta = solve_placement_cyclic(
        J, logic_site_coords,
        k=k, k_u=k_u, max_iters=max_iters,
        seed=42, verbose=True,
    )
    elapsed_time = time.time() - start_time

    # Check feasibility (CyclicExpansion guarantees unique sites by construction)
    n_unique = len(torch.unique(site_indices))
    INFO(f"Unique sites used: {n_unique} / {num_inst} instances")
    if n_unique < num_inst:
        INFO(f"Only {n_unique} distinct sites — {num_inst - n_unique} instances overlap")

    INFO("Legalizing placement...")
    coords = logic_grid.to_real_coords_tensor(grid_coords)
    
    # Legalize placement
    INFO("Legalizing placement...")
    legalizer = Legalizer(placer=fpga_placer, device=dev)
    logic_ids, io_ids = fpga_placer.get_ids()
    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        coords, logic_ids
    )

    inst_num = inst_num['logic_inst_num']
    iterations = meta['iterations']
    hpwl_before = hpwl_before['hpwl_no_io']
    hpwl_after = hpwl_after['hpwl_no_io']

    print(f"{'Benchmarks':<12} {instance:<10} {inst_num:<6} {overlap:<8}"
        f"{hpwl_before:<14.2f} {hpwl_after:<14.2f} {energy:12.2f} {iterations:<6} {elapsed_time:<10.2f}")
