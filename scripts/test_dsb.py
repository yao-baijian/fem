"""
Test script for FPGA placement using discrete Simulated Bifurcation (dSB).

This script solves the placement problem by constructing a full QUBO matrix
with one-hot and at-most-one constraints, then solving it with the
simulated-bifurcation library (heated discrete mode).

Uses the master branch FpgaPlacer API (net_manager for coupling matrix).
"""

import sys
import os
import json
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    Legalizer,
    solve_placement_sb
)
from fem_placer.logger import *
SET_LEVEL('WARNING')  # Set higher to suppress unnecessary logs from libraries

# Configuration
instances = ['c2670', 'c5315', 'c6288', 'c7552',
             's1488', 's5378', 's9234', 's15850']  # Change to your desired instance

# instances = ['c1355', 'c2670', 'c5315', 'c6288']

agents = 32
max_steps = 1000
default_lam = 0.1        # one-hot constraint weight, fallback default
default_mu = 1200         # at-most-one constraint weight, fallback default
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = './scripts/config/bsb_summary.json'
bsb_config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        bsb_config = json.load(f)

print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Overlap':<8} "
      f"{'HPWL Init':<18} {'HPWL Final':<16} {'QUBO Energy':<12}")

for instance in instances:
    lam = bsb_config.get(instance, {}).get('lam', default_lam)
    mu = bsb_config.get(instance, {}).get('mu', default_mu)
    
    dcp_file = f'./vivado/output_dir/{instance}/post_impl.dcp'
    output_file = f'bsb_placement_{instance}.dcp'
    INFO(f"Processing instance: {dcp_file}")
    fpga_placer = FpgaPlacer(utilization_factor=0.4)
    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(dcp_file=dcp_file, dcp_output=output_file)

    J = fpga_placer.net_manager.insts_matrix
    num_inst = fpga_placer.instances['logic'].num
    # Get grid information
    logic_grid = fpga_placer.get_grid('logic')
    # Create site coordinates matrix
    logic_site_coords = torch.cartesian_prod(
        torch.arange(logic_grid.width, dtype=torch.float32),
        torch.arange(logic_grid.height, dtype=torch.float32)
    )

    # QUBO size warning
    n_sites = logic_site_coords.shape[0]
    qubo_dim = num_inst * n_sites + n_sites
    qubo_memory_gb = qubo_dim ** 2 * 4 / 1024 ** 3
    INFO(f"qubo matrix size: {qubo_dim} x {qubo_dim}, memory: ~{qubo_memory_gb:.2f} GB")
    INFO(f"dsb params: agents={agents}, max_steps={max_steps}, lam={lam}, mu={mu}")

    # Solve with dSB
    INFO("solving placement with dsb...")
    site_indices, grid_coords, energy, meta = solve_placement_sb(
        J, logic_site_coords,
        lam=lam, mu=mu,
        agents=agents, max_steps=max_steps,
        best_only=True,
    )
    # Check feasibility
    n_unique = len(torch.unique(site_indices))
    INFO(f"Unique sites used: {n_unique} / {num_inst} instances")
    if n_unique < num_inst:
        INFO(f"Only {n_unique} distinct sites — {num_inst - n_unique} instances overlap")
    INFO(f"QUBO energy: {energy:.4f}")

    # Legalizing
    INFO("Legalizing placement...")
    coords = logic_grid.to_real_coords_tensor(grid_coords)
    logic_ids, io_ids = fpga_placer.get_ids()
    legalizer = Legalizer(placer=fpga_placer, device=dev)
    placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
        coords, logic_ids
    )

    inst_num = inst_num['logic_inst_num']
    hpwl_before = hpwl_before['hpwl_no_io']
    hpwl_after = hpwl_after['hpwl_no_io']

    print(f"{'Benchmarks':<12} {instance:<10} {inst_num:<6} {overlap:<8} "
        f"{hpwl_before:<18.2f} {hpwl_after:<16.2f} {energy:<12.2f}")
    

    # print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Site/Total':<14} {'Overlap':<8} "
    #     f"{'HPWL Init':<18} {'HPWL Final':<16} {'QUBO Energy':<12}")
