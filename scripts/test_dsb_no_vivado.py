import sys
import os
import json
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FPGAPlacementOptimizer,
    solve_placement_sb
)
from fem_placer.logger import *
SET_LEVEL('WARNING')

# Configuration
instances = ['FPGA-example1']  # Change to your target instances placed in result/...

agents = 32
max_steps = 1000
default_lam = 0.1
default_mu = 1200
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

config_path = './scripts/config/bsb_summary_no_vivado.json'
bsb_config = {}
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        bsb_config = json.load(f)

print(f"{'Benchmarks':<12} {'Instance':<10} {'Inst':<6} {'Overlap':<8} {'QUBO Energy':<12}")

for instance in instances:
    lam = bsb_config.get(instance, {}).get('lam', default_lam)
    mu = bsb_config.get(instance, {}).get('mu', default_mu)
    
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

    J = optimizer.J_LL.cpu()
    D = optimizer.D_LL.cpu()
    num_inst = J.shape[0]
    n_sites = D.shape[0]
    
    # Recreate coordinates strictly to match constraint expectations inside SB
    grid_side = int(n_sites**0.5)
    logic_site_coords = torch.cartesian_prod(
        torch.arange(grid_side, dtype=torch.float32),
        torch.arange(n_sites // grid_side, dtype=torch.float32)
    )

    if logic_site_coords.shape[0] != n_sites:
        logic_site_coords = torch.zeros(n_sites, 2)
        logic_site_coords[:, 0] = torch.arange(n_sites)

    qubo_dim = num_inst * n_sites + n_sites
    qubo_memory_gb = qubo_dim ** 2 * 4 / 1024 ** 3
    print(f"Processing {instance}...")
    print(f"qubo matrix size: {qubo_dim} x {qubo_dim}, memory: ~{qubo_memory_gb:.2f} GB")
    print(f"dsb params: agents={agents}, max_steps={max_steps}, lam={lam}, mu={mu}")

    # Solve with dSB
    site_indices, grid_coords, energy, meta = solve_placement_sb(
        J, logic_site_coords,
        lam=lam, mu=mu,
        agents=agents, max_steps=max_steps,
        best_only=True,
    )
    
    # Check feasibility
    n_unique = len(torch.unique(site_indices))
    overlap = num_inst - n_unique
    
    print(f"{'Benchmarks':<12} {instance:<10} {num_inst:<6} {overlap:<8} {energy:<12.2f}")
