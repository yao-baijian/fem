import sys
import os
import io
import contextlib
import json
sys.path.insert(0, '.')

import torch
import itertools
from fem_placer import FPGAPlacementOptimizer, solve_placement_sb
from fem_placer.logger import *
SET_LEVEL('WARNING')

def evaluate_params(J, logic_site_coords, num_inst, lam, mu, agents, max_steps):
    # Suppress output from simulated-bifurcation solver
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        site_indices, grid_coords, energy, meta = solve_placement_sb(
            J, logic_site_coords,
            lam=lam, mu=mu,
            agents=agents, max_steps=max_steps,
            best_only=True,
            verbose=False,
        )
    
    n_unique = len(torch.unique(site_indices))
    overlap = num_inst - n_unique
    return overlap, energy.item()

def tune_parameters_no_vivado(instance, lam_vals, mu_vals, dev='cpu', agents=16, max_steps=1000):
    optimizer = FPGAPlacementOptimizer.from_saved_params(
        f'result/{instance}/init_params.json',
        num_trials=1,
        num_steps=1,
        dev=dev
    )

    J = optimizer.J_LL.cpu()
    num_inst = J.shape[0]
    
    D = optimizer.D_LL.cpu()
    # site coords extraction based on D size implicitly (assumes n_sites grid)
    # The actual FpgaPlacementOptimizer contains n_sites, though we don't have explicit grid width/height
    # However, solve_placement_sb accepts site_coords. We reconstruct it or grab it if stored
    # As a fallback, reconstruct 1D coords for SB just for D computation inside SB matching n_sites
    n_sites = D.shape[0]
    
    # If the user saved explicit site coordinates, you'd typically retrieve them.
    # We will just generate dummy coordinates because the objective largely consumes D directly initially, 
    # but export_placement_qubo needs site coordinates to compute D. 
    # As a workaround, we'll create a 1D mapping to feed export_placement_qubo, but ideally export_placement_qubo 
    # should just take D. Since we must pass coords, let's create a linear sequence that forms the exact distance mapping 
    # or just bypass it. Oh wait, export_placement_qubo strictly calls get_site_distance_matrix. 
    # We should reconstruct a pseudo-grid that matches n_sites.
    # Assuming square grid for the sake of the QUBO solver API if explicit coords are not stored in init_params
    grid_side = int(n_sites**0.5)
    logic_site_coords = torch.cartesian_prod(
        torch.arange(grid_side, dtype=torch.float32),
        torch.arange(n_sites // grid_side, dtype=torch.float32)
    )

    # To be safe, try to match size exactly
    if logic_site_coords.shape[0] != n_sites:
        logic_site_coords = torch.zeros(n_sites, 2)
        logic_site_coords[:, 0] = torch.arange(n_sites)

    J_max = J.max().item()
    D_max = D.max().item()
    max_coupling = J_max * D_max
    print(f"Max coupling (J_max * D_max) approx = {max_coupling:.2f}")
    
    results = []
    print(f"{'lam':<10} | {'mu':<10} | {'Overlap':<10} | {'Energy':<15}")
    print("-" * 50)
    
    for lam, mu in itertools.product(lam_vals, mu_vals):
        if lam >= mu:
            continue
        overlap, energy = evaluate_params(J, logic_site_coords, num_inst, lam, mu, agents, max_steps)
        results.append((lam, mu, overlap, energy))
        print(f"{lam:<10.2f} | {mu:<10.2f} | {overlap:<10} | {energy:<15.2f}")
        
    if not results:
        print("No valid parameter combinations evaluated (all skipped due to lam >= mu).")
        return None, None
        
    best_result = min(results, key=lambda x: (x[2], x[3])) # Primary: min overlap, Secondary: min energy
    print(f"\nBest parameters for {instance}: lam={best_result[0]:.2f}, mu={best_result[1]:.2f} with overlap {best_result[2]}")
    
    return best_result[0], best_result[1]


if __name__ == "__main__":
    test_instances = ['FPGA-example1'] # Replace with actual instances available in result/ directory
    
    lam_values_coarse = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
    mu_values_coarse = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
    
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    summary_results = {}
    config_dir = './scripts/config'
    os.makedirs(config_dir, exist_ok=True)
    
    for instance in test_instances:
        print(f"\n{'='*50}")
        print(f"Starting coarse grain search for {instance}...")
        print(f"{'='*50}")
        
        try:
            best_lam, best_mu = tune_parameters_no_vivado(instance, lam_values_coarse, mu_values_coarse, dev=dev, agents=16, max_steps=1000)
        except Exception as e:
            print(f"Skipping {instance} due to error (perhaps init_params not found): {e}")
            continue
            
        if best_lam is None:
            continue
            
        print(f"\nStarting fine grain search around best parameters for {instance}...")
        lam_step = max(best_lam * 0.2, 10.0)
        mu_step = max(best_mu * 0.2, 10.0)
        
        lam_values_fine = [max(best_lam - lam_step, 0.1), best_lam, best_lam + lam_step]
        mu_values_fine = [max(best_mu - mu_step, 0.1), best_mu, best_mu + mu_step]
        
        final_lam, final_mu = tune_parameters_no_vivado(instance, lam_values_fine, mu_values_fine, dev=dev, agents=16, max_steps=2000)
        
        if final_lam is not None:
            print(f"\nFinal best parameters for {instance} after fine-tuning:")
            print(f"lam = {final_lam}")
            print(f"mu = {final_mu}")
            
            summary_results[instance] = {
                "lam": final_lam,
                "mu": final_mu
            }
    
    summary_path = os.path.join(config_dir, 'bsb_summary_no_vivado.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nAll instances processed. Summary saved to {summary_path}")
