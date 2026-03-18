import sys
import os
import io
import contextlib
import json
sys.path.insert(0, '.')

import torch
import itertools
from fem_placer import FpgaPlacer, solve_placement_sb
from fem_placer.logger import *
SET_LEVEL('WARNING')  # Set higher to suppress unnecessary logs from libraries

def evaluate_params(J, logic_site_coords, num_inst, lam, mu, agents, max_steps):
    # Suppress output from simulated-bifurcation solver
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        site_indices, grid_coords, energy, meta = solve_placement_sb(
            J, logic_site_coords,
            lam=lam, mu=mu,
            agents=agents, max_steps=max_steps,
            best_only=True,
            verbose=False, # Attempt to disable verbose if supported
        )
    
    n_unique = len(torch.unique(site_indices))
    overlap = num_inst - n_unique
    return overlap, energy.item()

def tune_parameters(instance, lam_vals, mu_vals, agents=16, max_steps=1000):
    dcp_file = f'./vivado/output_dir/{instance}/post_impl.dcp'
    fpga_placer = FpgaPlacer(utilization_factor=0.3)
    _, inst_num, _ = fpga_placer.init_placement(dcp_file=dcp_file, dcp_output=None)

    J = fpga_placer.net_manager.insts_matrix
    num_inst = fpga_placer.instances['logic'].num

    logic_grid = fpga_placer.get_grid('logic')
    logic_site_coords = torch.cartesian_prod(
        torch.arange(logic_grid.width, dtype=torch.float32),
        torch.arange(logic_grid.height, dtype=torch.float32)
    )

    J_max = J.max().item()
    D = torch.cdist(logic_site_coords, logic_site_coords)
    D_max = D.max().item()
    max_coupling = J_max * D_max
    INFO(f"Max coupling (J_max * D_max) approx = {max_coupling:.2f}")
    
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

def save_config(lam, mu, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"lam={lam}\n")
        f.write(f"mu={mu}\n")
    print(f"Saved configuration to {filepath}")

if __name__ == "__main__":
    test_instances = ['c2670', 'c5315', 'c6288', 'c7552', 's1488', 's5378', 's9234', 's15850']
    
    lam_values_coarse = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
    mu_values_coarse = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
    
    summary_results = {}
    config_dir = './scripts/config'
    os.makedirs(config_dir, exist_ok=True)
    
    for instance in test_instances:
        print(f"\n{'='*50}")
        print(f"Starting coarse grain search for {instance}...")
        print(f"{'='*50}")
        
        best_lam, best_mu = tune_parameters(instance, lam_values_coarse, mu_values_coarse, agents=16, max_steps=1000)
        
        if best_lam is None:
            continue
            
        print(f"\nStarting fine grain search around best parameters for {instance}...")
        # Create fine-grained grid around the best coarse result
        lam_step = max(best_lam * 0.2, 10.0)
        mu_step = max(best_mu * 0.2, 10.0)
        
        lam_values_fine = [max(best_lam - lam_step, 0.1), best_lam, best_lam + lam_step]
        mu_values_fine = [max(best_mu - mu_step, 0.1), best_mu, best_mu + mu_step]
        
        final_lam, final_mu = tune_parameters(instance, lam_values_fine, mu_values_fine, agents=16, max_steps=2000)
        
        if final_lam is not None:
            print(f"\nFinal best parameters for {instance} after fine-tuning:")
            print(f"lam = {final_lam}")
            print(f"mu = {final_mu}")
            
            summary_results[instance] = {
                "lam": final_lam,
                "mu": final_mu
            }
    
    summary_path = os.path.join(config_dir, 'bsb_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nAll instances processed. Summary saved to {summary_path}")
