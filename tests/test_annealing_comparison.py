"""
Test script for comparing FPGA placement optimizer performance across different annealing schedules.

This script runs the FPGAPlacementOptimizer with three different annealing strategies
(linear, exponential, and inverse) and visualizes the total loss comparison to determine
which annealing schedule achieves the best convergence.
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    FPGAPlacementOptimizer
)
from fem_placer.logger import *
from fem_placer.config import *
from fem_placer.objectives import get_loss_history, clear_history
from ml.dataset import *
from ml.predict import predict_alpha

SET_LEVEL('INFO')

# Configuration
INSTANCE = 'c5315'
NUM_TRIALS = 10
NUM_STEPS = 200
DEVICE = 'cpu'
MANUAL_GRAD = False
ANNEAL_TYPES = ['lin', 'exp', 'inverse']
DRAW_COMPARISON = True
SAVE_COMPARISON_PATH = 'result/annealing_comparison.png'

def run_optimization_with_anneal(fpga_placer, instance, anneal_type, num_trials, num_steps, device):
    """
    Run a single optimization with specified annealing schedule.
    
    Args:
        fpga_placer: FpgaPlacer instance
        instance: Instance name (for logging)
        anneal_type: Type of annealing ('lin', 'exp', or 'inverse')
        num_trials: Number of trials
        num_steps: Number of optimization steps
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        loss_history: Dictionary containing loss histories
        optimizer_result: Result from the optimizer
        config: Configuration from the optimizer
    """
    INFO(f'Running optimization for {instance} with {anneal_type.upper()} annealing...')
    
    # Clear history before each run
    clear_history()
    
    optimizer = FPGAPlacementOptimizer(
        num_inst=fpga_placer.opti_insts_num,
        num_fixed_inst=fpga_placer.fixed_insts_num,
        num_site=fpga_placer.get_grid('logic').area,
        coupling_matrix=fpga_placer.net_manager.insts_matrix,
        site_coords_matrix=fpga_placer.logic_site_coords,
        io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
        io_site_coords=fpga_placer.io_site_coords,
        bbox_length=fpga_placer.grids['logic'].area_length,
        constraint_alpha=fpga_placer.constraint_alpha,
        constraint_beta=fpga_placer.constraint_alpha,
        num_trials=num_trials,
        num_steps=num_steps,
        dev=device,
        betamin=0.01,
        betamax=0.5,
        anneal=anneal_type,  # Use the specified annealing type
        optimizer='adam',
        learning_rate=0.1,
        h_factor=0.01,
        seed=1,
        dtype=torch.float32,
        with_io=False,
        manual_grad=MANUAL_GRAD
    )
    
    config, result = optimizer.optimize()
    loss_history = get_loss_history()
    
    return loss_history, result, config


def main():
    # Initialize FPGA placer once (same instance for all annealing runs)
    place_type = PlaceType.CENTERED
    debug = False
    
    INFO(f'Initializing FPGA placer for instance {INSTANCE}...')
    fpga_placer = FpgaPlacer(
        place_type,
        GridType.SQUARE,
        0.4,
        debug,
        device=DEVICE
    )
    
    vivado_hpwl, site_num, site_net_num, total_net_num = fpga_placer.init_placement(
        f'./vivado/output_dir/{INSTANCE}/post_impl.dcp',
        f'./vivado/output_dir/{INSTANCE}/optimized_placement.pl'
    )
    
    area_size = fpga_placer.grids['logic'].area
    
    # Predict alpha for constraint weighting
    row = extract_features_from_placer(fpga_placer, hpwl_before=0, hpwl_after=0, overlap_after=0, alpha=0)
    alpha = predict_alpha(row)
    INFO(f'Predicted alpha: {alpha}')
    fpga_placer.set_alpha(alpha)
    
    # Create drawer for visualization
    global_drawer = PlacementDrawer(placer=fpga_placer)
    
    # Run optimizations with different annealing schedules
    annealing_results = {}
    optimization_metrics = {}
    
    for anneal_type in ANNEAL_TYPES:
        loss_history, result, config = run_optimization_with_anneal(
            fpga_placer,
            INSTANCE,
            anneal_type,
            NUM_TRIALS,
            NUM_STEPS,
            DEVICE
        )
        
        annealing_results[anneal_type] = loss_history
        
        # Calculate metrics
        final_loss = loss_history['total_losses'][-1]
        min_loss = min(loss_history['total_losses'])
        avg_loss = sum(loss_history['total_losses']) / len(loss_history['total_losses'])
        
        optimization_metrics[anneal_type] = {
            'final_loss': final_loss,
            'min_loss': min_loss,
            'avg_loss': avg_loss,
            'result': result,
            'config': config
        }
        
        INFO(f'{anneal_type.upper()} Annealing Results:')
        INFO(f'  Final Loss: {final_loss:.6f}')
        INFO(f'  Minimum Loss: {min_loss:.6f}')
        INFO(f'  Average Loss: {avg_loss:.6f}')
    
    # Print comparison summary
    print("\n" + "="*70)
    print(f"{'Annealing Type':<15} {'Final Loss':<18} {'Min Loss':<18} {'Avg Loss':<18}")
    print("="*70)
    
    best_anneal = None
    best_final_loss = float('inf')
    
    for anneal_type in ANNEAL_TYPES:
        metrics = optimization_metrics[anneal_type]
        print(f"{anneal_type.upper():<15} {metrics['final_loss']:<18.6f} "
              f"{metrics['min_loss']:<18.6f} {metrics['avg_loss']:<18.6f}")
        
        if metrics['final_loss'] < best_final_loss:
            best_final_loss = metrics['final_loss']
            best_anneal = anneal_type
    
    print("="*70)
    print(f"\nBest performing annealing: {best_anneal.upper()} with final loss: {best_final_loss:.6f}\n")
    
    # Draw comparison plot
    if DRAW_COMPARISON:
        INFO(f'Generating annealing comparison plot...')
        global_drawer.plot_annealing_comparison(
            annealing_results,
            save_path=SAVE_COMPARISON_PATH
        )
    
    # Optionally, legalize and route the best result
    best_metrics = optimization_metrics[best_anneal]
    optimal_inds = torch.argwhere(best_metrics['result'] == best_metrics['result'].min()).reshape(-1)
    
    legalizer = Legalizer(placer=fpga_placer, device=DEVICE)
    router = Router(placer=fpga_placer)
    logic_ids, io_ids = fpga_placer.get_ids()
    
    real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(
        best_metrics['config'][optimal_inds[0]]
    )
    
    placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(
        real_logic_coords,
        logic_ids
    )
    
    routes = router.route_connections(
        fpga_placer.net_manager.insts_matrix,
        placement_legalized[0]
    )
    
    INFO(f'\nFinal Results with Best Annealing ({best_anneal.upper()}):')
    print(f"{'Benchmark':<12} {INSTANCE:<10} {site_num:<6} {f'{site_net_num}/{total_net_num}':<14} "
          f"{overlap:<8} {fem_hpwl_initial['hpwl_no_io']:<18.2f} "
          f"{fem_hpwl_final['hpwl_no_io']:<16.2f} {vivado_hpwl:<12.2f}")


if __name__ == '__main__':
    main()
