"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
"""

import sys
sys.path.append('.')
from FEM import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.drawer import PlacementDrawer
from FEM.placement.legalizer import Legalizer
from FEM.placement.router import Router
from FEM.placement.logger import *
from FEM.placement.config import *
from FEM.ml_alpha.dataset import *
from FEM.ml_alpha.predict import predict_alpha
# sys.path.insert(0, '.')

import torch
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    FPGAPlacementOptimizer
)
from fem_placer.utils import parse_fpga_design


instances = ['c5315']
            # , 'c1355', 'c2670', 'c5315', 'c6288', 'c7552'
            #  's713', 's1238', 's1488', 's5378', 's9234', 's15850', 'FPGA-example1']
SET_LEVEL('INFO')
# # Configuration
# num_trials = 10
# num_steps = 2000
# dev = 'cpu'

# Initialize FPGA placer
print("=" * 80)
print("Testing FPGA Placement Optimizer on Real FPGA Design")
print("=" * 80)

for instance in instances:
    place_type = PlaceType.IO
    debug = False
    fpga_placer = FpgaPlacer(place_type, 
                            GridType.SQUARE,
                            0.4,
                            debug,
                            device=dev)
    
    vivado_hpwl, site_num, site_net_num, total_net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    area_size = fpga_placer.grids['logic'].area
    global_drawer = PlacementDrawer(placer=fpga_placer)
    
    row = extract_features_from_placer(fpga_placer, hpwl_before=0, hpwl_after=0, overlap_after=0, alpha=0)
    alpha = predict_alpha(row)
    INFO(f'instance {instance}, predicted alpha {alpha}')
    fpga_placer.set_alpha(alpha)
    
    case_placements = FEM.from_file(case_type, instance, fpga_placer, index_start=1)

    case_placements.set_up_solver(num_trials, num_steps, betamin=0.001, betamax=0.5, anneal=anneal, dev=dev, q=area_size, 
                                manual_grad=manual_grad, drawer=global_drawer)
    config, result = case_placements.solve()
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)

    legalizer = Legalizer(placer=fpga_placer,
                        device=dev)
    router = Router(placer=fpga_placer)
    logic_ids, io_ids = fpga_placer.get_ids()

    if place_type == PlaceType.IO:
        real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[0][optimal_inds[0]])
        real_io_coords = fpga_placer.get_grid('io').to_real_coords_tensor(config[1][optimal_inds[0]])
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids, real_io_coords, io_ids, include_io = True)
        all_coords = torch.cat([placement_legalized[0], placement_legalized[1]], dim=0)
        routes = router.route_connections(fpga_placer.net_manager.insts_matrix, all_coords)
    else:
        real_logic_coords = fpga_placer.get_grid('logic').to_real_coords_tensor(config[optimal_inds[0]])
        placement_legalized, overlap, fem_hpwl_initial, fem_hpwl_final = legalizer.legalize_placement(real_logic_coords, logic_ids)
        routes = router.route_connections(fpga_placer.net_manager.insts_matrix, (placement_legalized[0]))

    print(f"{'Benchmarks':<12} {instance:<10} {site_num:<6} {f'{site_net_num}/{total_net_num}':<14} {overlap:<8} "
            f"{fem_hpwl_initial['hpwl_no_io']:<18.2f} {fem_hpwl_final['hpwl_no_io']:<16.2f} {vivado_hpwl:<12.2f}")
    
    if draw_loss_function:
        global_drawer.plot_fpga_placement_loss('hpwl_loss.png')

    if draw_evolution:
        global_drawer.draw_multi_step_placement('placement_evolution.png')

    if draw_final_placement:
        global_drawer.draw_place_and_route(placement_legalized[0], routes, None, False, 1000, title_suffix="Final Placement with Routing")
            

        
        
        
# print("\nINFO: Loading FPGA design...")
# fpga_wrapper = FpgaPlacer()
# fpga_wrapper.init_placement('./vivado/output_dir/post_impl.dcp', 'optimized_placement.dcp')

# # Parse FPGA design to get coupling matrix
# num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)

# # Get layout dimensions
# area_length = fpga_wrapper.bbox['area_length']

# print(f"INFO: Number of instances: {num_inst}")
# print(f"INFO: Number of sites: {num_site}")
# print(f"INFO: Area length: {area_length}")
# print(f"INFO: Grid size: {area_length} × {area_length} = {area_length ** 2} positions")

# # Create site coordinates matrix for all grid positions
# print("\nINFO: Creating site coordinates matrix...")
# site_coords_matrix = torch.cartesian_prod(
#     torch.arange(area_length, dtype=torch.float32),
#     torch.arange(area_length, dtype=torch.float32)
# )
# print(f"INFO: Site coordinates shape: {site_coords_matrix.shape}")

# # Set up visualization
# global_drawer = PlacementDrawer(placer=fpga_wrapper, num_subplots=5, debug_mode=False)

# # Create optimizer
# optimizer = FPGAPlacementOptimizer(
#     num_inst=num_inst,
#     num_site=site_coords_matrix.shape[0],
#     coupling_matrix=J,
#     site_coords_matrix=site_coords_matrix,
#     drawer=None,
#     visualization_steps=[0, 250, 500, 750, 999],
#     constraint_weight=1.0
# )

# # Solve with optimizer
# print("\nINFO: Starting FEM optimization...")
# config, result = optimizer.optimize(
#     num_trials=num_trials,
#     num_steps=num_steps,
#     dev=dev,
#     area_width=area_length,
#     betamin=0.01,
#     betamax=0.5,
#     anneal='inverse',
#     optimizer='adam',
#     learning_rate=0.1,
#     h_factor=0.01,
#     seed=1,
#     dtype=torch.float32
# )

# # Find optimal solution
# optimal_inds = torch.argwhere(result == result.min()).reshape(-1)
# print(f"\nINFO: Optimal indices: {optimal_inds.tolist()} with min HPWL: {result.min():.2f}")
# grid_coords = config[optimal_inds[0]]  # Shape: [num_inst, 2]

# # Convert grid coordinates to real FPGA coordinates
# print("INFO: Converting grid coordinates to real coordinates...")
# logic_grid = fpga_wrapper.get_grid('logic')
# real_coords = logic_grid.to_real_coords_tensor(grid_coords)

# # Legalize placement
# print("INFO: Legalizing placement...")
# legalizer = Legalizer(placer=fpga_wrapper, device=dev)
# logic_ids = torch.arange(num_inst)
# placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
#     real_coords, logic_ids
# )

# # Calculate final HPWL
# print(f"INFO: HPWL before legalization: {hpwl_before['hpwl']:.2f}")
# print(f"INFO: HPWL after legalization: {hpwl_after['hpwl']:.2f}")
# print(f"INFO: Overlap violations: {overlap}")

# # Route connections
# print("INFO: Routing connections...")
# router = Router(placer=fpga_wrapper)
# routes = router.route_connections(J, placement_legalized[0])

# # Visualize final result
# global_drawer.draw_place_and_route(
#     logic_coords=placement_legalized[0],
#     routes=routes,
#     io_coords=None,
#     include_io=False,
#     iteration=num_steps,
#     title_suffix="Final Placement"
# )

# print("\nINFO: FPGA placement complete!")
