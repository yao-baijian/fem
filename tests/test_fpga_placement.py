"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
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
from ml.dataset import *
from ml.predict import predict_alpha

SET_LEVEL('INFO')

# instances = ['c1355', 'c2670', 'c5315', 'c6288', 'c7552',
#              's713', 's1238', 's1488', 's5378', 's9234', 's15850']

# 'FPGA-example1'

instances = ['c7552']
            
draw_evolution = False
draw_loss_function = False
draw_final_placement = False
num_trials = 5
num_steps = 400
dev = 'cpu'
manual_grad = False
anneal='lin'
case_type = 'fpga_placement'

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
    global_drawer = PlacementDrawer(placer=fpga_placer)
    
    row = extract_features_from_placer(fpga_placer,
                                       alpha=0, 
                                       beta=0, 
                                       with_io=False)
    
    alpha = predict_alpha(row)
    INFO(f'instance {instance}, predicted alpha {alpha}')
    fpga_placer.set_alpha(alpha)

    if place_type == PlaceType.IO:
        fpga_placer.set_beta(10)

    # fpga_placer.set_alpha(30)
    
    optimizer = FPGAPlacementOptimizer(
        num_inst=fpga_placer.opti_insts_num,
        num_fixed_inst=fpga_placer.fixed_insts_num,
        num_site=fpga_placer.get_grid('logic').area,
        num_fixed_site=fpga_placer.get_grid('io').area_width,
        logic_grid_width=fpga_placer.get_grid('logic').area_width,
        coupling_matrix=fpga_placer.net_manager.insts_matrix,
        site_coords_matrix=fpga_placer.logic_site_coords,
        io_site_connect_matrix=fpga_placer.net_manager.io_insts_matrix,
        io_site_coords=fpga_placer.io_site_coords,
        constraint_alpha=fpga_placer.constraint_alpha,
        constraint_beta=fpga_placer.constraint_beta,  # For IO placements, beta is set separately
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
        global_drawer.plot_fpga_placement_loss('result/hpwl_loss.png')

    if draw_evolution:
        global_drawer.draw_multi_step_placement('result/placement_evolution.png')

    if draw_final_placement:
        global_drawer.draw_place_and_route(placement_legalized[0], routes, None, False, 1000, title_suffix="Final Placement with Routing")

