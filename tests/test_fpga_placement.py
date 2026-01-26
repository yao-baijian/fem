import torch
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

num_trials = 5
num_steps = 200
dev = 'cuda'
manual_grad = False
anneal='inverse'

case_type = 'fpga_placement'
# instances = ['c880', 'c1355', 'c2670', 'c5315', 'c6288', 'c7552'
#              's713', 's1238', 's1488', 's5378', 's9234', 's15850',
#              'FPGA-example1', 'FPGA-example2', 'FPGA-example3', 'FPGA-example4']
draw_evolution = False
draw_loss_function = False
draw_final_placement = False

instances = ['c880', 'c1355', 'c2670', 'c5315', 'c6288', 'c7552']
            #  's713', 's1238', 's1488', 's5378', 's9234', 's15850', 'FPGA-example1']
SET_LEVEL('WARNING')

print("\n" + "="*105)
print(f"{'Data Set':<12} {'Instance':<10} {'Site':<6} {'Net|Total Net':<14} {'Overlap':<8} {'FEM HPWL Initial':<18} {'FEM HPWL Final':<16} {'VIVADO HPWL':<12}")
print("-" * 105)

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
        
    range_end = 100
    for used_alpha in range(1, range_end):
        fpga_placer.set_alpha(used_alpha)
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
        
        obj = fem_hpwl_final['hpwl_no_io'] + (overlap * 10)
        
        row = extract_features_from_placer(fpga_placer, hpwl_before=fem_hpwl_initial['hpwl_no_io'], hpwl_after=fem_hpwl_final['hpwl_no_io'], overlap_after=overlap, alpha=used_alpha)
        append_row(row)
        
        if draw_loss_function:
            global_drawer.plot_fpga_placement_loss('hpwl_loss.png')

        if draw_evolution:
            global_drawer.draw_multi_step_placement('placement_evolution.png')

        if draw_final_placement:
            global_drawer.draw_place_and_route(placement_legalized[0], routes, None, False, 1000, title_suffix="Final Placement with Routing")
        
        
        
