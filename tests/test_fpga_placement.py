import torch
import sys
sys.path.append('.')
from FEM import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.drawer import PlacementDrawer
from FEM.placement.legalizer import Legalizer
from FEM.placement.router import Router
from FEM.placement.logger import *

num_trials = 20
num_steps = 2000
dev = 'cuda'

case_type = 'fpga_placement'
# instances = ['c880', 'c1355', 'c2670', 'c5315', 'c6288', 'c7552'
#              's713', 's1238', 's1488', 's5378', 's9234', 's15850',
#              'FPGA-example1', 'FPGA-example2', 'FPGA-example3', 'FPGA-example4']

instances = ['FPGA-example1']
SET_LEVEL('INFO')

print("\n" + "="*105)
print(f"{'Data Set':<12} {'Instance':<10} {'Site':<6} {'Net|Total Net':<14} {'Overlap':<8} {'FEM HPWL Initial':<18} {'FEM HPWL Final':<16} {'VIVADO HPWL':<12}")
print("-" * 105)

for instance in instances:    
    # try:
    fpga_placement_wrapper = FpgaPlacer()
    vivado_hpwl = fpga_placement_wrapper.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', 'optimized_placement.dcp')
    
    site_num = len(fpga_placement_wrapper.optimizable_insts)
    site_net = len(fpga_placement_wrapper.net_manager.site_to_site_connectivity)
    total_net = len(fpga_placement_wrapper.net_manager.nets)
    
    case_placements = FEM.from_file(case_type, instance, fpga_placement_wrapper, index_start=1)
    site_to_site_connect_matrix = case_placements.problem.coupling_matrix

    area_size = fpga_placement_wrapper.bbox['area_size']
    global_drawer = PlacementDrawer(bbox = fpga_placement_wrapper.bbox)

    case_placements.set_up_solver(num_trials, num_steps, betamin=0.001, betamax=0.5, dev=dev, q=area_size, manual_grad= False, drawer=global_drawer)
    config, result = case_placements.solve()
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
    # print(f"INFO <Top> : optimal inds: {optimal_inds.tolist()} with min hpwl: {result.min():.2f}")
    best_config = config[optimal_inds[0]]

    fpga_placement_wrapper.debug = False

    legalizer = Legalizer(fpga_placement_wrapper.bbox, placer=fpga_placement_wrapper)
    placement_legalized, overlap, stage1_result, stage2_result = legalizer.legalize_placement(best_config, max_attempts=100)
    
    fem_hpwl_initial = stage1_result['hpwl_before']
    fem_hpwl_final = stage2_result['hpwl_optimized']
    
    router = Router(fpga_placement_wrapper.bbox)
    routes = router.route_connections(site_to_site_connect_matrix, placement_legalized.unsqueeze(0))[0]

    print(f"{'Benchmarks':<12} {instance:<10} {site_num:<6} {f'{site_net}/{total_net}':<14} {overlap:<8} "
              f"{fem_hpwl_initial:<18.2f} {fem_hpwl_final:<16.2f} {vivado_hpwl:<12.2f}")
    
    # drawer.draw_hard_placement(placement_legalized, 1000, title_suffix="Final Placement")

    # global_drawer.draw_complete_placement(placement_legalized, routes, 1000, title_suffix="Final Placement with Routing")
    
    
    # except Exception as e:
    #     print(f'ERROR')

# print(best_config)
# print(result.min())