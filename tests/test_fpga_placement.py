import torch
import sys
sys.path.append('.')
from FEM import FEM
from FEM.placement.placement import FpgaPlacer
from FEM.placement.drawer import PlacementDrawer
from FEM.placement.legalizer import Legalizer
from FEM.placement.router import Router

# with cell and net: ./vivado/output_dir/post_impl.dcp
# only logic netlist: ./vivado/output_dir/post_synth.dcp


num_trials = 20
num_steps = 2000
dev = 'cpu'

case_type = 'fpga_placement'
instance = 's9234'

fpga_placement_wrapper = FpgaPlacer()
fpga_placement_wrapper.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', 'optimized_placement.dcp')
# fpga_placement_wrapper.init_placement(f'../../Benchmarks/ispd/FPGA01/design.dcp', 'optimized_placement.dcp')

case_placements = FEM.from_file(case_type, instance, fpga_placement_wrapper, index_start=1)
site_to_site_connect_matrix = case_placements.problem.coupling_matrix

area_size = fpga_placement_wrapper.bbox['area_size']
global_drawer = PlacementDrawer(bbox = fpga_placement_wrapper.bbox)

case_placements.set_up_solver(num_trials, num_steps, betamin=0.001, betamax=0.5, dev=dev, q=area_size, manual_grad= False, drawer=global_drawer)
config, result = case_placements.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
print(f"INFO: optimal inds: {optimal_inds.tolist()} with min hpwl: {result.min():.2f}")
best_config = config[optimal_inds[0]]

fpga_placement_wrapper.debug = False

legalizer = Legalizer(fpga_placement_wrapper.bbox, placer=fpga_placement_wrapper)
placement_legalized = legalizer.legalize_placement(best_config, max_attempts=100)

# legalized_coords = placement_legalized['legalized_coords']
hpwl_after_legalization = fpga_placement_wrapper.estimate_solver_hpwl(placement_legalized, io_coords=None, include_io=False)

print(f"INFO: total hpwl after fem solver: {hpwl_after_legalization:.2f}")

router = Router(fpga_placement_wrapper.bbox)
routes = router.route_connections(site_to_site_connect_matrix, placement_legalized.unsqueeze(0))[0]

# drawer.draw_hard_placement(placement_legalized, 1000, title_suffix="Final Placement")

global_drawer.draw_complete_placement(placement_legalized, routes, 1000, title_suffix="Final Placement with Routing")

# print(best_config)
# print(result.min())