from placement_simplified import FpgaPlacer
from FEM import FEM
import torch
from drawer import PlacementDrawer
from legalizer import Legalizer

# with cell and net: ./vivado/output_dir/post_impl.dcp
# only logic netlist: ./vivado/output_dir/post_synth.dcp
num_trials = 10
num_steps = 1000
dev = 'cpu'

case_type = 'fpga_placement'
instance = ''

fpga_placement_wrapper = FpgaPlacer()
fpga_placement_wrapper.init_placement('./vivado/output_dir/post_impl.dcp', 'optimized_placement.dcp')

case_placements = FEM.from_file(case_type, instance, fpga_placement_wrapper, index_start=1)

area_size = fpga_placement_wrapper.bbox['area_size']

case_placements.set_up_solver(num_trials, num_steps, dev=dev, q=area_size, manual_grad= False)
config, result = case_placements.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
best_config = config[optimal_inds[0]]

legalizer = Legalizer(fpga_placement_wrapper.bbox)
placement_legalized = legalizer.legalize_placement(best_config, max_attempts=100)

drawer = PlacementDrawer(bbox = fpga_placement_wrapper.bbox)
drawer.draw_hard_placement(placement_legalized, 1000, title_suffix="Final Placement")

# print(best_config)
# print(result.min())