import torch
import sys
sys.path.append('.')
from FEM import FEM
from FEM.placement.placer import FpgaPlacer
from FEM.placement.logger import *
from FEM.placement.config import *
from FEM.ml_alpha.dataset import *
from FEM.ml_alpha.predict import predict_alpha

num_trials = 5
num_steps = 200
dev = 'cuda'
manual_grad = False
anneal='inverse'
case_type = 'fpga_placement'

instances = ['c880', 'c1355', 'c2670', 'c5315', 'c6288', 'c7552']
            #  's713', 's1238', 's1488', 's5378', 's9234', 's15850', 'FPGA-example1']
SET_LEVEL('WARNING')

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
    
    case_placements = FEM.from_file(case_type, instance, fpga_placer, index_start=1)

    row = extract_features_from_placer(fpga_placer, hpwl_before=0, alpha=0)
    alpha = predict_alpha(row)
    print(f'instance {instance}, predicted alpha {alpha}')
    fpga_placer.set_alpha(alpha)