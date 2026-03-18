"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
"""

import sys
sys.path.insert(0, '.')
from fem_placer import (
    FpgaPlacer
)
from fem_placer.logger import *
from fem_placer.config import *
from ml.dataset import *

SET_LEVEL('WARNING')

instances = ['c2670', 'c5315', 'c6288', 'c7552',
             's1488', 's5378', 's9234', 's15850', 'FPGA-example1']
            
num_trials = 5
num_steps = 200
dev = 'cpu'
manual_grad = False
anneal='lin'

for instance in instances:
    place_type = PlaceType.CENTERED
    debug = False
    fpga_placer = FpgaPlacer(place_orientation = place_type, 
                            grid_type = GridType.SQUARE,
                            place_mode = IoMode.NORMAL,
                            utilization_factor = 0.4,
                            debug = debug,
                            device = dev)
    
    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    fpga_placer.set_alpha(30)
    fpga_placer.set_beta(30)
    fpga_placer.save_init_params(instance_name=instance)

