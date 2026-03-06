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

instances = ['FPGA-example1']

# 'FPGA-example1'
            
num_trials = 5
num_steps = 200
dev = 'cpu'
manual_grad = False
anneal='lin'

for instance in instances:
    place_type = PlaceType.IO
    debug = False
    fpga_placer = FpgaPlacer(place_type, 
                            GridType.SQUARE,
                            0.4,
                            debug,
                            device=dev)
    
    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    fpga_placer.set_alpha(30)
    fpga_placer.set_beta(30)
    fpga_placer.save_init_params(instance_name=instance)

