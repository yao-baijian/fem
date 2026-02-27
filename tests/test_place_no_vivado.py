"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
"""

import sys
sys.path.insert(0, '.')

import torch
from fem_placer import (
    FPGAPlacementOptimizer
)
from fem_placer.logger import *
from fem_placer.config import *
from ml.dataset import *

SET_LEVEL('INFO')

# instances = ['c1355', 'c2670', 'c5315', 'c6288', 'c7552',
#              's713', 's1238', 's1488', 's5378', 's9234', 's15850']

# 'FPGA-example1'

instances = ['c7552']
            
num_trials = 5
num_steps = 200
dev = 'cpu'
manual_grad = False
anneal='lin'

for instance in instances:
    place_type = PlaceType.IO

    optimizer = FPGAPlacementOptimizer.from_saved_params(
        f'result/{instance}/init_params.json',
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev
    )

    config, result = optimizer.optimize()
    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)

