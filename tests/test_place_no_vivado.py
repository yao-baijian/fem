"""
Test script for FPGA placement optimizer using QUBO formulation.

This script tests the FPGAPlacementOptimizer on a real FPGA design.
"""

import sys
sys.path.insert(0, '.')
import time
import torch
from fem_placer import (
    FPGAPlacementOptimizer
)
from fem_placer.logger import *
from fem_placer.config import *
from ml.dataset import *

SET_LEVEL('INFO')

instances = ['c2670', 'c2670', 'c5315', 'c6288', 'c7552',
             's1488', 's5378', 's9234', 's15850', 'bgm', 'sha1', 'RLE_BlobMerging', 'FPGA-example1']

# instances = ['c2670_boundary', 'c2670_boundary', 'c5315_boundary', 'c6288_boundary', 'c7552_boundary',
#              's1488_boundary', 's5378_boundary', 's9234_boundary', 's15850_boundary', 'bgm_boundary', 'sha1_boundary', 'RLE_BlobMerging_boundary', 'FPGA-example1_boundary']

# 'FPGA-example1'

# instances = ['c7552']
            
num_trials = 5
num_steps = 200
dev = 'cuda'
manual_grad = False
anneal='lin'

print(f"{'Instance':<15} {'Mem(MB)':<10} {'Time(s)':<10}")

for instance in instances:
    place_type = PlaceType.IO

    optimizer = FPGAPlacementOptimizer.from_saved_params(
        f'result/{instance}/init_params.json',
        num_trials=num_trials,
        num_steps=num_steps,
        dev=dev
    )

    N = optimizer.num_inst
    M = optimizer.num_site
    
    # FPGAPlacementOptimizer single-trial memory footprint dominated by:
    # `p`: [1, N, M]
    # `D`: [M, M]
    # `J`: [N, N]
    # `PD`: [1, N, M] 
    # `E_matrix`: [1, N, N]
    num_elements = (2 * 1 * N * M) + (M * M) + (N * N) + (1 * N * N)
    memory_mb = (num_elements * 4) / (1024 ** 2)

    start_time = time.time()
    config, result = optimizer.optimize()
    end_time = time.time()
    optimize_time = end_time - start_time

    print(f"{instance:<15} {memory_mb:<10.2f} {optimize_time:<10.2f}")

    optimal_inds = torch.argwhere(result==result.min()).reshape(-1)

