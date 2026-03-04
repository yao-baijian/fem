"""
Test script for FPGA placement optimizer using QUBO formulation.

Compares IO constraint modes:
  - exactly_one: (u_s - 1)^2 QUBO (default)
  - at_most_one_sq: squared at-most-one PUBO (same as logic)
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

instances = ['c7552']

num_trials = 10
num_steps = 3000
dev = 'cpu'

for instance in instances:
    place_type = PlaceType.IO

    for io_mode in ['exactly_one', 'at_most_one_sq']:
        print(f"\n{'='*60}")
        print(f"Instance: {instance}, IO mode: {io_mode}")
        print(f"{'='*60}")

        optimizer = FPGAPlacementOptimizer.from_saved_params(
            f'result/{instance}/init_params.json',
            num_trials=num_trials,
            num_steps=num_steps,
            dev=dev,
            io_mode=io_mode,
        )
        constraint_alpha = optimizer.num_inst * 7.0
        optimizer.constraint_alpha = constraint_alpha

        config, result = optimizer.optimize()
        optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
        print(f"\nResult (HPWL): {result}")
        print(f"Best: {result.min().item():.2f}, Mean: {result.mean().item():.2f}")
