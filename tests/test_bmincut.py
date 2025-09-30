import sys
sys.path.append('.')
import subprocess
from FEM import FEM
import torch
import time

from utils import *

# num_trials = 500
# num_steps = 1000

num_trials = 20
num_steps = 800
dev = 'cuda' # if you do not have gpu in your computing devices, then choose 'cpu' here

# normal graph
# case_type = 'bmincut'
# instance = 'tests/test_instances/karate.txt'
# case_bmincut = FEM.from_file(case_type, instance, index_start=1)
# case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=3)
# config, result = case_bmincut.solve()
# optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
# print(f'{instance}, optimal value {result.min()}')

# instance = '../partition/data/ash219/ash219.mtx'
# case_bmincut = FEM.from_file(case_type, instance, index_start=1)
# case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=3)
# config, result = case_bmincut.solve()
# optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
# print(f'{instance}, optimal value {result.min()}')

# hyper graph
case_type = 'bmincut'
instance = '../partition/data/hypergraph_set/bibd_49_3.mtx.hgr'


# instance = '../partition/data/hypergraph_set/Pd_rhs.mtx.hgr'
start_time = time.time()
case_bmincut = FEM.from_file(case_type, instance, index_start=1)
# print(f"FEM.from_file took: {time.time() - start_time:.4f} seconds")

# start_time = time.time()
case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=2, manual_grad= True)
# print(f"set_up_solver took: {time.time() - start_time:.4f} seconds")

# start_time = time.time()
config, result = case_bmincut.solve()
print(f"solve took: {time.time() - start_time:.4f} seconds")

optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
best_config = config[optimal_inds[0]]
print(f'{instance}, optimal value {result.min()}')