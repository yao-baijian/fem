import sys
sys.path.append('.')
import subprocess
from FEM import FEM
import torch
import multiprocessing as mp

from utils import *

def run_single_trial(trial_id, case_type, instance, num_steps, dev, q):
    torch.manual_seed(trial_id)
    case_bmincut = FEM.from_file(case_type, instance, index_start=1)
    case_bmincut.set_up_solver(1, num_steps, dev=dev, q=q)  # num_trials=1
    config, result = case_bmincut.solve()
    return config, result.item()

def run_parallel_trials(case_type, instance, processe_num=4):

    num_trials = 200
    num_steps = 1000
    dev = 'cuda'
    q = 2
    
    with mp.Pool(processes = processe_num) as pool:
        results = pool.starmap(
            run_single_trial,
            [(i, case_type, instance, num_steps, dev, q) for i in range(num_trials)]
        )
    
    all_configs = []
    all_results = []
    for config, result in results:
        all_configs.append(config)
        all_results.append(result)
    
    all_results = torch.tensor(all_results)
    optimal_ind = torch.argmin(all_results)
    best_config = all_configs[optimal_ind]
    best_result = all_results[optimal_ind]
    
    print(f'{instance}, optimal value {best_result}')
    return best_config, best_result

case_type = 'bmincut'
instance = '../partition/data/hypergraph_set/bibd_49_3.mtx.hgr'

best_config, best_result = run_parallel_trials(case_type, instance, processe_num=4)

hyperedges = parse_hypergraph_edges(instance)
group_assignment = best_config.argmax(dim=1).cpu().numpy()
true_cut_value = evaluate_cut_value(group_assignment, hyperedges)
print(f'True cut value: {true_cut_value}')