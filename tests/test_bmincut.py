import sys
sys.path.append('.')
import subprocess
from FEM import FEM
import torch
import numpy as np

def parse_hypergraph_edges(instance_path: str) -> list:
    hyperedges = []
    try:
        with open(instance_path, 'r') as f:
            f.readline()
            for line in f:
                if line.strip():
                    vertices = [int(v) - 1 for v in line.split() if v.strip()]
                    if len(vertices) > 1:  
                        hyperedges.append(vertices)
        print(f"Parsed {len(hyperedges)} hyperedges from {instance_path}")
        return hyperedges
    except Exception as e:
        print(f"Error parsing hypergraph: {e}")
        return []

def evaluate_cut_value(assignment: np.ndarray, hyperedges: list) -> int:
    cut_count = 0
    for hyperedge in hyperedges:
        groups_in_hyperedge = set()
        for vertex in hyperedge:
            if vertex < len(assignment):
                groups_in_hyperedge.add(assignment[vertex])
        

        if len(groups_in_hyperedge) > 1:
            cut_count += 1
    
    return cut_count

# num_trials = 500
# num_steps = 1000

num_trials = 1
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
case_bmincut = FEM.from_file(case_type, instance, index_start=1)
case_bmincut.set_up_solver(num_trials, num_steps, dev=dev, q=2)
config, result = case_bmincut.solve()
optimal_inds = torch.argwhere(result==result.min()).reshape(-1)
best_config = config[optimal_inds[0]]
print(f'{instance}, optimal value {result.min()}')

hyperedges = parse_hypergraph_edges(instance)
group_assignment = best_config.argmax(dim=1).cpu().numpy()
true_cut_value = evaluate_cut_value(group_assignment, hyperedges)
print(f'True cut value: {true_cut_value}')

# kahypar_cmd = [
#     '../kahypar/build/kahypar/application/KaHyPar',
#     '-h', instance,
#     '-k', '3',
#     '-e', '0.03',
#     '-o', 'km1',
#     '-m', 'direct',
#     '-p', 'config/km1_kKaHyPar_sea20.ini'
# ]

# result = subprocess.run(kahypar_cmd, capture_output=True, text=True)
# print("KaHyPar output:")
# print(result.stdout)
# if result.stderr:
#     print("KaHyPar errors:")
#     print(result.stderr)