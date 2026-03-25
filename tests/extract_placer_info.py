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

# instances = ['c2670', 'c5315', 'c6288', 'c7552',
#              's1488', 's5378', 's9234', 's15850', 'FPGA-example1']

# instances = ['bgm', 'sha1', 'RLE_BlobMerging']

instances = ['c2670_boundary', 'c5315_boundary', 'c6288_boundary', 'c7552_boundary',
             's1488_boundary', 's5378_boundary', 's9234_boundary', 's15850_boundary', 'bgm_boundary', 'sha1_boundary', 'RLE_BlobMerging_boundary', 'FPGA-example1_boundary']

num_trials = 5
num_steps = 200
dev = 'cpu'
manual_grad = False
anneal='lin'

connectivity_results = []

for instance in instances:
    place_type = PlaceType.IO
    debug = False
    fpga_placer = FpgaPlacer(place_orientation = place_type, 
                            grid_type = GridType.SQUARE,
                            place_mode = IoMode.VIRTUAL_NODE,
                            utilization_factor = 0.4,
                            debug = debug,
                            device = dev)
    
    fpga_placer.set_instance_name(instance)
    
    vivado_hpwl, inst_num, net_num = fpga_placer.init_placement(f'./vivado/output_dir/{instance}/post_impl.dcp', f'./vivado/output_dir/{instance}/optimized_placement.pl')
    
    # Calculate connectivity
    net_sizes = [len(sites) for sites in fpga_placer.net_manager.net_to_sites.values()]
    if net_sizes:
        min_conn = min(net_sizes)
        max_conn = max(net_sizes)
        avg_conn = sum(net_sizes) / len(net_sizes)
    else:
        min_conn = max_conn = avg_conn = 0
        
    connectivity_results.append({
        'Instance': instance,
        'Min': min_conn,
        'Max': max_conn,
        'Avg': avg_conn
    })
    
    fpga_placer.set_alpha(30)
    fpga_placer.set_beta(30)
    fpga_placer.save_init_params(instance_name=instance)

print("\n" + "="*65)
print(f"{'Instance':<25} | {'Min Conn':<10} | {'Max Conn':<10} | {'Avg Conn':<10}")
print("-" * 65)
for res in connectivity_results:
    print(f"{res['Instance']:<25} | {res['Min']:<10} | {res['Max']:<10} | {res['Avg']:<10.2f}")
print("="*65 + "\n")

