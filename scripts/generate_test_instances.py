#!/usr/bin/env python3
"""
Generate test instances from master branch code for verification.

This script runs master branch functions and saves their outputs as test fixtures.
These fixtures will be used to verify that the refactored code produces identical results.
"""

import os
import sys
import torch
import numpy as np

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the specific modules directly to avoid rapidwright dependency
import importlib.util

def load_module_directly(module_path, module_name):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fp = load_module_directly(
    os.path.join(project_root, 'FEM', 'customized_problem', 'fpga_placement.py'),
    'fpga_placement'
)
hb = load_module_directly(
    os.path.join(project_root, 'FEM', 'customized_problem', 'hyper_bmincut.py'),
    'hyper_bmincut'
)

def generate_objectives_test_data():
    """Generate test data for fpga_placement objective functions."""
    print("Generating objectives test data...")

    torch.manual_seed(42)
    np.random.seed(42)

    # Test parameters
    batch_size = 5
    num_instances = 20
    num_sites = 25  # 5x5 grid
    area_width = 5

    # Create coupling matrix J (symmetric, sparse connectivity)
    J = torch.zeros(num_instances, num_instances)
    # Add random connections
    for i in range(num_instances):
        for j in range(i + 1, num_instances):
            if torch.rand(1).item() < 0.3:  # 30% connection probability
                weight = torch.rand(1).item()
                J[i, j] = weight
                J[j, i] = weight

    # Create probability distributions p
    h = torch.randn(batch_size, num_instances, num_sites)
    p = torch.softmax(h, dim=2)

    # Create site coordinates matrix
    site_coords_matrix = torch.zeros(num_sites, 2)
    for idx in range(num_sites):
        site_coords_matrix[idx, 0] = idx % area_width  # x
        site_coords_matrix[idx, 1] = idx // area_width  # y

    # Generate expected outputs for each function
    results = {
        'batch_size': batch_size,
        'num_instances': num_instances,
        'num_sites': num_sites,
        'area_width': area_width,
        'J': J,
        'h': h,
        'p': p,
        'site_coords_matrix': site_coords_matrix,
    }

    # Test coordinate functions
    inst_indices = torch.argmax(p, dim=2)
    results['inst_indices'] = inst_indices
    results['inst_coords_from_index'] = fp.get_inst_coords_from_index(inst_indices, area_width)
    results['site_distance_matrix'] = fp.get_site_distance_matrix(site_coords_matrix)
    results['expected_placements'] = fp.get_expected_placements_from_index(p, site_coords_matrix)
    results['hard_placements'] = fp.get_hard_placements_from_index(p, site_coords_matrix)
    results['st_placements'] = fp.get_placements_from_index_st(p, site_coords_matrix)

    # Test HPWL functions
    results['hpwl_qubo'] = fp.get_hpwl_loss_qubo(J, p, site_coords_matrix)

    # Test with IO
    num_io = 5
    J_LL = J.clone()
    J_LI = torch.rand(num_instances, num_io) * 0.5

    h_logic = h.clone()
    p_logic = p.clone()

    h_io = torch.randn(batch_size, num_io, num_io)
    p_io = torch.softmax(h_io, dim=2)

    io_site_coords = torch.zeros(num_io, 2)
    io_site_coords[:, 0] = -1  # x = -1 for IO
    io_site_coords[:, 1] = torch.arange(num_io).float()

    results['num_io'] = num_io
    results['J_LL'] = J_LL
    results['J_LI'] = J_LI
    results['h_logic'] = h_logic
    results['p_logic'] = p_logic
    results['h_io'] = h_io
    results['p_io'] = p_io
    results['io_site_coords'] = io_site_coords

    results['hpwl_with_io'] = fp.get_hpwl_loss_qubo_with_io(
        J_LL, J_LI, p_logic, p_io, site_coords_matrix, io_site_coords
    )

    # Test constraint functions
    results['constraints'] = fp.get_constraints_loss(p)
    results['constraints_with_io'] = fp.get_constraints_loss_with_io(p_logic, p_io)

    # Test expected placement (combined loss)
    alpha = 10.0
    step = 0
    results['alpha'] = alpha
    results['expected_placement'] = fp.expected_fpga_placement(
        J, p, site_coords_matrix, step, area_width, alpha
    )

    results['expected_placement_with_io'] = fp.expected_fpga_placement_with_io(
        J_LL, J_LI, p_logic, p_io, site_coords_matrix, io_site_coords
    )

    # Test inference functions
    inferred_coords, inferred_hpwl = fp.infer_placements(J, p, area_width, site_coords_matrix)
    results['inferred_coords'] = inferred_coords
    results['inferred_hpwl'] = inferred_hpwl

    inferred_coords_io, inferred_hpwl_io = fp.infer_placements_with_io(
        J_LL, J_LI, p_logic, p_io, area_width, site_coords_matrix, io_site_coords
    )
    results['inferred_coords_logic'] = inferred_coords_io[0]
    results['inferred_coords_io'] = inferred_coords_io[1]
    results['inferred_hpwl_with_io'] = inferred_hpwl_io

    return results


def generate_hyper_bmincut_test_data():
    """Generate test data for hyper_bmincut functions."""
    print("Generating hyper_bmincut test data...")

    torch.manual_seed(123)
    np.random.seed(123)

    # Test parameters
    batch_size = 3
    num_nodes = 16
    num_clusters = 4

    # Create hyperedges (list of node indices for each hyperedge)
    hyperedges = [
        [0, 1, 2],
        [2, 3, 4, 5],
        [4, 5, 6],
        [6, 7, 8, 9],
        [8, 9, 10, 11],
        [10, 11, 12, 13],
        [12, 13, 14, 15],
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ]

    # Create coupling matrix J (node-hyperedge incidence)
    num_hyperedges = len(hyperedges)
    J = torch.zeros(num_hyperedges, num_nodes)
    for he_idx, he in enumerate(hyperedges):
        for node in he:
            J[he_idx, node] = 1.0

    # Create probability distributions
    h = torch.randn(batch_size, num_nodes, num_clusters)
    p = torch.softmax(h, dim=2)

    # Balance constraint parameters
    target_size = num_nodes / num_clusters
    epsilon = 0.2  # 20% imbalance tolerance
    U_max = target_size * (1 + epsilon)
    L_min = target_size * (1 - epsilon)

    results = {
        'batch_size': batch_size,
        'num_nodes': num_nodes,
        'num_clusters': num_clusters,
        'num_hyperedges': num_hyperedges,
        'hyperedges': hyperedges,
        'J': J,
        'h': h,
        'p': p,
        'U_max': U_max,
        'L_min': L_min,
    }

    # Test balance constraint
    results['balance_constrain'] = hb.balance_constrain(J, p, U_max, L_min)

    # Test expected hyperbmincut
    results['expected_hyperbmincut'] = hb.expected_hyperbmincut(J, p, hyperedges)

    # Test infer hyperbmincut
    # Note: infer_hyperbmincut expects J shape to match p structure
    # Create a node-node coupling matrix for inference
    J_nodes = torch.zeros(num_nodes, num_nodes)
    for he in hyperedges:
        for i in range(len(he)):
            for j in range(i + 1, len(he)):
                J_nodes[he[i], he[j]] = 1.0
                J_nodes[he[j], he[i]] = 1.0

    results['J_nodes'] = J_nodes

    # infer_hyperbmincut uses J_nodes shape
    try:
        config, cut_value = hb.infer_hyperbmincut(J_nodes, p, hyperedges)
        results['infer_config'] = config
        results['infer_cut_value'] = cut_value
    except Exception as e:
        print(f"Warning: infer_hyperbmincut failed: {e}")
        results['infer_config'] = None
        results['infer_cut_value'] = None

    # Test variant functions
    results['expected_hyperbmincut_temped'] = hb.expected_hyperbmincut_expected_nodes_temped(J, p, hyperedges)
    results['expected_hyperbmincut_simplified'] = hb.expected_hyperbmincut_expected_crossing_simplified(J, p, hyperedges)

    return results


def main():
    print("=" * 60)
    print("Generating test fixtures from master branch")
    print("=" * 60)

    fixtures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'tests', 'fixtures')
    os.makedirs(fixtures_dir, exist_ok=True)

    # Generate objectives test data
    objectives_data = generate_objectives_test_data()
    objectives_path = os.path.join(fixtures_dir, 'objectives_data.pt')
    torch.save(objectives_data, objectives_path)
    print(f"Saved objectives test data to {objectives_path}")

    # Generate hyper_bmincut test data
    hyper_bmincut_data = generate_hyper_bmincut_test_data()
    hyper_bmincut_path = os.path.join(fixtures_dir, 'hyper_bmincut_data.pt')
    torch.save(hyper_bmincut_data, hyper_bmincut_path)
    print(f"Saved hyper_bmincut test data to {hyper_bmincut_path}")

    print("\n" + "=" * 60)
    print("Test fixture generation complete!")
    print("=" * 60)

    # Print summary
    print("\nObjectives data contents:")
    for key, value in objectives_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor{tuple(value.shape)}")
        else:
            print(f"  {key}: {type(value).__name__}")

    print("\nHyper_bmincut data contents:")
    for key, value in hyper_bmincut_data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor{tuple(value.shape)}")
        elif isinstance(value, list):
            print(f"  {key}: list[{len(value)}]")
        else:
            print(f"  {key}: {type(value).__name__}")


if __name__ == '__main__':
    main()
