# User Guide

Comprehensive guide for using the FPGA Placement FEM package.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Workflow](#basic-workflow)
4. [Advanced Usage](#advanced-usage)
5. [Examples](#examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch (for tensor operations)
- RapidWright (for FPGA design handling)

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yao-baijian/fem.git
cd fem

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install PyTorch
uv pip install torch

# Install RapidWright
uv pip install rapidwright

# Optional: Install ML dependencies
uv pip install scikit-learn joblib
```

### Option 2: Using pip

```bash
git clone https://github.com/yao-baijian/fem.git
cd fem

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or .venv\Scripts\activate  # On Windows

# Install package
pip install -e .

# Install PyTorch
pip install torch

# Install RapidWright
pip install rapidwright

# Optional: ML dependencies
pip install -e ".[ml]"
```

### Verify Installation

```python
import torch
import fem_placer

print(f"PyTorch version: {torch.__version__}")
print(f"fem_placer available classes: {dir(fem_placer)}")
```

---

## Quick Start

Here's a minimal example to get you started:

```python
import torch
from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
    Legalizer,
    Router,
    PlacementDrawer
)
from fem_placer.utils import parse_fpga_design

# 1. Load FPGA design
placer = FpgaPlacer(utilization_factor=0.4)
placer.init_placement('design.dcp', 'output.dcp')

# 2. Parse design to get connectivity
num_inst, num_site, J, J_extend = parse_fpga_design(placer)

# 3. Create site coordinates matrix
area_length = placer.bbox['area_length']
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

# 4. Create optimizer
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=site_coords,
    constraint_weight=1.0
)

# 5. Run optimization
config, result = optimizer.optimize(
    num_trials=10,
    num_steps=1000,
    dev='cpu',
    area_width=area_length,
    betamin=0.01,
    betamax=0.5
)

# 6. Get best solution
best_idx = torch.argmin(result)
grid_coords = config[best_idx]
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(grid_coords)

# 7. Legalize
legalizer = Legalizer(placer=placer, device='cpu')
logic_ids = torch.arange(num_inst)
placement, overlaps, hpwl_before, hpwl_after = legalizer.legalize_placement(
    real_coords, logic_ids
)

print(f"Final HPWL: {hpwl_after['hpwl_no_io']:.2f}")
print(f"Overlaps: {overlaps}")
```

---

## Basic Workflow

### Step 1: Initialize the Placer

The `FpgaPlacer` is your main interface to RapidWright:

```python
from fem_placer import FpgaPlacer

# Create placer with desired utilization
placer = FpgaPlacer(utilization_factor=0.4)

# Load design
placer.init_placement(
    dcp_file='path/to/design.dcp',
    dcp_output='path/to/output.dcp'
)
```

**Parameters:**
- `utilization_factor`: Controls how densely instances are packed (0.0-1.0)
  - Lower values (0.3-0.5): More space, easier routing, better timing
  - Higher values (0.7-0.95): Denser placement, harder routing

### Step 2: Parse the Design

Extract connectivity information:

```python
from fem_placer.utils import parse_fpga_design

num_inst, num_site, J, J_extend = parse_fpga_design(placer)

print(f"Number of instances: {num_inst}")
print(f"Number of sites: {num_site}")
print(f"Connectivity matrix shape: {J.shape}")
```

**Returns:**
- `num_inst`: Number of logic instances to place
- `num_site`: Number of available placement sites
- `J`: Logic-to-logic connectivity matrix [num_inst, num_inst]
- `J_extend`: Extended connectivity including IO [num_inst+num_io, num_inst+num_io]

### Step 3: Create Site Coordinates

Build a matrix of all possible placement site coordinates:

```python
import torch

# Get grid dimensions
area_length = placer.bbox['area_length']

# Create all possible site coordinates
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

print(f"Site coordinates shape: {site_coords.shape}")  # [num_site, 2]
```

### Step 4: Configure the Optimizer

Create the FEM optimizer with QUBO formulation:

```python
from fem_placer import FPGAPlacementOptimizer

optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=site_coords,
    constraint_weight=num_inst / 2.0  # Balance between HPWL and constraints
)
```

**Key Parameters:**
- `constraint_weight`: Controls trade-off between HPWL minimization and legality
  - Higher values: More legal placements, possibly higher HPWL
  - Lower values: Lower HPWL, possibly more overlaps

### Step 5: Run Optimization

Execute the FEM algorithm:

```python
configs, energies = optimizer.optimize(
    num_trials=10,        # Number of independent runs
    num_steps=1000,       # Optimization steps per trial
    dev='cpu',            # 'cpu' or 'cuda'
    area_width=area_length,
    betamin=0.01,         # Initial temperature (high = more exploration)
    betamax=0.5,          # Final temperature (low = more exploitation)
    anneal='inverse',     # Annealing schedule
    optimizer='adam',     # Optimizer type
    learning_rate=0.1     # Learning rate
)
```

**Optimization Parameters:**

- `num_trials`: Run multiple independent trials, keep the best
  - Small designs: 5-10 trials
  - Large designs: 10-20 trials

- `num_steps`: Iterations per trial
  - Small designs (<100 instances): 500-1000
  - Medium (100-500): 1000-2000
  - Large (>500): 2000-5000

- `betamin/betamax`: Temperature schedule
  - Start high (0.01-0.05) for exploration
  - End low (0.5-1.0) for refinement

- `anneal`: Annealing strategy
  - `'inverse'`: Recommended for most cases
  - `'exp'`: Faster convergence
  - `'lin'`: Linear schedule

### Step 6: Extract Best Solution

Get the best placement from all trials:

```python
# Find trial with lowest energy
best_idx = torch.argmin(energies)
best_energy = energies[best_idx]
best_config = configs[best_idx]

print(f"Best energy: {best_energy:.2f}")

# Convert to real FPGA coordinates
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(best_config)
```

### Step 7: Legalize Placement

Resolve any overlaps:

```python
from fem_placer import Legalizer

legalizer = Legalizer(placer=placer, device='cpu')

# Create instance IDs
logic_ids = torch.arange(num_inst)

# Legalize
placement, overlaps, hpwl_before, hpwl_after = legalizer.legalize_placement(
    logic_coords=real_coords,
    logic_ids=logic_ids,
    io_coords=None,
    io_ids=None,
    include_io=False
)

print(f"Overlaps: {overlaps}")
print(f"HPWL before legalization: {hpwl_before['hpwl_no_io']:.2f}")
print(f"HPWL after legalization: {hpwl_after['hpwl_no_io']:.2f}")
```

### Step 8: Route and Visualize

Create routing and visualization:

```python
from fem_placer import Router, PlacementDrawer

# Route connections
router = Router(placer=placer)
routes = router.route_connections(J, placement[0])

# Visualize
drawer = PlacementDrawer(placer=placer, num_subplots=5)
drawer.draw_place_and_route(
    logic_coords=placement[0],
    routes=routes,
    io_coords=None,
    include_io=False,
    iteration=1000,
    title_suffix='Final Placement'
)
```

---

## Advanced Usage

### Using GPU Acceleration

Enable CUDA for faster optimization:

```python
import torch

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Run optimization on GPU
configs, energies = optimizer.optimize(
    dev=device,
    num_trials=20,  # Can run more trials on GPU
    num_steps=2000
)

# Legalizer also supports GPU
legalizer = Legalizer(placer=placer, device=device)
```

### Timing-Aware Placement

Use the Timer class for timing-critical designs:

```python
from fem_placer import Timer
import torch

# Create timer
timer = Timer()

# Setup timing analysis (if you have timing data)
# timer.setup_timing_analysis(design, timing_library)

# Create timing criticality weights
# Higher values = more critical paths
timing_criticality = torch.ones(num_inst, num_inst)
timing_criticality[critical_path_nets] *= 10.0

# Calculate timing-aware HPWL
timing_hpwl = timer.calculate_timing_based_hpwl(
    J=J,
    p=best_config,
    area_width=area_length,
    timing_criticality=timing_criticality
)

print(f"Timing-weighted HPWL: {timing_hpwl:.2f}")
```

### Congestion-Aware Placement

Avoid congested regions:

```python
from fem_placer import Timer

timer = Timer()

# Create congestion map (penalty per site)
congestion_map = torch.ones(num_site)
# Mark congested regions with higher penalties
congested_sites = [10, 11, 12, 20, 21, 22]
congestion_map[congested_sites] = 10.0

# Calculate congestion-aware HPWL
congestion_hpwl = timer.calculate_congestion_aware_hpwl(
    J=J,
    p=best_config,
    area_width=area_length,
    congestion_map=congestion_map
)
```

### Using Simulated Bifurcation Solver

Alternative solver for comparison:

```python
from fem_placer import SBSolver

# Create SB solver
sb_solver = SBSolver(mode='discrete', heated=True, device='cpu')

# Solve as QUBO problem
# First, create QUBO matrix from connectivity
Q = create_qubo_matrix(J, site_coords)  # User-defined function

# Solve
bits, energy = sb_solver.solve_qubo(Q, agents=10, max_steps=5000)

print(f"SB solution energy: {energy:.2f}")
```

### Hypergraph Partitioning

For hierarchical placement or partitioning:

```python
from fem_placer import (
    expected_hyperbmincut,
    balance_constrain,
    infer_hyperbmincut
)

# Define hyperedges (nets connecting multiple instances)
hyperedges = [
    [0, 1, 2],      # Net connecting instances 0, 1, 2
    [1, 3, 4],      # Net connecting instances 1, 3, 4
    [2, 4, 5, 6]    # Net connecting instances 2, 4, 5, 6
]

# Create partition probabilities (2 partitions)
p_partition = torch.rand(1, num_inst, 2)
p_partition = torch.softmax(p_partition, dim=-1)

# Calculate expected cut
cut_value = expected_hyperbmincut(J, p_partition, hyperedges)

# Add balance constraint (40-60% split)
balance_loss = balance_constrain(J, p_partition, U_max=0.6, L_min=0.4)

# Total objective
total_loss = cut_value + balance_loss

# Infer final partition
partition, final_cut = infer_hyperbmincut(J, p_partition, hyperedges)
print(f"Partition A: {(partition == 0).sum()} instances")
print(f"Partition B: {(partition == 1).sum()} instances")
print(f"Cut edges: {final_cut}")
```

### Multi-Step Visualization

Track optimization progress:

```python
# Create drawer with visualization at specific steps
drawer = PlacementDrawer(placer=placer, num_subplots=5)

optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=site_coords,
    drawer=drawer,
    visualization_steps=[100, 300, 500, 700, 900]  # Steps to visualize
)

# Run optimization (will automatically save intermediate results)
configs, energies = optimizer.optimize(num_trials=5, num_steps=1000)

# Draw multi-step visualization
drawer.draw_multi_step_placement('optimization_progress.png')
```

### Custom Annealing Schedules

Implement custom temperature schedules:

```python
# Use built-in schedules
configs, energies = optimizer.optimize(
    anneal='inverse',  # or 'lin', 'exp'
    betamin=0.01,
    betamax=0.5
)

# For custom schedules, modify betamin/betamax
# Linear: beta_t = betamin + (betamax - betamin) * t / T
# Exponential: beta_t = betamin * (betamax / betamin) ^ (t / T)
# Inverse: beta_t = betamin * betamax / (betamin + (betamax - betamin) * t / T)
```

---

## Examples

### Example 1: Basic Placement

```python
import torch
from fem_placer import FpgaPlacer, FPGAPlacementOptimizer, Legalizer
from fem_placer.utils import parse_fpga_design

# Load design
placer = FpgaPlacer(utilization_factor=0.5)
placer.init_placement('design.dcp', 'output.dcp')

# Parse
num_inst, num_site, J, _ = parse_fpga_design(placer)
area_length = placer.bbox['area_length']
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

# Optimize
optimizer = FPGAPlacementOptimizer(
    num_inst, site_coords.shape[0], J, site_coords, constraint_weight=1.0
)
configs, energies = optimizer.optimize(num_trials=5, num_steps=500, dev='cpu', area_width=area_length)

# Legalize
best_idx = torch.argmin(energies)
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(configs[best_idx])
legalizer = Legalizer(placer, 'cpu')
placement, overlaps, _, hpwl_after = legalizer.legalize_placement(
    real_coords, torch.arange(num_inst)
)

print(f"Final HPWL: {hpwl_after['hpwl_no_io']:.2f}, Overlaps: {overlaps}")
```

### Example 2: GPU-Accelerated Placement

```python
import torch
from fem_placer import FpgaPlacer, FPGAPlacementOptimizer, Legalizer
from fem_placer.utils import parse_fpga_design

device = 'cuda' if torch.cuda.is_available() else 'cpu'

placer = FpgaPlacer(utilization_factor=0.4)
placer.init_placement('design.dcp', 'output.dcp')

num_inst, num_site, J, _ = parse_fpga_design(placer)
area_length = placer.bbox['area_length']
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

optimizer = FPGAPlacementOptimizer(num_inst, site_coords.shape[0], J, site_coords)
configs, energies = optimizer.optimize(
    num_trials=20, num_steps=2000, dev=device, area_width=area_length
)

best_idx = torch.argmin(energies)
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(configs[best_idx])
legalizer = Legalizer(placer, device)
placement, _, _, hpwl = legalizer.legalize_placement(real_coords, torch.arange(num_inst))

print(f"GPU-accelerated HPWL: {hpwl['hpwl_no_io']:.2f}")
```

### Example 3: Timing-Aware Placement

```python
from fem_placer import Timer

# ... (setup as in Example 1)

timer = Timer()
timing_criticality = torch.ones(num_inst, num_inst)
# Mark critical paths with higher weights
timing_criticality[critical_nets] = 10.0

timing_hpwl = timer.calculate_timing_based_hpwl(
    J, configs[best_idx], area_length, timing_criticality
)
print(f"Timing-weighted HPWL: {timing_hpwl:.2f}")
```

### Example 4: Comparing Multiple Solutions

```python
# Run multiple trials
configs, energies = optimizer.optimize(num_trials=20, num_steps=1000)

# Get top 5 solutions
top_5_indices = torch.topk(energies, k=5, largest=False).indices

print("Top 5 solutions:")
for i, idx in enumerate(top_5_indices):
    print(f"{i+1}. Energy: {energies[idx]:.2f}")

    # Legalize each solution
    logic_grid = placer.get_grid('logic')
    real_coords = logic_grid.to_real_coords_tensor(configs[idx])
    legalizer = Legalizer(placer, 'cpu')
    placement, overlaps, _, hpwl = legalizer.legalize_placement(
        real_coords, torch.arange(num_inst)
    )
    print(f"   HPWL: {hpwl['hpwl_no_io']:.2f}, Overlaps: {overlaps}")
```

---

## Best Practices

### 1. Choosing Utilization Factor

```python
# Low utilization (0.3-0.5): Better for timing-critical designs
placer = FpgaPlacer(utilization_factor=0.4)

# High utilization (0.7-0.9): Dense packing, harder routing
placer = FpgaPlacer(utilization_factor=0.8)
```

**Guidelines:**
- Start with 0.5 and adjust based on results
- Lower utilization if routing fails or timing is not met
- Higher utilization for area-constrained designs

### 2. Optimization Parameters

```python
# Quick test run
configs, energies = optimizer.optimize(
    num_trials=3,
    num_steps=500,
    betamin=0.01,
    betamax=0.3
)

# Production run
configs, energies = optimizer.optimize(
    num_trials=20,
    num_steps=2000,
    betamin=0.01,
    betamax=0.5,
    anneal='inverse'
)
```

### 3. Constraint Weight Tuning

```python
# Start with default
optimizer = FPGAPlacementOptimizer(
    ...,
    constraint_weight=num_inst / 2.0
)

# If many overlaps, increase weight
optimizer = FPGAPlacementOptimizer(
    ...,
    constraint_weight=num_inst  # 2x default
)

# If HPWL is too high, decrease weight
optimizer = FPGAPlacementOptimizer(
    ...,
    constraint_weight=num_inst / 4.0  # 0.5x default
)
```

### 4. Memory Management

```python
# For large designs, use float16 to save memory
configs, energies = optimizer.optimize(
    ...,
    dtype=torch.float16
)

# Clear unused tensors
del configs
torch.cuda.empty_cache()  # If using CUDA
```

### 5. Reproducibility

```python
# Set random seed for reproducible results
torch.manual_seed(42)

configs, energies = optimizer.optimize(
    ...,
    seed=42
)
```

---

## Troubleshooting

### Issue: High number of overlaps after legalization

**Solution:**
```python
# Increase constraint weight
optimizer = FPGAPlacementOptimizer(
    ...,
    constraint_weight=num_inst * 2  # Increase
)

# Or run more optimization steps
configs, energies = optimizer.optimize(
    num_steps=2000  # Increase from 1000
)
```

### Issue: HPWL is too high

**Solution:**
```python
# Decrease constraint weight
optimizer = FPGAPlacementOptimizer(
    ...,
    constraint_weight=num_inst / 4.0
)

# Try more trials
configs, energies = optimizer.optimize(
    num_trials=30
)
```

### Issue: Optimization is slow

**Solution:**
```python
# Use GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
configs, energies = optimizer.optimize(dev=device)

# Reduce num_steps for initial testing
configs, energies = optimizer.optimize(num_steps=500)

# Use fewer trials
configs, energies = optimizer.optimize(num_trials=5)
```

### Issue: Out of memory

**Solution:**
```python
# Use float16
configs, energies = optimizer.optimize(dtype=torch.float16)

# Reduce batch size (num_trials)
configs, energies = optimizer.optimize(num_trials=5)

# Clear cache
torch.cuda.empty_cache()
```

### Issue: Results are inconsistent

**Solution:**
```python
# Set random seed
torch.manual_seed(42)
configs, energies = optimizer.optimize(seed=42)

# Run more trials and average
configs, energies = optimizer.optimize(num_trials=50)
```

### Issue: FileNotFoundError when loading DCP

**Solution:**
```python
import os

dcp_file = 'path/to/design.dcp'
if not os.path.exists(dcp_file):
    print(f"Error: {dcp_file} not found")
else:
    placer.init_placement(dcp_file, 'output.dcp')
```

---

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed documentation
- Check the [Algorithm Overview](ALGORITHM.md) for technical details
- See the [Testing Guide](testing.md) for testing
- Explore the [Migration Guide](MIGRATION_GUIDE.md) for API changes

For questions or issues, visit: https://github.com/yao-baijian/fem/issues
