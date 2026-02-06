# Quick Start Guide

Get started with FPGA Placement FEM in minutes.

## 5-Minute Tutorial

### Step 1: Install

```bash
# Clone and install
git clone https://github.com/yao-baijian/fem.git
cd fem
pip install -e .
pip install torch rapidwright
```

### Step 2: Prepare Design

You need an FPGA design in DCP (Design Checkpoint) format. If you don't have one, you can:

- Use test designs from the repository
- Generate one from Vivado
- Download example designs

### Step 3: Basic Placement

Create a Python script `placement.py`:

```python
import torch
from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
    Legalizer
)
from fem_placer.utils import parse_fpga_design

# Load design
print("Loading design...")
placer = FpgaPlacer(utilization_factor=0.4)
placer.init_placement('design.dcp', 'output.dcp')

# Parse connectivity
print("Parsing design...")
num_inst, num_site, J, _ = parse_fpga_design(placer)
print(f"Instances: {num_inst}, Sites: {num_site}")

# Create site coordinates
area_length = placer.bbox['area_length']
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

# Optimize
print("Optimizing placement...")
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=site_coords
)

configs, energies = optimizer.optimize(
    num_trials=5,
    num_steps=500,
    dev='cpu',
    area_width=area_length
)

# Legalize
print("Legalizing placement...")
best_idx = torch.argmin(energies)
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(configs[best_idx])

legalizer = Legalizer(placer=placer, device='cpu')
placement, overlaps, _, hpwl = legalizer.legalize_placement(
    real_coords, torch.arange(num_inst)
)

print(f"✅ Done!")
print(f"Final HPWL: {hpwl['hpwl_no_io']:.2f}")
print(f"Overlaps: {overlaps}")
```

Run the script:

```bash
python placement.py
```

## Understanding the Output

The script will output:

```
Loading design...
Parsing design...
Instances: 150, Sites: 400
Optimizing placement...
Trial 1/5: Energy = 1245.32
Trial 2/5: Energy = 1198.45
Trial 3/5: Energy = 1201.78
Trial 4/5: Energy = 1189.23
Trial 5/5: Energy = 1195.67
Legalizing placement...
✅ Done!
Final HPWL: 1189.23
Overlaps: 0
```

- **Energy**: Lower is better (HPWL + constraint violations)
- **HPWL**: Half-Perimeter Wirelength
- **Overlaps**: Should be 0 after legalization

## Add Visualization

Enhance the script with routing and visualization:

```python
from fem_placer import Router, PlacementDrawer

# ... (previous code)

# Route connections
print("Routing...")
router = Router(placer=placer)
routes = router.route_connections(J, placement[0])

# Visualize
print("Drawing...")
drawer = PlacementDrawer(placer=placer)
drawer.draw_place_and_route(
    logic_coords=placement[0],
    routes=routes,
    io_coords=None,
    include_io=False,
    iteration=500,
    title_suffix='Final Placement'
)
```

This will display a visualization of the placement and routing.

## GPU Acceleration

To use GPU (if available):

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

configs, energies = optimizer.optimize(
    num_trials=10,  # Can run more trials on GPU
    num_steps=1000,  # More steps for better quality
    dev=device,
    area_width=area_length
)

legalizer = Legalizer(placer=placer, device=device)
```

## Common Parameters

### Utilization Factor

Controls placement density:

```python
# Low utilization (more space)
placer = FpgaPlacer(utilization_factor=0.3)

# High utilization (dense packing)
placer = FpgaPlacer(utilization_factor=0.8)
```

**Recommendation**: Start with 0.4-0.5

### Optimization Parameters

```python
configs, energies = optimizer.optimize(
    num_trials=10,      # More trials = better solution
    num_steps=1000,     # More steps = better convergence
    betamin=0.01,       # Starting temperature (exploration)
    betamax=0.5,        # Ending temperature (refinement)
    learning_rate=0.1   # Optimization step size
)
```

**Recommendations**:
- Small designs (<100 instances): `num_trials=5, num_steps=500`
- Medium designs (100-500): `num_trials=10, num_steps=1000`
- Large designs (>500): `num_trials=20, num_steps=2000`

## Example Workflows

### Workflow 1: Quick Test

Fast placement for testing:

```python
configs, energies = optimizer.optimize(
    num_trials=3,
    num_steps=300,
    dev='cpu'
)
```

### Workflow 2: Production Quality

High-quality placement:

```python
configs, energies = optimizer.optimize(
    num_trials=20,
    num_steps=2000,
    dev='cuda',
    betamin=0.01,
    betamax=0.5,
    anneal='inverse'
)
```

### Workflow 3: Timing-Critical

Timing-aware placement:

```python
from fem_placer import Timer

timer = Timer()

# Create timing weights
timing_criticality = torch.ones(num_inst, num_inst)
timing_criticality[critical_nets] = 10.0

# Calculate timing-weighted HPWL
timing_hpwl = timer.calculate_timing_based_hpwl(
    J, configs[best_idx], area_length, timing_criticality
)
```

## Next Steps

Now that you've run a basic placement:

1. **Explore Parameters**: Adjust `utilization_factor`, `num_trials`, `num_steps`
2. **Try Different Designs**: Test on various DCP files
3. **Add Timing**: Use `Timer` for timing-aware placement
4. **Compare Solvers**: Try `SBSolver` for comparison
5. **Read Documentation**: Check [User Guide](USER_GUIDE.md) and [API Reference](API_REFERENCE.md)

## Common Issues

### "FileNotFoundError: design.dcp not found"

Make sure your DCP file path is correct:

```python
import os
dcp_path = 'path/to/design.dcp'
if not os.path.exists(dcp_path):
    print(f"File not found: {dcp_path}")
```

### "RuntimeError: CUDA out of memory"

Reduce batch size or use CPU:

```python
configs, energies = optimizer.optimize(
    num_trials=5,  # Reduce from 10
    dev='cpu'      # Use CPU instead of CUDA
)
```

### High HPWL after legalization

Try adjusting constraint weight:

```python
optimizer = FPGAPlacementOptimizer(
    ...,
    constraint_weight=num_inst * 2  # Increase weight
)
```

## Tips

!!! tip "Start Small"
    Begin with small designs and quick optimization to understand the workflow, then scale up.

!!! tip "Use GPU"
    GPU acceleration can be 10-100x faster for large designs.

!!! tip "Multiple Trials"
    Running multiple trials and keeping the best result significantly improves quality.

!!! tip "Visualize Progress"
    Use visualization to understand how the optimizer converges.

## Learn More

- **[User Guide](USER_GUIDE.md)**: Comprehensive tutorials
- **[API Reference](API_REFERENCE.md)**: Complete API documentation
- **[User Guide](USER_GUIDE.md)**: Comprehensive examples and tutorials
- **[Algorithm](ALGORITHM.md)**: Understanding the math

---

Ready to dive deeper? Check out the [User Guide](USER_GUIDE.md) for advanced features!
