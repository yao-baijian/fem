# FPGA Placement using Free Energy Minimization

A modular FPGA placement toolkit using Free Energy Minimization (FEM) with QUBO formulation for optimization.

## 🚀 Installation

### Prerequisites

- Python 3.9+
- PyTorch
- RapidWright (for FPGA design handling)

### Installation Steps

**Using uv (recommended - fast and modern)**

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

**Using pip (traditional)**

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

**Build Documentations**
```bash
uv pip install ".[docs]"

./build_docs.sh build

./build_docs.sh serve
```

## 📖 Usage

### Basic FPGA Placement Example

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

# 1. Initialize FPGA placer and load design
fpga_placer = FpgaPlacer(utilization_factor=0.4)
fpga_placer.init_placement('design.dcp', 'output.dcp')

# 2. Parse design to get coupling matrix
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_placer)

# 3. Create site coordinates matrix for all grid positions
area_length = fpga_placer.bbox['area_length']
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

# 4. Create optimizer with QUBO formulation
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
    betamax=0.5,
    anneal='inverse',
    optimizer='adam',
    learning_rate=0.1
)

# 6. Get best solution and convert to real coordinates
best_idx = torch.argmin(result)
grid_coords = config[best_idx]
logic_grid = fpga_placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(grid_coords)

# 7. Legalize placement
legalizer = Legalizer(placer=fpga_placer, device='cpu')
logic_ids = torch.arange(num_inst)
placement_legalized, overlap, hpwl_before, hpwl_after = legalizer.legalize_placement(
    real_coords, logic_ids
)

# 8. Route and visualize
router = Router(placer=fpga_placer)
routes = router.route_connections(J, placement_legalized[0])

drawer = PlacementDrawer(placer=fpga_placer, num_subplots=5)
drawer.draw_place_and_route(
    logic_coords=placement_legalized[0],
    routes=routes,
    io_coords=None,
    include_io=False,
    iteration=1000,
    title_suffix='Final'
)

print(f"HPWL after legalization: {hpwl_after['hpwl_no_io']:.2f}")
print(f"Overlaps: {overlap}")
```

### ML Parameter Prediction (Optional)

```python
from ml_alpha import train_from_csv, predict_alpha, create_default_model

# Train model
model = train_from_csv('training_data.csv')

# Predict optimal alpha parameter
features = {
    'num_inst': 100,
    'num_site': 200,
    'num_trial': 10,
    'betamin': 0.01,
    # ... other features
}
alpha = predict_alpha(model, features)
```

## 🧪 Testing

Run the full test suite:

```bash
# Run all tests (50+ tests)
uv run pytest tests/ -v
```

## 📐 Architecture

### Algorithm: QUBO-Based FEM

The optimizer uses a QUBO (Quadratic Unconstrained Binary Optimization) formulation:

1. **Variables**: Probability distribution `p[i, s]` for instance `i` at site `s`
2. **Objective**: Minimize expected HPWL + constraint violations
3. **Optimization**: Free energy minimization with temperature annealing
4. **Inference**: Argmax over probabilities to get hard assignments

### Key Components

- **FpgaPlacer**: RapidWright interface for loading/saving FPGA designs
- **FPGAPlacementOptimizer**: FEM-based placement using QUBO formulation
- **Grid & NetManager**: Design structure management (from master branch)
- **Legalizer**: Resolves overlaps using greedy and global optimization
- **Router**: Manhattan routing for visualization
- **Timer**: Timing-aware and congestion-aware placement

## 🔧 Advanced Features

### Timing-Aware Placement

```python
from fem_placer import Timer

timer = Timer()
timing_hpwl = timer.calculate_timing_based_hpwl(
    J, p, area_width, timing_criticality
)
```

### Hypergraph Balanced Min-Cut

```python
from fem_placer import expected_hyperbmincut, balance_constrain

hyperedges = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
cut = expected_hyperbmincut(J, p, hyperedges)
balance_loss = balance_constrain(J, p, U_max=0.6, L_min=0.4)
```

### Code Quality

The codebase follows Python best practices:
- Type hints where applicable
- Comprehensive docstrings
- Clean imports (no unused dependencies)
- No trailing whitespace
- Proper error handling

## 📚 References

- [Original FEM framework](https://github.com/Fanerst/FEM)
- [RapidWright](https://www.rapidwright.io/) - FPGA CAD framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Simulated Bifurcation Library](https://pypi.org/project/simulated-bifurcation/)


## 📄 License

MIT License - see LICENSE file for details

