# FPGA Placement using Free Energy Minimization

<div style="text-align: center; margin: 2em 0;">
  <h2>🚀 A Modern FPGA Placement Toolkit</h2>
  <p style="font-size: 1.2em; color: #666;">
    Optimize your FPGA designs with Free Energy Minimization and QUBO formulation
  </p>
</div>

---

## Overview

**FPGA Placement FEM** is a modular toolkit for optimizing FPGA placements using advanced optimization techniques. Built on PyTorch and RapidWright, it provides a clean, efficient API for placement optimization with state-of-the-art algorithms.

### Key Features

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **Fast & Efficient**

    ---

    GPU-accelerated optimization with PyTorch, supporting designs with 1000+ instances

-   :material-math-integral:{ .lg .middle } **QUBO Formulation**

    ---

    Uses Quadratic Unconstrained Binary Optimization with site coordinate matrices for flexible placement

-   :material-auto-fix:{ .lg .middle } **Complete Pipeline**

    ---

    Placement → Legalization → Routing → Visualization in a single workflow

-   :material-chart-line:{ .lg .middle } **Multiple Solvers**

    ---

    FEM-based optimizer with Simulated Bifurcation baseline for comparison

-   :material-timer:{ .lg .middle } **Timing-Aware**

    ---

    Built-in congestion and timing analysis for optimized placements

-   :material-test-tube:{ .lg .middle } **Well-Tested**

    ---

    50+ unit tests with comprehensive coverage and continuous integration

</div>

---

## Quick Example

```python
import torch
from fem_placer import (
    FpgaPlacer,
    FPGAPlacementOptimizer,
    Legalizer
)
from fem_placer.utils import parse_fpga_design

# 1. Load FPGA design
placer = FpgaPlacer(utilization_factor=0.4)
placer.init_placement('design.dcp', 'output.dcp')

# 2. Parse design
num_inst, num_site, J, _ = parse_fpga_design(placer)

# 3. Create site coordinates
area_length = placer.bbox['area_length']
site_coords = torch.cartesian_prod(
    torch.arange(area_length, dtype=torch.float32),
    torch.arange(area_length, dtype=torch.float32)
)

# 4. Optimize placement
optimizer = FPGAPlacementOptimizer(
    num_inst=num_inst,
    num_site=site_coords.shape[0],
    coupling_matrix=J,
    site_coords_matrix=site_coords,
    constraint_weight=1.0
)

configs, energies = optimizer.optimize(
    num_trials=10,
    num_steps=1000,
    dev='cpu',
    area_width=area_length
)

# 5. Legalize
best_idx = torch.argmin(energies)
logic_grid = placer.get_grid('logic')
real_coords = logic_grid.to_real_coords_tensor(configs[best_idx])

legalizer = Legalizer(placer=placer, device='cpu')
placement, overlaps, _, hpwl = legalizer.legalize_placement(
    real_coords, torch.arange(num_inst)
)

print(f"Final HPWL: {hpwl['hpwl_no_io']:.2f}")
```

---

## Installation

=== "Using uv (Recommended)"

    ```bash
    # Clone repository
    git clone https://github.com/yao-baijian/fem.git
    cd fem

    # Install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Install dependencies
    uv sync
    uv pip install torch rapidwright

    # Optional: ML dependencies
    uv pip install scikit-learn joblib
    ```

=== "Using pip"

    ```bash
    # Clone repository
    git clone https://github.com/yao-baijian/fem.git
    cd fem

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Install package
    pip install -e .
    pip install torch rapidwright

    # Optional: ML dependencies
    pip install -e ".[ml]"
    ```

---

## Architecture

The toolkit follows a clean, modular architecture:

```mermaid
graph LR
    A[DCP File] --> B[FpgaPlacer]
    B --> C[parse_fpga_design]
    C --> D[FPGAPlacementOptimizer]
    D --> E[Legalizer]
    E --> F[Router]
    F --> G[PlacementDrawer]

    style D fill:#e1f5ff
    style E fill:#ffe1e1
    style G fill:#e1ffe1
```

### Core Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **FpgaPlacer** | RapidWright interface | Design loading, grid management |
| **FPGAPlacementOptimizer** | FEM optimization | QUBO formulation, annealing |
| **Legalizer** | Overlap resolution | Greedy + global optimization |
| **Router** | Connection routing | Manhattan routing |
| **PlacementDrawer** | Visualization | Multi-step tracking |
| **Timer** | Timing analysis | Congestion-aware placement |

---

## Algorithm Overview

The optimizer uses a **QUBO (Quadratic Unconstrained Binary Optimization)** formulation with **Free Energy Minimization**:

1. **Variables**: Probability distribution `p[i, s]` for instance `i` at site `s`
2. **Objective**: Minimize expected HPWL + constraint violations
3. **Optimization**: Temperature annealing with gradient descent
4. **Inference**: Argmax over probabilities for hard assignments

### Mathematical Formulation

Expected HPWL:
```
E[HPWL] = Σ_{i,j} J[i,j] * ||E[coords_i] - E[coords_j]||
```

Where expected coordinates:
```
E[coords_i] = Σ_s p[i,s] * site_coords[s]
```

Constraints:
```
L_constraint = Σ_i (Σ_s p[i,s] - 1)² + Σ_s max(0, Σ_i p[i,s] - 1)²
```

Total objective:
```
L(p) = E[HPWL] + α * L_constraint
```

!!! tip "Learn More"
    See the [Algorithm Overview](ALGORITHM.md) for detailed mathematical formulation and complexity analysis.

---

## Features in Detail

### 🎯 QUBO-Based Optimization

- Soft probability assignments enable gradient-based optimization
- Temperature annealing for global optimization
- Multiple trials with best-solution selection

### ⚡ Performance

- **GPU Acceleration**: CUDA support for large designs
- **Parallel Trials**: Run multiple optimizations concurrently
- **Efficient Implementation**: Vectorized tensor operations

### 🔧 Flexibility

- **Multiple Solvers**: FEM, Simulated Bifurcation
- **Customizable Objectives**: Timing, congestion, custom metrics
- **Extensible Architecture**: Easy to add new features

### 📊 Visualization

- **Multi-Step Tracking**: Visualize optimization progress
- **Routing Display**: Manhattan routing visualization
- **Congestion Maps**: Identify problematic regions

---

## Comparison with Other Tools

| Feature | FEM Placer | VPR | Commercial Tools |
|---------|-----------|-----|------------------|
| Algorithm | FEM + QUBO | Simulated Annealing | Proprietary |
| Flexibility | High | Medium | Low |
| GPU Support | ✅ Yes | ❌ No | Varies |
| Open Source | ✅ Yes | ✅ Yes | ❌ No |
| Python API | ✅ Yes | ❌ No | ❌ No |
| Extensible | ✅ Yes | ⚠️ Limited | ❌ No |

---

## Use Cases

### Research

- **Algorithm Development**: Test new placement algorithms
- **Benchmarking**: Compare different approaches
- **Publication**: Reproducible results with open-source code

### Industry

- **FPGA Design**: Optimize custom FPGA designs
- **Rapid Prototyping**: Quick placement iterations
- **Integration**: Python API for custom workflows

### Education

- **Learning**: Understand FPGA placement algorithms
- **Teaching**: Use in courses on EDA and optimization
- **Projects**: Base for student projects

---

## Documentation Structure

<div class="grid cards" markdown>

-   :material-book-open-page-variant:{ .lg } **[User Guide](USER_GUIDE.md)**

    ---

    Learn how to use the toolkit with step-by-step tutorials and examples

-   :material-code-braces:{ .lg } **[API Reference](API_REFERENCE.md)**

    ---

    Complete API documentation for all classes and functions

-   :material-math-compass:{ .lg } **[Algorithm](ALGORITHM.md)**

    ---

    Technical details of the QUBO formulation and FEM approach

</div>

---

## Getting Help

- **Documentation**: Browse the complete documentation on this site
- **GitHub Issues**: [Report bugs or request features](https://github.com/yao-baijian/fem/issues)
- **Examples**: Check the `tests/` and `scripts/` directories
- **Migration Guide**: [Migrating from master branch](MIGRATION_GUIDE.md)

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{fpga_placement_fem,
  title = {FPGA Placement using Free Energy Minimization},
  author = {FPGA Placement Team},
  year = {2026},
  url = {https://github.com/yao-baijian/fem}
}
```

---

## License

MIT License

---

## Next Steps

<div class="grid cards" markdown>

-   Start with the [Quick Start Guide](quickstart.md)
-   Explore [Code Explained](CODE_EXPLAINED.md)
-   Read the [API Reference](API_REFERENCE.md)
-   Check the [Algorithm Overview](ALGORITHM.md)

</div>
