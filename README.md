# FPGA Placement using FEM Framework

A clean, modular FPGA placement tool built on top of the [FEM (Field Effect Method)](https://github.com/Fanerst/FEM) optimization framework.

## 🎯 Key Features

- **Clean Architecture**: FEM framework as external dependency (git submodule)
- **Extensible Design**: Uses FEM's `customize` interface - no core modifications
- **Standard Python Package**: Modern `pyproject.toml` configuration
- **FPGA-Specific**: Optimized objectives (HPWL, timing) and constraints
- **Complete Pipeline**: Placement → Legalization → Routing → Visualization

## 📦 Project Structure

```
FPGA-Placement-FEM/
├── external/
│   └── FEM/  (git submodule)     # Original FEM framework
├── fem_placer/             # Our FPGA placement package
│   ├── __init__.py
│   ├── placer.py                 # RapidWright interface
│   ├── objectives.py             # HPWL and constraint functions
│   ├── drawer.py                 # Visualization tools
│   ├── legalizer.py              # Placement legalization
│   ├── router.py                 # Connection routing
│   └── utils.py                  # Utility functions
├── tests/
│   └── test_fpga_placement.py
├── tcl/                          # Vivado TCL scripts
├── pyproject.toml                # Modern Python package config
├── requirements.txt              # Dependencies
└── README.md
```

## 🚀 Installation

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/yourusername/fpga-placement.git
cd fpga-placement
```

If you forgot `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install Dependencies

**Option A: Using uv (recommended - fast and modern)**
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install optional dependencies
uv pip install rapidwright  # For FPGA design handling
```

**Option B: Using pip (traditional)**
```bash
pip install -r requirements.txt
pip install rapidwright  # For FPGA design handling
```

**Option C: Development mode with pip**
```bash
pip install -e .
pip install -e ".[dev,rapidwright]"
```

### 3. Add FEM to Python Path

The package automatically handles FEM imports from `external/FEM`.

## 📖 Usage

### Basic Example

```python
import torch
import sys
sys.path.insert(0, 'external')  # Add FEM to path

from FEM import FEM
from fem_placer import (
    FpgaPlacer,
    PlacementDrawer,
    Legalizer,
    Router,
    expected_fpga_placement_xy,
    infer_placements_xy
)
from fem_placer.utils import parse_fpga_design

# Initialize FPGA placer
fpga_wrapper = FpgaPlacer()
fpga_wrapper.init_placement('./design.dcp', 'output.dcp')

# Parse design
num_inst, num_site, J, J_extend = parse_fpga_design(fpga_wrapper)

# Define customize functions for FEM
def customize_expected_func(coupling_matrix, p_list):
    p_x, p_y = p_list
    return expected_fpga_placement_xy(coupling_matrix, p_x, p_y)

def customize_infer_func(coupling_matrix, p_list):
    p_x, p_y = p_list
    return infer_placements_xy(coupling_matrix, p_x, p_y)

# Create FEM problem using customize interface
case_placements = FEM.from_couplings(
    'customize',  # Use FEM's customize type
    num_inst,
    num_inst * (num_inst - 1) // 2,
    J,
    customize_expected_func=customize_expected_func,
    customize_infer_func=customize_infer_func
)

# Solve
case_placements.set_up_solver(num_trials=10, num_steps=1000, dev='cpu')
config, result = case_placements.solve()

# Legalize and route
legalizer = Legalizer(fpga_wrapper.bbox)
placement = legalizer.legalize_placement(config[0])
```

### Run Tests

```bash
python tests/test_fpga_placement_refactored.py
```

## 🔄 Updating FEM

To update FEM to the latest version:

```bash
cd external/FEM
git pull origin main
cd ../..
git add external/FEM
git commit -m "Update FEM to latest version"
```

## 📐 Architecture

### Why This Design?

1. **FEM as Submodule**: Keeps FEM code separate, easy to update from upstream
2. **Customize Interface**: Uses FEM's built-in extensibility - no modifications needed
3. **Standard Package**: Follows Python packaging best practices
4. **Modular**: Each component (placer, legalizer, router) is independent

### Key Components

- **FpgaPlacer**: Interfaces with RapidWright for FPGA design handling
- **Objectives**: HPWL calculation, constraint functions
- **Legalizer**: Resolves overlaps, snaps to valid sites
- **Router**: Computes wire paths for visualization
- **Drawer**: Matplotlib-based visualization

## 🆚 Comparison with Direct Copy

| Aspect | Old (Direct Copy) | New (Submodule) |
|--------|------------------|-----------------|
| FEM Updates | Manual merge | `git submodule update` |
| Code Ownership | Unclear | Crystal clear |
| Modifications | Mixed with FEM | Separate package |
| Maintenance | Difficult | Easy |
| Distribution | Bloated | Clean |

## 📝 Development

### Project Goals

This project demonstrates how to:
- Build domain-specific tools on top of general frameworks
- Use git submodules for dependency management
- Follow Python packaging best practices
- Maintain clean code boundaries

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 References

- [FEM Framework](https://github.com/Fanerst/FEM)
- [RapidWright](https://www.rapidwright.io/)
- [Python Packaging Guide](https://packaging.python.org/)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- FEM framework by [Fanerst](https://github.com/Fanerst/FEM)
- RapidWright by Xilinx Research Labs
