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

### Basic FPGA Placement Workflow

1. **Load Design**: Initialize `FpgaPlacer` with a RapidWright DCP file
2. **Create Optimizer**: Set up `FPGAPlacementOptimizer` with coupling matrices and site coordinates
3. **Run Optimization**: Execute FEM with temperature annealing to minimize HPWL + constraints
4. **Legalize**: Use `Legalizer` to resolve placement overlaps
5. **Route**: Apply `Router` for connection visualization

See [tests/test_fpga_placement.py](tests/test_fpga_placement.py) for a complete working example.

### ML-Assisted Parameter Tuning

The framework includes machine learning models to predict optimal constraint weights (`alpha`, `beta`) based on circuit features:

```bash
# Train models on historical placement data
python -c "from ml.train import train_from_csv; train_from_csv(target='alpha')"

# Use trained model to predict parameters for new designs
from ml.predict import predict_alpha
optimal_alpha = predict_alpha(circuit_features)
```

See [tests/test_train_alpha.py](tests/test_train_alpha.py) for the two-stage parameter sweep.

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

## 🔧 Advanced Features

- **Multi-Region Placement**: Separate optimization for logic and IO areas with independent constraint weights (α, β)
- **Custom Cost Functions**: Extensible objective framework for domain-specific optimizations

See [fem_placer/objectives.py](fem_placer/objectives.py) for API details.

## 📚 References

- [Original FEM framework](https://github.com/Fanerst/FEM)
- [RapidWright](https://www.rapidwright.io/) - FPGA CAD framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Simulated Bifurcation Library](https://pypi.org/project/simulated-bifurcation/)


## 📄 License

MIT License - see LICENSE file for details

