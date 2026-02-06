# Installation Guide

Complete installation instructions for the FPGA Placement FEM package.

## Prerequisites

- **Python**: 3.9 or higher
- **Operating System**: macOS, Linux, or Windows
- **Hardware**:
  - CPU: Any modern processor
  - GPU: NVIDIA GPU with CUDA support (optional, for acceleration)
  - RAM: Minimum 8GB, 16GB+ recommended for large designs

## Quick Install

=== "Using uv (Recommended)"

    [uv](https://github.com/astral-sh/uv) is a fast, modern Python package installer.

    ```bash
    # Install uv if not already installed
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Clone the repository
    git clone https://github.com/yao-baijian/fem.git
    cd fem

    # Create virtual environment and install dependencies
    uv sync

    # Install PyTorch (CPU version)
    uv pip install torch

    # Or install PyTorch with CUDA support
    uv pip install torch --index-url https://download.pytorch.org/whl/cu118

    # Install RapidWright
    uv pip install rapidwright

    # Optional: Install ML dependencies
    uv pip install scikit-learn joblib
    ```

=== "Using pip"

    Traditional installation using pip:

    ```bash
    # Clone the repository
    git clone https://github.com/yao-baijian/fem.git
    cd fem

    # Create virtual environment
    python -m venv .venv

    # Activate virtual environment
    source .venv/bin/activate  # On macOS/Linux
    # or
    .venv\Scripts\activate  # On Windows

    # Install package
    pip install -e .

    # Install PyTorch (CPU version)
    pip install torch

    # Or install PyTorch with CUDA support
    pip install torch --index-url https://download.pytorch.org/whl/cu118

    # Install RapidWright
    pip install rapidwright

    # Optional: Install ML dependencies
    pip install -e ".[ml]"
    ```

## Verify Installation

Run these commands to verify everything is installed correctly:

```python
# Test basic imports
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

import fem_placer
print(f"fem_placer imported successfully")

# Test RapidWright
try:
    from com.xilinx.rapidwright.design import Design
    print("RapidWright imported successfully")
except ImportError:
    print("RapidWright not available (optional)")

# List available modules
print(f"Available classes: {dir(fem_placer)}")
```

## Detailed Setup

### 1. Python Installation

If Python is not installed:

=== "macOS"

    ```bash
    # Using Homebrew
    brew install python@3.11
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3-pip
    ```

=== "Windows"

    Download from [python.org](https://www.python.org/downloads/) and run the installer.
    Make sure to check "Add Python to PATH" during installation.

### 2. Git Installation

=== "macOS"

    ```bash
    brew install git
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt install git
    ```

=== "Windows"

    Download from [git-scm.com](https://git-scm.com/download/win)

### 3. CUDA Setup (Optional, for GPU)

For GPU acceleration, install CUDA toolkit:

=== "Ubuntu/Debian"

    ```bash
    # Add NVIDIA repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update

    # Install CUDA toolkit
    sudo apt-get install cuda-11-8
    ```

=== "Windows"

    Download CUDA toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)

Verify CUDA installation:

```bash
nvidia-smi
nvcc --version
```

### 4. Virtual Environment

Always use a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Deactivate (when done)
deactivate
```

## Dependencies

### Core Dependencies

These are automatically installed:

- **numpy**: Numerical operations
- **scipy**: Scientific computing
- **matplotlib**: Visualization
- **pandas**: Data manipulation
- **rapidwright**: FPGA design handling
- **simulated-bifurcation**: Alternative solver

### Optional Dependencies

Install as needed:

```bash
# Machine learning features
pip install scikit-learn joblib

# Development tools
pip install pytest pytest-cov black ruff mypy

# Documentation
pip install mkdocs mkdocs-material
```

## Platform-Specific Notes

### macOS

If you encounter issues with architecture (Apple Silicon):

```bash
# Use Rosetta for x86_64 packages if needed
arch -x86_64 pip install <package>
```

### Linux

Ensure you have build tools:

```bash
sudo apt install build-essential python3-dev
```

### Windows

Some packages may require Visual C++ build tools:

- Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

## Docker Installation (Alternative)

Run in a container:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y git

# Clone repository
RUN git clone https://github.com/yao-baijian/fem.git

# Install package
WORKDIR /app/fem
RUN pip install -e .
RUN pip install torch rapidwright

CMD ["python"]
```

Build and run:

```bash
docker build -t fpga-placement-fem .
docker run -it --rm fpga-placement-fem
```

## Troubleshooting

### Issue: PyTorch not found

**Solution:**
```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: RapidWright import error

**Solution:**
RapidWright requires Java. Install JDK:

=== "macOS"
    ```bash
    brew install openjdk@11
    ```

=== "Ubuntu"
    ```bash
    sudo apt install openjdk-11-jdk
    ```

### Issue: CUDA not available

**Solution:**
Check NVIDIA driver:
```bash
nvidia-smi
```

If not working, reinstall CUDA toolkit and drivers.

### Issue: Permission denied

**Solution:**
Use user installation:
```bash
pip install --user -e .
```

### Issue: Package conflicts

**Solution:**
Use fresh virtual environment:
```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Next Steps

After installation:

1. [Run the Quick Start](quickstart.md)
2. [Read the User Guide](USER_GUIDE.md)
3. [Read the User Guide](USER_GUIDE.md)

## Getting Help

If you encounter issues:

- Check [GitHub Issues](https://github.com/yao-baijian/fem/issues)
- Check [Testing Guide](testing.md)
- Open a new issue with:
  - Python version: `python --version`
  - PyTorch version: `pip show torch`
  - Error messages and stack trace
