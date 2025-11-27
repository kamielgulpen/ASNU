# Installation Guide

## Standard Installation

Install ASNU using pip:

```bash
pip install asnu
```

## Installation from Source

### Prerequisites

- Python >= 3.8
- pip

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/asnu.git
cd asnu
```

2. Install the package:
```bash
pip install .
```

3. Or install in editable mode for development:
```bash
pip install -e .
```

## Optional Dependencies

### For Analysis and Visualization

Install additional packages for network analysis:

```bash
pip install asnu[analysis]
```

This includes:
- matplotlib (visualization)
- seaborn (statistical visualization)
- SALib (sensitivity analysis)
- mlflow (experiment tracking)

### For Development

Install development dependencies:

```bash
pip install asnu[dev]
```

This includes:
- pytest (testing framework)
- pytest-cov (coverage reporting)
- black (code formatter)
- flake8 (linter)

### All Dependencies

Install everything:

```bash
pip install asnu[dev,analysis]
```

## Verifying Installation

Test your installation:

```python
import asnu
print(asnu.__version__)
```

Or run the example scripts:

```bash
cd examples
python basic_example.py
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install numpy pandas networkx scipy openpyxl
```

### Version Conflicts

If you have version conflicts, try creating a clean virtual environment:

```bash
python -m venv asnu_env
source asnu_env/bin/activate  # On Windows: asnu_env\Scripts\activate
pip install asnu
```

## System Requirements

- **RAM**: Minimum 4GB, recommended 8GB+ for large networks
- **Storage**: Varies with network size (networks saved as .gpickle files)
- **OS**: Windows, macOS, or Linux

## Uninstallation

To remove ASNU:

```bash
pip uninstall asnu
```
