# toy_datasets

Python package providing a unified interface for loading synthetic and real-world toy datasets for ML experimentation.

## Development Environment

Always use PDM (Python Dependency Manager) for environment and dependency management.

- `pdm install` - Install dependencies and create virtual environment
- `pdm install -G dev` - Install with dev dependencies
- `pdm add <package>` - Add a new dependency
- `pdm run <command>` - Run command in the PDM environment
- `pdm add -e path/to/package --dev` - Install a package in development mode

Config in `pyproject.toml` with `distribution = false` (application mode, not library).

## Architecture

- `data_loaders/main.py` - Central registry (`AVAILABLE_DATASETS`) and `get_dataset(name)` entry point
- `data_loaders/abstract_loader.py` - Base `AbstractLoader` class all loaders inherit from
- `data_loaders/loaders/synthetic_generators/` - XOR, Moons, Blobs, Circles, Gaussian, Madelon
- `data_loaders/loaders/web_loaders/` - Iris, Wine, MNIST, Heart Disease, Breast Cancer (from sklearn/online)
- `data_loaders/loaders/local_loaders/` - Breast cancer variants, diabetes, banknote, wheat seeds (from local CSV files in `data/`)
- `data_loaders/loaders/external_loaders/` - MIMIC-III/IV medical datasets (require special access)
- `data_loaders/utils/` - Normalization (`normalisation.py`), shuffling/seeding (`shuffling.py`), train/test splitting (`splitting.py`)
- `data_loaders/embeddings/` - Dimensionality reduction subpackage: `base.py` (ABC), `pca.py`, `tsne.py`, `umap.py`, `dim_reducer.py` (string-dispatch orchestrator)
- `data_loaders/plotting/` - Dataset visualisation (`visualisation.py`) and terminal rendering (`terminal_plots.py`)

## Key Patterns

- All loaders return dict format: `{'X': features, 'y': labels}`
- Loaders provide: `get_X()`, `get_y()`, `get_train_test_split()`, `plot_dataset()`
- Normalization uses MinMax scaling to [-1, 1] range
- Train/test splits support configurable ratio, shuffling, and class balance

## Public API

Each `__init__.py` defines `__all__` to explicitly export the public API.

**Main package (`from data_loaders import ...`):**
- `get_dataset` - Main entry point for loading datasets
- `AVAILABLE_DATASETS` - Registry dict of all loaders
- `AbstractLoader` - Base class for custom loaders
- `normaliser`, `proportional_split`, `proportional_downsample` - Utilities
- `dim_reducer` - Dimensionality reduction wrapper
- `get_available_dataset_list()`, `print_available_datasets()` - Helpers

**Submodules:**
- `data_loaders.loaders.synthetic_generators` - All generator classes
- `data_loaders.loaders.web_loaders` - All web-based loader classes
- `data_loaders.loaders.local_loaders` - All CSV-based loader classes

## Available Datasets (~25)

Synthetic: xor, moons, blobs, circles, gaussian, madelon
Classic: iris, wine, breast_cancer
Medical: diabetes, heart_disease, breast_cancer_* variants
Other: banknote, wheat_seeds, mnist

## Testing

Uses pytest for testing. Install dev dependencies first: `pdm install -G dev`

```bash
pdm run pytest                              # Run all tests
pdm run pytest -m "not slow"                # Skip slow tests (MNIST, t-SNE)
pdm run pytest --cov=data_loaders           # With coverage report
pdm run pytest -v                           # Verbose output
pdm run pytest tests/test_utils.py          # Run specific test file
pdm run pytest -k "test_shuffle"            # Run tests matching pattern
```

Test structure:
- `tests/conftest.py` - Shared fixtures (MockLoader, sample data)
- `tests/test_utils.py` - Utility functions (normaliser, shuffle, split)
- `tests/test_abstract_loader.py` - Base AbstractLoader class
- `tests/test_embeddings.py` - Dimensionality reduction (PCA, UMAP)
- `tests/test_main.py` - Registry and all dataset loaders
- `tests/test_loaders.py` - Individual loader category tests


## coding style
always wise clear and consise code.


## documentation
always document features in the readme and docstrings. Use type hints for clarity in the input arguments.