# toy_datasets

Python package providing a unified interface for loading synthetic and real-world toy datasets for ML experimentation.

## Development Environment

Always use uv for environment and dependency management.

- `uv sync` - Install dependencies and create virtual environment
- `uv sync --group dev` - Install with dev dependencies
- `uv add <package>` - Add a new dependency
- `uv run <command>` - Run command in the uv environment
- `uv add --editable path/to/package --group dev` - Install a package in development mode

Config in `pyproject.toml`.

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
Image: mnist, cifar-10, cifar-100, cifar-10n
Other: banknote, wheat_seeds

## Testing

Uses pytest for testing. Install dev dependencies first: `uv sync --group dev`

```bash
uv run pytest                              # Run all tests
uv run pytest -m "not slow"                # Skip slow tests (MNIST, t-SNE)
uv run pytest --cov=data_loaders           # With coverage report
uv run pytest -v                           # Verbose output
uv run pytest tests/test_utils.py          # Run specific test file
uv run pytest -k "test_shuffle"            # Run tests matching pattern
```

Test structure:
- `tests/conftest.py` - Shared fixtures (MockLoader, sample data)
- `tests/test_utils.py` - Utility functions (normaliser, shuffle, split)
- `tests/test_abstract_loader.py` - Base AbstractLoader class
- `tests/test_embeddings.py` - Dimensionality reduction (PCA, UMAP)
- `tests/test_main.py` - Registry and all dataset loaders
- `tests/test_loaders.py` - Individual loader category tests


## README figures

The README contains two auto-generated collapsible sections. Keep them in sync whenever relevant code changes.

**When to regenerate:**
- New dataset added to `AVAILABLE_DATASETS` in `data_loaders/main.py` → run the gallery script and add the dataset to `DATASET_GROUPS` in `scripts/generate_dataset_figures.py`
- New loader option added to `AbstractLoader` → add a figure function to `scripts/generate_options_figures.py` and run it

**Scripts:**
```bash
uv run python scripts/generate_dataset_figures.py   # regenerates assets/figures/*.png + GALLERY_SECTION.md
uv run python scripts/generate_options_figures.py   # regenerates assets/figures/options/*.png + OPTIONS_SECTION.md
```

**After running**, paste the generated `GALLERY_SECTION.md` or `OPTIONS_SECTION.md` content into the corresponding `<details>` block in `README.md`:
- Gallery section: replaces content between `<summary><strong>All datasets — figures &amp; stats</strong></summary>` and its closing `</details>`
- Options section: replaces content between `<summary><strong>Loader options — visual demos</strong></summary>` and its closing `</details>`


## coding style
always wise clear and consise code.


## documentation
always document features in the readme and docstrings. Use type hints for clarity in the input arguments.