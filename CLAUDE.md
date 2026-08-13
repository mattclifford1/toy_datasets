# toy_datasets

Python package providing a unified interface for loading synthetic and real-world toy datasets for ML experimentation.

## Design Philosophy

We try to keep this package in line with the "deep-module" philosophy: a simple,
shallow interface backed by a powerful, deep implementation. The common case
should be trivial to get up and running without getting bogged down in details
or arguments — `get_dataset(name)` then `get_X()` / `get_train_test_split()` /
`plot_dataset()` — but we still allow for deep customisation when the user wants
it (splitting, class balancing, scaling, dimensionality reduction, post-process
hooks, etc.).

Consistency across the datasets is key. Practical rules that follow from this:

- **Push shared work down into `AbstractLoader`, not up into every loader.** A
  leaf loader should ideally only implement `load_data()` and pass a handful of
  sensible defaults to `super().__init__()`. Splitting, shuffling, scaling,
  dimensionality reduction, info/plotting, and reading packaged dataset files
  all live in the base class so individual loaders stay lean and identical in
  shape.
- **Every loader returns the same dict contract:** `{'X', 'y'}` required, plus
  optional `'feature_names'`, `'label_names'`, `'description'` (and
  `'cost_matrix'` for cost-sensitive datasets). Don't invent per-loader shapes.
- **No copy-pasted infrastructure in leaf loaders** (path building, file
  reading, re-shuffling, normalisation). If two loaders need the same helper,
  it belongs in the base class or a shared util — not duplicated.
- **Defaults over arguments.** Prefer good per-dataset defaults set in
  `__init__` over forcing the caller to pass options; expose the knob via
  `**kwargs` so power users can still override it.

## Development Environment

Always use uv for environment and dependency management.

- `uv sync` - Install the base dependencies and create the virtual environment
- `uv sync --all-extras --group dev` - what development actually needs; the base install
  deliberately omits torch, torchvision, medmnist, umap-learn and openTSNE
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
- `data_loaders/utils/` - Normalization (`normalisation.py`), shuffling/seeding (`shuffling.py`), train/test splitting (`splitting.py`), label remapping (`labels.py`)
- `data_loaders/embeddings/` - Dimensionality reduction subpackage: `base.py` (ABC), `pca.py`, `tsne.py`, `umap.py`, `dim_reducer.py` (string-dispatch orchestrator)
- `data_loaders/plotting/` - Dataset visualisation (`visualisation.py`) and terminal rendering (`terminal_plots.py`)

## Optional dependencies

The base install is deliberately lean: `numpy`, `scipy`, `scikit-learn`, `pandas`,
`ucimlrepo`, `tqdm`, `matplotlib`, `imbalanced-learn`, `pillow`. Everything heavier lives
in an extra --- `image` (torch, torchvision), `medmnist`, `embeddings` (umap-learn,
openTSNE). Together they weigh several GB, mostly CUDA, so a numerics project that only
wants the tabular datasets should not have to install them.

**This is load-bearing, not cosmetic. Every optional dependency must be imported inside
the function or method that uses it, never at module scope** --- including in a package
`__init__.py`, which runs whenever any of its submodules is imported. `main.py` builds
`AVAILABLE_DATASETS` from lazy factories precisely so that requesting one tabular dataset
does not import every loader; a module-level `from torchvision import ...` anywhere under
`loaders/web_loaders/` silently defeats that for the whole package. See
`embeddings/umap.py` and `web_loaders/mnist.py` for the pattern.

## Key Patterns

- All loaders return dict format: `{'X': features, 'y': labels}`
- Loaders provide: `get_X()`, `get_y()`, `get_train_test_split()`, `plot_dataset()`
- Normalization uses MinMax scaling to [-1, 1] range
- Train/test splits support configurable ratio, shuffling, and class balance
- Loaders implement only `load_data()`; shuffling is done by `AbstractLoader.get_data_dict()` (don't re-shuffle inside `load_data()`)
- Local loaders read bundled files via `self.local_dataset_path(name)` / `self.local_dataset_description(name)` — never rebuild `os.path` strings
- Remap raw labels (string or numeric) to integer classes with `binarise_labels(y, mapping)` from `data_loaders.utils` — handles merges and swaps safely; don't do in-place `y[y==k]=v`
- Image loaders set class attributes `is_image = True` plus `image_shape` (per-sample reshape) and `channels_first` (True for torchvision `C,H,W` storage, False for `H,W[,C]`). The base class then provides `as_image()`, `sample_images_per_class()` and `plot_class_samples()` for free, and the gallery script adds example-image previews automatically — don't hand-roll image reshaping in leaf loaders

## Public API

Each `__init__.py` defines `__all__` to explicitly export the public API.

**Main package (`from data_loaders import ...`):**
- `get_dataset` - Main entry point for loading datasets
- `AVAILABLE_DATASETS` - Registry dict of all loaders
- `AbstractLoader` - Base class for custom loaders
- `Normaliser`, `proportional_split`, `proportional_downsample`, `binarise_labels` - Utilities
- `dim_reducer` - Dimensionality reduction wrapper
- `get_available_dataset_list()`, `print_available_datasets()` - Helpers

**Submodules:**
- `data_loaders.loaders.synthetic_generators` - All generator classes
- `data_loaders.loaders.web_loaders` - All web-based loader classes
- `data_loaders.loaders.local_loaders` - All CSV-based loader classes

## Available Datasets (~35)

Synthetic: xor, moons, blobs, circles, gaussian, madelon
Classic: iris, wine, breast_cancer
Medical: diabetes, heart_disease, heart_failure, breast_cancer_* variants,
  thyroid_sick, stroke, framingham, thoracic_surgery, spectf, mammographic_mass,
  hcc_survival, zalizadeh_sani (many binary + imbalanced)
Image: mnist, cifar-10, cifar-100, cifar-10n
Other: banknote, wheat_seeds

## Testing

Uses pytest for testing. Install first with `uv sync --all-extras --group dev` --- the
image and embedding tests need the optional extras, and without them they fail rather
than skip.

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

The README contains three auto-generated collapsible sections. Keep them in sync whenever relevant code changes.

**When to regenerate:**
- New dataset added to `AVAILABLE_DATASETS` in `data_loaders/main.py` → run the gallery script and add the dataset to `DATASET_GROUPS` in `scripts/generate_dataset_figures.py`; also add it to `DATASET_GROUPS` in `scripts/generate_classifier_benchmark_figures.py` and re-run that script
- New loader option added to `AbstractLoader` → add a figure function to `scripts/generate_options_figures.py` and run it

**Scripts:**
```bash
uv run python scripts/generate_dataset_figures.py              # regenerates assets/figures/*.png + GALLERY_SECTION.md
uv run python scripts/generate_options_figures.py              # regenerates assets/figures/options/*.png + OPTIONS_SECTION.md
uv run python scripts/generate_classifier_benchmark_figures.py # regenerates assets/figures/benchmark_*.png + BENCHMARK_SECTION.md
```

**After running**, paste the generated markdown file content into the corresponding `<details>` block in `README.md`:
- Gallery section: replaces content between `<summary><strong>All datasets — figures &amp; stats</strong></summary>` and its closing `</details>`
- Options section: replaces content between `<summary><strong>Loader options — visual demos</strong></summary>` and its closing `</details>`
- Benchmark section: replaces content between `<summary><strong>Classifier benchmarks</strong></summary>` and its closing `</details>`


## coding style
always wise clear and consise code.


## documentation
always document features in the readme and docstrings. Use type hints for clarity in the input arguments.