# toy_datasets

Python package providing a unified interface for loading synthetic and real-world toy datasets for ML experimentation.

## Features

- Unified API for 20+ datasets via `get_dataset(name)`
- Consistent dict format: `{'X': features, 'y': labels}`
- Built-in train/test splitting with class balance preservation
- MinMax normalization to [-1, 1] range
- Dimensionality reduction (PCA, UMAP, t-SNE)
- Terminal-based plotting (sixel/kitty protocols)

## Installation

Requires Python 3.11+. Uses PDM for dependency management.

```bash
# Clone and install
git clone https://github.com/your-username/toy_datasets.git
cd toy_datasets
pdm install

# Or install with dev dependencies
pdm install -G dev
```

## Quick Start

```python
import data_loaders

# Load a dataset
loader = data_loaders.get_dataset('Iris')

# Get features and labels
X = loader.get_X()
y = loader.get_y()

# Get train/test split (preserves class proportions)
train, test = loader.get_train_test_split()

# With options
loader = data_loaders.get_dataset(
    'Moons',
    scale=True,           # Normalize to [-1, 1]
    train_size=0.8,       # 80% train, 20% test
    dim_reducer='PCA',    # Apply PCA
    reduce_to_dim=2       # Reduce to 2 dimensions
)
```

## Available Datasets

**Synthetic:**
- `XOR`, `Moons`, `Blobs`, `Circles`, `Gaussian`, `Sklearn Normal`

**Classic (sklearn):**
- `Iris`, `Wine`, `Breast Cancer`

**Medical:**
- `Diabetes Pima Indian`, `Heart Disease`, `Breast Cancer Wisconsin`
- `Habermans Breast Cancer`, `Chronic Kidney Disease`, `Hepatitis`

**Other:**
- `Banknote Authentication`, `Wheat Seeds`, `Ionosphere`
- `Sonar Rocks vs Mines`, `Abalone Gender`, `MNIST`
- `Costcla Credit Scoring Kaggle 2011`, `Costcla Credit Scoring PAKDD 2009`
- `Costcla Direct Marketing`

List all available datasets:
```python
from data_loaders.main import AVAILABLE_DATASETS
print(list(AVAILABLE_DATASETS.keys()))
```

## Loader API

All loaders inherit from `AbstractLoader` and provide:

```python
loader.get_X()                  # Feature array (numpy)
loader.get_y()                  # Label array (numpy)
loader.get_train_test_split()   # Returns (train_dict, test_dict)
loader.get_description()        # Dataset description
loader.get_feature_names()      # List of feature names
loader.get_label_names()        # List of class names
loader.get_info()               # Full dataset info string
loader.plot_dataset()           # Visualize the dataset
loader.plot_train_test_split()  # Visualize train/test split
```

## Common Options

```python
loader = data_loaders.get_dataset(
    'Iris',
    shuffle=True,            # Shuffle data (default: True)
    set_seed=42,             # Random seed for reproducibility
    train_size=0.5,          # Train set proportion (default: 0.5)
                             # The rest is used for the test set
    minority_reduce_scaler=2,# Reduce minority class in train set to 1/2 
                             # (default: None, no reduction)
    minority_reduce_scaler_test=2,# Reduce minority class in test set to 1/2 
                             # (default: None, no reduction)    
    equal_test=False,        # Whether to make the test set 
                             # perfectly balanced (overrides 
                             # minority_reduce_scaler_test)                   
    scale=True,              # Apply MinMax scaling
    percent_of_data=50,      # Use only 50% of data
    equal_test=True,         # Balance test set classes
    dim_reducer='PCA',       # 'PCA', 'UMAP', 'TSNE', 'kernelPCA'
    reduce_to_dim=2,         # Target dimensions
)
```

## Testing

```bash
pdm run pytest                     # Run all tests
pdm run pytest -m "not slow"       # Skip slow tests (MNIST, t-SNE)
pdm run pytest --cov=data_loaders  # With coverage report
```

## Sub-packages

Each sub-package has its own README with full details.

| Sub-package | Description |
|---|---|
| [`data_loaders/utils/`](data_loaders/utils/README.md) | Normalization, shuffling, seeding, train/test splitting |
| [`data_loaders/resampling/`](data_loaders/resampling/README.md) | Upsampling, SMOTE, and downsampling for class imbalance |
| [`data_loaders/embeddings/`](data_loaders/embeddings/README.md) | PCA, kernel PCA, t-SNE, UMAP dimensionality reduction |
| [`data_loaders/plotting/`](data_loaders/plotting/README.md) | Dataset visualisation and terminal rendering |
| [`data_loaders/loaders/synthetic_generators/`](data_loaders/loaders/synthetic_generators/README.md) | XOR, Moons, Blobs, Circles, Gaussian generators |
| [`data_loaders/loaders/web_loaders/`](data_loaders/loaders/web_loaders/README.md) | Iris, Wine, Breast Cancer, Heart Disease, MNIST |
| [`data_loaders/loaders/local_loaders/`](data_loaders/loaders/local_loaders/readme.md) | CSV-backed loaders (diabetes, banknote, costcla, etc.) |
| [`data_loaders/loaders/external_loaders/`](data_loaders/loaders/external_loaders/README.md) | MIMIC-III/IV medical datasets (require special access) |

## Project Structure

```
toy_datasets/
├── data_loaders/
│   ├── main.py                   # Registry and get_dataset()
│   ├── utils/                    # Normalization, splitting utilities
│   ├── resampling/               # Upsampling and downsampling
│   ├── embeddings/               # PCA, UMAP, t-SNE wrappers
│   ├── plotting/                 # Visualisation and terminal rendering
│   └── loaders/
│       ├── abstract_loader.py    # Base class for all loaders
│       ├── synthetic_generators/ # XOR, Moons, Blobs, etc.
│       ├── web_loaders/          # Iris, Wine, MNIST, etc.
│       ├── local_loaders/        # CSV-based datasets
│       └── external_loaders/     # MIMIC (requires access)
├── data_loaders/datasets/        # Bundled CSV files
├── tests/                        # pytest test suite
└── pyproject.toml                # PDM configuration
```

## License

MIT
