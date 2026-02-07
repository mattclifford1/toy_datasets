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

## Visualization

### Basic Plotting

```python
# Default behavior: creates figure and calls plt.show()
loader = data_loaders.get_dataset('Moons')
fig, ax = loader.plot_dataset()  # Returns (fig, ax) for further customization

# Train/test split visualization
fig, axes = loader.plot_train_test_split()  # Returns (fig, [ax1, ax2])
```

### Custom Axes Integration

Plot on your own matplotlib axes for full control:

```python
import matplotlib.pyplot as plt

# Single dataset on custom axes
fig, ax = plt.subplots(figsize=(8, 8))
loader.plot_dataset(ax=ax)  # Returns None when axes provided
ax.set_title("My Custom Title")
plt.show()

# Train/test split on custom axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
loader.plot_train_test_split(ax=(ax1, ax2))
fig.suptitle("Custom Comparison")
plt.show()
```

### Multiple Datasets in Grid

```python
# Compare multiple datasets
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
datasets = ['XOR', 'Moons', 'Circles', 'Blobs']

for ax, name in zip(axes.flat, datasets):
    data_loaders.get_dataset(name).plot_dataset(ax=ax)
    ax.set_title(name)

plt.tight_layout()
plt.show()
```

### Terminal Plotting

```python
# Render plots directly in terminal (supports sixel/kitty protocols)
loader.plot_dataset(terminal_plot=True)
loader.plot_train_test_split(terminal_plot=True)
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

## Project Structure

```
toy_datasets/
├── data_loaders/
│   ├── main.py              # Registry and get_dataset()
│   ├── abstract_loader.py   # Base class for all loaders
│   ├── utils.py             # Normalization, splitting utilities
│   ├── embeddings.py        # PCA, UMAP, t-SNE
│   ├── terminal_plots.py    # Terminal rendering
│   ├── synthetic_generators/  # XOR, Moons, Blobs, etc.
│   ├── web_loaders/           # Iris, Wine, MNIST, etc.
│   ├── local_loaders/         # CSV-based datasets
│   └── external_loaders/      # MIMIC (requires access)
├── data/                    # Local CSV datasets
├── tests/                   # pytest test suite
└── pyproject.toml           # PDM configuration
```

## License

MIT
