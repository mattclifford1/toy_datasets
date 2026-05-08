# data_loaders.plotting

Dataset visualisation and terminal rendering utilities.

## Overview

`plot_dataset` produces a 2D scatter plot of any dataset, applying
dimensionality reduction automatically when the data has more than two
features. The terminal rendering helpers let you display plots directly in
your terminal without a GUI.

## Visual Examples

**Single low-dimensional dataset** — raw 2D scatter, no reduction needed:

![Moons dataset](../../assets/figures/moons.png)

**High-dimensional dataset** — 13 features projected to 2D via PCA automatically:

![Wine dataset (PCA)](../../assets/figures/wine.png)

**Train/test split view** — `plot_train_test_split` renders both sets side-by-side with
consistent axis scaling:

![Train/test split](../../assets/figures/options/train_test_split.png)

## Functions and Classes

**`plot_dataset(X, y, X_test, y_test, dataset_name, label_names, terminal_plot, dim_reducer_method, ax)`**
Plot a dataset in 2D. Supports:
- Train-only or train/test side-by-side layout.
- Custom `ax` (single `Axes` or a 2-tuple for split view).
- Dimensionality reduction methods: `'TSNE'` (default), `'PCA'`, `'UMAP'`.
- `terminal_plot=True` to render in the terminal (requires `ax=None`).

Returns `(fig, ax)` or `(fig, [ax1, ax2])` when no axes are provided;
returns `None` when custom axes are given (caller controls display).

**`terminal_show()`**
Render all open matplotlib figures in the terminal using default settings,
then restore the original `plt.show()`.

**`enable_terminal_show(mode, width, height, clear, close, dpi, margin_cols)`**
Monkey-patch `plt.show()` to render in the terminal. Returns a
`TerminalPlotter` instance; call `.disable()` to restore normal behaviour.
Can also be used as a context manager.

**`TerminalPlotter`**
Class-based terminal renderer. Rendering modes:
- `'auto'` — iTerm2 inline PNG when available, otherwise text scatter.
- `'text'` — character-grid scatter plot with axis labels.
- `'ascii'` — ASCII-art rasterisation (requires `pillow`).
- `'iterm2'` — inline PNG via OSC 1337 protocol (iTerm2 only).

## Usage

```python
import matplotlib.pyplot as plt
from data_loaders.plotting import plot_dataset, enable_terminal_show

# Basic plot (opens a window)
loader = data_loaders.get_dataset('Moons')
train, test = loader.get_train_test_split()
plot_dataset(train['X'], train['y'], X_test=test['X'], y_test=test['y'],
             dataset_name='Moons', dim_reducer_method='PCA')

# Embed in a larger figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_dataset(train['X'], train['y'], X_test=test['X'], y_test=test['y'], ax=axes)
plt.show()

# Render in terminal
with enable_terminal_show():
    plot_dataset(train['X'], train['y'], terminal_plot=True)
```
