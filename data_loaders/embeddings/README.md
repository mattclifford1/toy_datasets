# data_loaders.embeddings

Dimensionality reduction wrappers for visualization and preprocessing.

## Overview

`DimReducer` is the main entry point. It is fitted on training data and
exposes a `transform` method for projecting any split into the reduced space.
If the input already has `<= num_dims` features, no reduction is applied.

## Classes

**`DimReducer(X_train, y_train, reducer, num_dims)`**
String-dispatch orchestrator. Accepted `reducer` values:
- `'PCA'` — principal component analysis
- `'kernelPCA'` — kernel PCA
- `'TSNE'` — t-SNE (fitted and transforms in one step)
- `'UMAP'` — UMAP unsupervised
- `'UMAP_supervised'` — UMAP with label guidance (requires `y_train`)

**`AbstractDimReducer`**
Abstract base class for all reducers. Implement `transform(X)`.

**`PCADimReducer`**, **`KernelPCADimReducer`**, **`TSNEDimReducer`**,
**`UMAPDimReducer`**, **`UMAPSupervisedDimReducer`**
Concrete reducer classes. Prefer `DimReducer` for normal use.

## Usage

```python
from data_loaders.embeddings import DimReducer

reducer = DimReducer(train['X'], reducer='PCA', num_dims=2)

X_train_2d = reducer.transform(train['X'])
X_test_2d  = reducer.transform(test['X'])

# Feature axis names are available for plot labels
print(reducer.feature_names)  # ['PCA 1', 'PCA 2']
```
