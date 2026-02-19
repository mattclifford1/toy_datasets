# data_loaders.loaders.web_loaders

Datasets loaded from sklearn's built-in collection or fetched online.

## Overview

These loaders wrap sklearn toy datasets and one online source (Heart Disease).
They all inherit from `AbstractLoader` and expose the standard API
(`get_X()`, `get_y()`, `get_train_test_split()`, `plot_dataset()`, etc.).

## Loaders

| Class | Samples | Features | Classes | Notes |
|---|---|---|---|---|
| `IrisLoader` | 150 | 4 | 3 | Flower species classification |
| `WineLoader` | 178 | 13 | 3 | Wine cultivar classification |
| `BreastCancerLoader` | 569 | 30 | 2 | Malignant / benign tumour |
| `HeartDiseaseLoader` | 303 | 13 | 2 | Presence of heart disease |
| `MnistLoader` | 70 000 | 784 | 10 | Handwritten digit images (marked slow) |

## Usage

```python
from data_loaders.loaders.web_loaders import IrisLoader

loader = IrisLoader(scale=True, train_size=0.8)

X = loader.get_X()
y = loader.get_y()
train, test = loader.get_train_test_split()

print(loader.get_feature_names())
print(loader.get_label_names())
```
