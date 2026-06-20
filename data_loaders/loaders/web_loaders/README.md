# data_loaders.loaders.web_loaders

Datasets loaded from sklearn's built-in collection or fetched online.

## Overview

These loaders wrap sklearn toy datasets and online sources (UCI via
`ucimlrepo`, OpenML, torchvision image datasets, and the MedMNIST collection).
They all inherit from `AbstractLoader` and expose the standard API
(`get_X()`, `get_y()`, `get_train_test_split()`, `plot_dataset()`, etc.).

## Loaders

| Class | Samples | Features | Classes | Notes |
|---|---|---|---|---|
| `IrisLoader` | 150 | 4 | 3 | Flower species classification |
| `WineLoader` | 178 | 13 | 3 | Wine cultivar classification |
| `BreastCancerLoader` | 569 | 30 | 2 | Malignant / benign tumour |
| `HeartDiseaseLoader` | 303 | 13 | 2 | Presence of heart disease |
| `ParkinsonsLoader` | 195 | 22 | 2 | Voice measurements, imbalanced (~75% PD) |
| `IndianLiverLoader` | 583 | 10 | 2 | Liver disease, imbalanced |
| `ArrhythmiaLoader` | 452 | 279 | 2 / 13 | ECG; binary by default, multi-class via `binary=False` |
| `MnistLoader` | 70 000 | 784 | 10 | Handwritten digit images (marked slow) |
| `FashionMnistLoader` | 60 000 | 784 | 10 | Clothing images, MNIST drop-in (slow) |
| `SVHNLoader` | 73 257 | 3072 | 10 | Street-view house-number digits (slow) |
| `EuroSATLoader` | 27 000 | 12 288 | 10 | Satellite land-use patches (slow) |
| `Cifar10Loader` | 50 000 | 3072 | 10 | Object images (slow) |
| `Cifar100Loader` | 50 000 | 3072 | 100 | Fine-grained object images (slow) |
| `Cifar10NLoader` | 50 000 | 3072 | 10 | CIFAR-10 with human label noise (slow) |
| `MedMNISTLoader` subclasses | varies | 784 / 2352 | 2-9 | Biomedical images: Pneumonia/Breast/Derma/Blood/Path/OCT (slow) |

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
