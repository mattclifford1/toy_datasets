# data_loaders.loaders.synthetic_generators

Procedural generators for synthetic classification datasets.

## Overview

Each generator is an `AbstractLoader` subclass that produces data on the fly
using sklearn or numpy. All generators accept a `num_samples` argument and
forward the standard `AbstractLoader` options (`shuffle`, `train_size`,
`scale`, `dim_reducer`, etc.) via `**kwargs`.

## Generators

| Class | Description |
|---|---|
| `XORGenerator` | XOR layout — four Gaussian clusters, non-linearly separable |
| `MoonsGenerator` | Two interlocking half-circle crescents |
| `BlobsGenerator` | Gaussian blob clusters |
| `CirclesGenerator` | Concentric circles |
| `GaussianGenerator` | Two Gaussian distributions |
| `SklearnNormalGenerator` | Wrapper around `sklearn.datasets.make_classification` |

### Constructor arguments

All generators accept:
- `num_samples : int | list[int]` — total samples, or `[n_class0, n_class1]`
  for per-class counts (where supported).
- `shuffle : bool` — shuffle after generation (default `True`).
- `**kwargs` — forwarded to `AbstractLoader` (e.g. `train_size`, `scale`).

`MoonsGenerator` additionally accepts `moons_noise` (Gaussian noise std,
default `0.2`).

## Usage

```python
from data_loaders.loaders.synthetic_generators import MoonsGenerator

loader = MoonsGenerator(num_samples=500, moons_noise=0.1, scale=True)

train, test = loader.get_train_test_split()
loader.plot_dataset()
```
