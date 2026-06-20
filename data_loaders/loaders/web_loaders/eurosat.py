from __future__ import annotations

import os
from typing import Any

import numpy as np
from torchvision import transforms, datasets
from torch.utils import data as torch_data
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake',
]


class EuroSATLoader(AbstractLoader):
    """Load the EuroSAT satellite land-use dataset.

    Real Sentinel-2 satellite image patches (64×64 RGB) covering 10 land-use
    and land-cover classes, flattened to 12288-dimensional vectors. In the
    default binary mode a specified minority class is labelled 1 and all
    remaining classes are labelled 0.

    Dataset stats: 27 000 samples, 12288 features per sample.
    Source: https://github.com/phelber/EuroSAT

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class in the train set by this factor.
    size : int, default=5000
        Number of samples to load (max 27 000). Kept modest by default to
        keep loading and dimensionality reduction light.
    minority_id : list[int] or None, default=None
        Class indices to treat as the minority class (class 1) in binary mode.
        Defaults to ``[3]`` (Highway).
    binary : bool, default=True
        If True, convert to binary classification (minority class vs. rest).
        If False, keep all 10 classes.
    classes_remove : list[int] or None, default=None
        Classes to remove entirely before creating the split.
    equal_test : bool, default=False
        If True, balance the test set classes.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    default_dim_reducer: str = 'TSNE'

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 minority_reduce_scaler: int | None = None,
                 size: int = 5000,
                 minority_id: list[int] | None = None,
                 binary: bool = True,
                 classes_remove: list[int] | None = None,
                 equal_test: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='EuroSAT',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id if minority_id is not None else [3]
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.equal_test = equal_test

    def load_data(self) -> DataDict:
        """Download (if needed) and load EuroSAT images.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(size, 12288)``), ``'y'``,
            ``'description'``, and ``'label_names'``.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(this_dir, '..', 'datasets', 'EuroSAT')
        eurosat_dataset = datasets.EuroSAT(download_dir, download=True,
                                           transform=transforms.ToTensor())
        size = min(self.size, len(eurosat_dataset))
        # shuffle indices so a size-limited load still spans every class
        rng = np.random.default_rng(42)
        order = rng.permutation(len(eurosat_dataset))[:size]
        subset = torch_data.Subset(eurosat_dataset, order.tolist())
        loader = torch_data.DataLoader(subset, batch_size=256, shuffle=False,
                                       drop_last=False)

        X = np.empty([size, 64 * 64 * 3], dtype=np.float32)
        y = np.zeros(size, dtype=np.int64)
        counter = 0
        for d, t in loader:
            n = d.shape[0]
            X[counter:counter + n] = d.numpy().reshape(n, -1)
            y[counter:counter + n] = t.numpy()
            counter += n

        for cls in self.classes_remove:
            mask = y != cls
            X, y = X[mask], y[mask]

        if self.binary:
            binary_y = np.zeros(len(y), dtype=np.int64)
            for id_ in self.minority_id:
                binary_y[y == id_] = 1
            y = binary_y
            minority_names = ', '.join(EUROSAT_CLASSES[i] for i in self.minority_id)
            label_names = ['Other classes', minority_names]
        else:
            label_names = list(EUROSAT_CLASSES)

        return {
            'X': X,
            'y': y,
            'description': 'EuroSAT: 64×64 RGB Sentinel-2 satellite patches across 10 land-use classes.',
            'label_names': label_names,
        }


if __name__ == '__main__':
    loader = EuroSATLoader()
    print(loader.get_info(long=True))
