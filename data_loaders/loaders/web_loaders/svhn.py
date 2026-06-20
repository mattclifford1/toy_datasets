from __future__ import annotations

import os
from typing import Any

import numpy as np
from torchvision import transforms, datasets
from torch.utils import data as torch_data
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

SVHN_CLASSES = [str(i) for i in range(10)]


class SVHNLoader(AbstractLoader):
    """Load the Street View House Numbers (SVHN) dataset.

    Real-world 32×32 RGB images of digits cropped from house-number plates,
    flattened to 3072-dimensional vectors. A more realistic, naturally slightly
    imbalanced counterpart to MNIST. In the default binary mode a specified
    minority digit is labelled 1 and all remaining digits are labelled 0.

    Dataset stats: up to 73 257 training samples, 3072 features per sample.
    Source: http://ufldl.stanford.edu/housenumbers/

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.1
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class in the train set by this factor.
    size : int, default=73257
        Number of samples to load from the training split (max 73 257).
    minority_id : list[int] or None, default=None
        Digit label(s) to treat as the minority class (class 1) in binary mode.
        Defaults to ``[0]`` (digit 0).
    binary : bool, default=True
        If True, convert to binary classification (minority digit vs. rest).
        If False, keep all 10 digit classes.
    classes_remove : list[int] or None, default=None
        Digit classes to remove entirely before creating the split.
    equal_test : bool, default=False
        If True, balance the test set classes.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    default_dim_reducer: str = 'TSNE'

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.1,
                 minority_reduce_scaler: int | None = None,
                 size: int = 73257,
                 minority_id: list[int] | None = None,
                 binary: bool = True,
                 classes_remove: list[int] | None = None,
                 equal_test: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='SVHN',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id if minority_id is not None else [0]
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.equal_test = equal_test

    def load_data(self) -> DataDict:
        """Download (if needed) and load SVHN training images.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(size, 3072)``), ``'y'``,
            ``'description'``, and ``'label_names'``.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(this_dir, '..', 'datasets', 'SVHN')
        svhn_dataset = datasets.SVHN(download_dir, split='train', download=True,
                                     transform=transforms.ToTensor())
        size = min(self.size, len(svhn_dataset))
        train_loader = torch_data.DataLoader(
            svhn_dataset, batch_size=256, shuffle=False, drop_last=False)

        X = np.empty([size, 32 * 32 * 3], dtype=np.float32)
        y = np.zeros(size, dtype=np.int64)
        counter = 0
        for _, (d, t) in enumerate(train_loader):
            n = min(d.shape[0], size - counter)
            X[counter:counter + n] = d.numpy()[:n].reshape(n, -1)
            y[counter:counter + n] = t.numpy()[:n]
            counter += n
            if counter >= size:
                break

        for cls in self.classes_remove:
            mask = y != cls
            X, y = X[mask], y[mask]

        if self.binary:
            binary_y = np.zeros(len(y), dtype=np.int64)
            for id_ in self.minority_id:
                binary_y[y == id_] = 1
            y = binary_y
            minority_names = ', '.join(SVHN_CLASSES[i] for i in self.minority_id)
            label_names = ['Other digits', minority_names]
        else:
            label_names = list(SVHN_CLASSES)

        return {
            'X': X,
            'y': y,
            'description': 'SVHN: 32×32 RGB real-world images of house-number digits (0-9).',
            'label_names': label_names,
        }


if __name__ == '__main__':
    loader = SVHNLoader()
    print(loader.get_info(long=True))
