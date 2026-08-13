from __future__ import annotations

import os
from typing import Any

import numpy as np
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

FASHION_MNIST_CLASSES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
]


class FashionMnistLoader(AbstractLoader):
    """Load the Fashion-MNIST clothing-image dataset.

    A drop-in, more realistic replacement for MNIST: 28×28 greyscale photos of
    clothing items, flattened to 784-dimensional vectors. In the default binary
    mode a specified minority class is labelled 1 and all remaining classes are
    labelled 0.

    Dataset stats: up to 60 000 training samples, 784 features per sample.
    Source: https://github.com/zalandoresearch/fashion-mnist

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.1
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class in the train set by this factor.
    size : int, default=60000
        Number of samples to load from the training split (max 60 000).
    minority_id : list[int] or None, default=None
        Class indices to treat as the minority class (class 1) in binary mode.
        Defaults to ``[0]`` (T-shirt/top).
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
    is_image: bool = True
    image_shape: tuple[int, ...] = (28, 28)

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.1,
                 minority_reduce_scaler: int | None = None,
                 size: int = 60000,
                 minority_id: list[int] | None = None,
                 binary: bool = True,
                 classes_remove: list[int] | None = None,
                 equal_test: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='Fashion-MNIST',
                         short_description='Clothing item recognition — 28×28 greyscale, 10 categories (Zalando)',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id if minority_id is not None else [0]
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.equal_test = equal_test

    def load_data(self) -> DataDict:
        """Download (if needed) and load Fashion-MNIST training images.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(size, 784)``), ``'y'``,
            ``'description'``, and ``'label_names'``.
        """
        # torch and torchvision are needed only by the image loaders. Importing them
        # here rather than at module scope keeps `import data_loaders` - and every
        # tabular loader in this package - free of a ~3 GB deep-learning dependency.
        from torchvision import transforms, datasets
        from torch.utils import data as torch_data

        this_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(this_dir, '..', 'datasets', 'FashionMNIST')
        train_loader = torch_data.DataLoader(
            datasets.FashionMNIST(download_dir, train=True, download=True,
                                  transform=transforms.ToTensor()),
            batch_size=256, shuffle=False, drop_last=False)

        X = np.empty([self.size, 28 * 28], dtype=np.float32)
        y = np.zeros(self.size, dtype=np.int64)
        counter = 0
        for _, (d, t) in enumerate(train_loader):
            n = min(d.shape[0], self.size - counter)
            X[counter:counter + n] = d.numpy()[:n].reshape(n, -1)
            y[counter:counter + n] = t.numpy()[:n]
            counter += n
            if counter >= self.size:
                break

        for cls in self.classes_remove:
            mask = y != cls
            X, y = X[mask], y[mask]

        if self.binary:
            binary_y = np.zeros(len(y), dtype=np.int64)
            for id_ in self.minority_id:
                binary_y[y == id_] = 1
            y = binary_y
            minority_names = ', '.join(FASHION_MNIST_CLASSES[i] for i in self.minority_id)
            label_names = ['Other classes', minority_names]
        else:
            label_names = list(FASHION_MNIST_CLASSES)

        return {
            'X': X,
            'y': y,
            'description': 'Fashion-MNIST: 28×28 greyscale images of 10 clothing categories.',
            'label_names': label_names,
        }


if __name__ == '__main__':
    loader = FashionMnistLoader()
    print(loader.get_info(long=True))
