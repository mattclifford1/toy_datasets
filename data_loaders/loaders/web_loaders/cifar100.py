from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from torchvision import transforms, datasets
from torch.utils import data as torch_data
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm',
]


class Cifar100Loader(AbstractLoader):
    """Load the CIFAR-100 image dataset.

    Images (32×32 RGB pixels) are flattened to 3072-dimensional vectors.
    In the default binary mode a specified minority class is labelled 1
    and all remaining classes are labelled 0.

    Dataset stats: up to 50 000 training samples, 3072 features per sample.

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.1
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class in the train set by this factor.
    size : int, default=50000
        Number of samples to load from the training split (max 50 000).
    minority_id : list[int] or None, default=None
        Fine-label indices (0–99) to treat as the minority class (class 1) in
        binary mode.  Defaults to ``[0]`` (apple).
    binary : bool, default=True
        If True, convert to binary classification (minority class vs. rest).
        If False, keep all 100 fine classes.
    classes_remove : list[int] or None, default=None
        Fine-label classes to remove entirely before creating the split.
    equal_test : bool, default=False
        If True, balance the test set classes.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    default_dim_reducer: str = 'TSNE'
    is_image: bool = True
    image_shape: tuple[int, ...] = (3, 32, 32)
    channels_first: bool = True

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.1,
                 minority_reduce_scaler: int | None = None,
                 size: int = 50000,
                 minority_id: list[int] | None = None,
                 binary: bool = True,
                 classes_remove: list[int] | None = None,
                 equal_test: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='CIFAR-100',
                         short_description='Fine-grained object recognition — 32×32 RGB, 100 classes',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id if minority_id is not None else [0]
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.equal_test = equal_test

    def load_data(self) -> DataDict:
        """Download (if needed) and load CIFAR-100 training images.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(size, 3072)``), ``'y'``,
            ``'description'``, and ``'label_names'``.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(this_dir, '..', 'datasets', 'CIFAR-100')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.exceptions.VisibleDeprecationWarning)
            cifar_dataset = datasets.CIFAR100(download_dir, train=True, download=True,
                                              transform=transforms.ToTensor())
            train_loader = torch_data.DataLoader(
                cifar_dataset, batch_size=256, shuffle=False, drop_last=False)

            X = np.empty([self.size, 32 * 32 * 3], dtype=np.float32)
            y = np.zeros(self.size, dtype=np.int64)
            counter = 0
            for _, (d, t) in enumerate(train_loader):
                d_np = d.numpy()
                for i in range(d.shape[0]):
                    X[counter, :] = d_np[i].reshape(-1)
                    y[counter] = t[i]
                    counter += 1
                    if counter == self.size:
                        break
                if counter == self.size:
                    break

        for cls in self.classes_remove:
            mask = y != cls
            X, y = X[mask], y[mask]

        if self.binary:
            binary_y = np.zeros(len(y), dtype=np.int64)
            for id_ in self.minority_id:
                binary_y[y == id_] = 1
            y = binary_y
            minority_names = ', '.join(CIFAR100_CLASSES[i] for i in self.minority_id)
            label_names = ['Other classes', minority_names]
        else:
            label_names = list(CIFAR100_CLASSES)

        return {
            'X': X,
            'y': y,
            'description': 'CIFAR-100: 32×32 RGB images across 100 fine-grained object categories.',
            'label_names': label_names,
        }


if __name__ == '__main__':
    loader = Cifar100Loader()
    print(loader.get_info(long=True))
