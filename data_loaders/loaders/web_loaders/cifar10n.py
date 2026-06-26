from __future__ import annotations

import os
import warnings
from typing import Any, Literal

import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils import data as torch_data
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from data_loaders.loaders.web_loaders.cifar10 import CIFAR10_CLASSES

_CIFAR10N_FILENAME = 'CIFAR-10_human.pt'
# Official CIFAR-10N human-annotated label file (Wei et al., ICLR 2022).
_CIFAR10N_URL = (
    'https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt'
)

LabelNoiseType = Literal[
    'aggre_label', 'random_label1', 'random_label2', 'random_label3', 'worst_label'
]


class Cifar10NLoader(AbstractLoader):
    """Load CIFAR-10N: CIFAR-10 images with real human annotation noise.

    Uses the same 32×32 RGB images as CIFAR-10 (flattened to 3072-dimensional
    vectors) but replaces the clean labels with one of five human-annotated
    noisy label sets from the CIFAR-10N paper (Wei et al., 2022).

    Both the CIFAR-10 images and the noisy label file (``CIFAR-10_human.pt``)
    are downloaded automatically on first use (the label file comes from the
    official CIFAR-10N release, https://github.com/UCSC-REAL/cifar-10-100n) and
    cached under ``data_loaders/loaders/datasets/``.

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
        Number of samples to load (max 50 000).
    minority_id : list[int] or None, default=None
        Class indices to treat as the minority class (class 1) in binary mode.
        Defaults to ``[0]`` (airplane).
    binary : bool, default=True
        If True, convert to binary classification (minority class vs. rest).
        If False, keep all 10 classes.
    classes_remove : list[int] or None, default=None
        Classes to remove entirely before creating the split.
    label_noise_type : str, default='aggre_label'
        Which noisy label set to use. One of:
        ``'aggre_label'``, ``'random_label1'``, ``'random_label2'``,
        ``'random_label3'``, ``'worst_label'``.
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
                 label_noise_type: LabelNoiseType = 'aggre_label',
                 equal_test: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='CIFAR-10N',
                         short_description='CIFAR-10 with noisy human labels for label-noise research',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id if minority_id is not None else [0]
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.label_noise_type = label_noise_type
        self.equal_test = equal_test

    def _load_noisy_labels(self, label_dir: str) -> dict:
        """Load the CIFAR-10N human label file, downloading it on first use."""
        label_path = os.path.join(label_dir, _CIFAR10N_FILENAME)
        if not os.path.exists(label_path):
            from torchvision.datasets.utils import download_url

            download_url(_CIFAR10N_URL, root=label_dir,
                         filename=_CIFAR10N_FILENAME)
        # weights_only=False is required because the official CIFAR-10N file
        # pickles numpy arrays; it is safe here as the file is fetched from the
        # pinned official release URL (_CIFAR10N_URL).
        return torch.load(label_path, weights_only=False)

    def load_data(self) -> DataDict:
        """Load CIFAR-10N training images with human-annotated noisy labels.

        Both the CIFAR-10 images and the noisy label file are downloaded
        automatically on first use.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(size, 3072)``), ``'y'``,
            ``'description'``, and ``'label_names'``.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cifar10_dir = os.path.join(this_dir, '..', 'datasets', 'CIFAR-10')
        label_dir = os.path.join(this_dir, '..', 'datasets', 'CIFAR-10N')
        os.makedirs(label_dir, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.exceptions.VisibleDeprecationWarning)
            cifar_dataset = datasets.CIFAR10(cifar10_dir, train=True, download=True,
                                             transform=transforms.ToTensor())
            train_loader = torch_data.DataLoader(
                cifar_dataset, batch_size=256, shuffle=False, drop_last=False)

            X = np.empty([self.size, 32 * 32 * 3], dtype=np.float32)
            counter = 0
            for _, (d, _) in enumerate(train_loader):
                d_np = d.numpy()
                for i in range(d.shape[0]):
                    X[counter, :] = d_np[i].reshape(-1)
                    counter += 1
                    if counter == self.size:
                        break
                if counter == self.size:
                    break

        noise_data = self._load_noisy_labels(label_dir)
        noisy_labels = noise_data[self.label_noise_type]
        if hasattr(noisy_labels, 'numpy'):
            noisy_labels = noisy_labels.numpy()
        y = noisy_labels[:self.size].astype(np.int64)

        for cls in self.classes_remove:
            mask = y != cls
            X, y = X[mask], y[mask]

        if self.binary:
            binary_y = np.zeros(len(y), dtype=np.int64)
            for id_ in self.minority_id:
                binary_y[y == id_] = 1
            y = binary_y
            minority_names = ', '.join(CIFAR10_CLASSES[i] for i in self.minority_id)
            label_names = ['Other classes', minority_names]
        else:
            label_names = list(CIFAR10_CLASSES)

        return {
            'X': X,
            'y': y,
            'description': (
                f'CIFAR-10N: CIFAR-10 images with real human annotation noise '
                f'(label set: {self.label_noise_type}).'
            ),
            'label_names': label_names,
        }


if __name__ == '__main__':
    loader = Cifar10NLoader()
    print(loader.get_info(long=True))
