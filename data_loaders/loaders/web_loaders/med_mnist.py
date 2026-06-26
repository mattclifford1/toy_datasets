from __future__ import annotations

import os
from typing import Any

import numpy as np
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class MedMNISTLoader(AbstractLoader):
    """Base loader for the MedMNIST family of biomedical image datasets.

    MedMNIST provides standardised, MNIST-shaped (28×28) biomedical images.
    Each concrete subclass sets :attr:`flag` (the MedMNIST subset key) and a
    sensible :attr:`default_minority` class. Images are flattened to
    ``size*size*n_channels``-dimensional vectors and scaled to ``[0, 1]``.

    Several subsets are naturally imbalanced (rare-disease classes), which is
    preserved in the multi-class mode (``binary=False``). In the default binary
    mode the :attr:`default_minority` class is labelled 1 and the rest 0.

    Source: https://medmnist.com/

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.1
        Fraction of data used for training in train/test splits.
    minority_reduce_scaler : int or None, default=None
        If set, reduce the minority class in the train set by this factor.
    minority_id : list[int] or None, default=None
        Class indices to treat as the minority class (class 1) in binary mode.
        Defaults to the subset's :attr:`default_minority`.
    binary : bool, default=True
        If True, convert to binary classification (minority class vs. rest).
        If False, keep the subset's native classes (preserves imbalance).
    classes_remove : list[int] or None, default=None
        Classes to remove entirely before creating the split.
    equal_test : bool, default=False
        If True, balance the test set classes.
    size : int, default=28
        MedMNIST image resolution (one of 28, 64, 128, 224).
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    default_dim_reducer: str = 'TSNE'
    is_image: bool = True

    #: MedMNIST subset key, e.g. ``'pneumoniamnist'`` (set by subclasses).
    flag: str = ''
    #: Registry-style display name, e.g. ``'PneumoniaMNIST'`` (set by subclasses).
    display_name: str = 'MedMNIST'
    #: Class index used as the minority (class 1) in default binary mode.
    default_minority: list[int] = [1]
    #: Number of image channels (1 for greyscale subsets, 3 for RGB subsets).
    n_channels: int = 1

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.1,
                 minority_reduce_scaler: int | None = None,
                 minority_id: list[int] | None = None,
                 binary: bool = True,
                 classes_remove: list[int] | None = None,
                 equal_test: bool = False,
                 size: int = 28,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name=self.display_name,
                         **kwargs)
        self.minority_id = minority_id if minority_id is not None else list(self.default_minority)
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.equal_test = equal_test
        self.size = size
        # MedMNIST images are stored H, W (greyscale) or H, W, C (RGB), i.e.
        # already in display order, so channels_first stays False.
        self.image_shape = (
            (self.size, self.size) if self.n_channels == 1
            else (self.size, self.size, self.n_channels)
        )

    def load_data(self) -> DataDict:
        """Download (if needed) and load the MedMNIST training split.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(n_samples, size*size*channels)``),
            ``'y'``, ``'description'``, and ``'label_names'``.
        """
        import medmnist
        from medmnist import INFO

        info = INFO[self.flag]
        this_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(this_dir, '..', 'datasets', 'MedMNIST')
        os.makedirs(download_dir, exist_ok=True)

        data_class = getattr(medmnist, info['python_class'])
        dataset = data_class(split='train', download=True, root=download_dir,
                             size=self.size)

        X = dataset.imgs.astype(np.float32).reshape(len(dataset.imgs), -1) / 255.0
        y = np.asarray(dataset.labels).squeeze().astype(np.int64)

        class_names = [info['label'][str(i)] for i in range(len(info['label']))]

        for cls in self.classes_remove:
            mask = y != cls
            X, y = X[mask], y[mask]

        if self.binary:
            binary_y = np.zeros(len(y), dtype=np.int64)
            for id_ in self.minority_id:
                binary_y[y == id_] = 1
            y = binary_y
            minority_names = ', '.join(class_names[i] for i in self.minority_id)
            label_names = ['Other classes', minority_names]
        else:
            label_names = class_names

        return {
            'X': X,
            'y': y,
            'description': f"{self.name}: {info['description']}",
            'label_names': label_names,
        }


class PneumoniaMNISTLoader(MedMNISTLoader):
    """Chest X-ray pneumonia screening (binary). Minority = pneumonia."""
    flag = 'pneumoniamnist'
    display_name = 'PneumoniaMNIST'
    default_minority = [1]


class BreastMNISTLoader(MedMNISTLoader):
    """Breast ultrasound (binary). Minority = malignant."""
    flag = 'breastmnist'
    display_name = 'BreastMNIST'
    default_minority = [0]


class DermaMNISTLoader(MedMNISTLoader):
    """Dermatoscope skin lesions (7 classes). Minority = melanoma."""
    flag = 'dermamnist'
    display_name = 'DermaMNIST'
    default_minority = [4]
    n_channels = 3


class BloodMNISTLoader(MedMNISTLoader):
    """Peripheral blood cell microscopy (8 classes). Minority = basophil."""
    flag = 'bloodmnist'
    display_name = 'BloodMNIST'
    default_minority = [0]
    n_channels = 3


class PathMNISTLoader(MedMNISTLoader):
    """Colon pathology tissue patches (9 classes). Minority = adenocarcinoma."""
    flag = 'pathmnist'
    display_name = 'PathMNIST'
    default_minority = [8]
    n_channels = 3


class OCTMNISTLoader(MedMNISTLoader):
    """Retinal OCT (4 classes). Minority = choroidal neovascularization."""
    flag = 'octmnist'
    display_name = 'OCTMNIST'
    default_minority = [0]


if __name__ == '__main__':
    loader = DermaMNISTLoader(binary=False)
    print(loader.get_info(long=True))
