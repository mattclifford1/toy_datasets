from __future__ import annotations

import os
from typing import Any

import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils import data as torch_data
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict


class MnistLoader(AbstractLoader):
    """Load the MNIST handwritten-digit dataset.

    Images (28×28 pixels) are flattened to 784-dimensional vectors.
    In the default binary mode, a specified minority digit (default: digit 0) is
    labelled 1 and all other digits are labelled 0.

    Dataset stats: up to 60 000 training samples, 784 features per sample.

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
        Digit label(s) to treat as the minority class (class 1) in binary
        mode.  Defaults to ``[0]`` (digit 9).
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
    is_image: bool = True
    image_shape: tuple[int, ...] = (28, 28)

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.1,
                 minority_reduce_scaler: int | None = None,
                 size: int = 60000,  # 60000 is full dataset
                 minority_id: list[int] | None = None,
                 binary: bool = True,
                 classes_remove: list[int] | None = None,
                 equal_test: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=minority_reduce_scaler,
                         dataset_name='MNIST',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id if minority_id is not None else [0]
        self.binary = binary
        self.classes_remove = classes_remove if classes_remove is not None else []
        self.equal_test = equal_test


    def load_data(self) -> DataDict:
        """Download (if needed) and load MNIST training images.

        Returns
        -------
        DataDict
            Dict with keys ``'X'`` (shape ``(size, 784)``), ``'y'``,
            ``'description'``, and ``'label_names'``.
        """
        this_dir = os.path.dirname(os.path.abspath(__file__))
        download_dir = os.path.join(this_dir, '..', 'datasets')
        train_loader = torch_data.DataLoader(
            datasets.MNIST(download_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=256, shuffle=False, drop_last=False)

        # test_loader = torch_data.DataLoader(
        #     datasets.MNIST(download_dir, train=False,
        #                 transform=transforms.Compose([
        #                     transforms.ToTensor(),
        #                     transforms.Normalize((0.1307,), (0.3081,))
        #                 ])),
        #     batch_size=2048, shuffle=False, drop_last=False)


        
        X = np.empty([self.size, 28*28], dtype=np.float32)
        y = np.empty(self.size, dtype=np.int64)
        counter = 0
        for _, (d, t) in enumerate(train_loader):
            n = min(d.shape[0], self.size - counter)
            X[counter:counter + n] = d.numpy()[:n].reshape(n, -1)
            y[counter:counter + n] = t.numpy()[:n]
            counter += n
            if counter >= self.size:
                break
        
        # remove some classes
        for cls in self.classes_remove:
            X = np.delete(X, np.where(y==cls)[0], axis=0)
            y = np.delete(y, np.where(y==cls)[0])
        # convert data to binary
        if self.binary == True:
            for id in self.minority_id:
                y[y == id] = 11  # take out of range of labels
            y[y != 11] = 0
            y[y == 11] = 1
            # add label names
            label_names = ['Digits 0-8', 'Digit 9']
        else:
            label_names = [str(i) for i in range(10)]
        # formatting
        # print(len(y), sum(y))
        data = {'X':X, 'y':y}
        data['description'] = "The MNIST database of handwritten digits, with images of size 28x28 pixels."
        data['label_names'] = label_names
        return data


if __name__ == '__main__':
    loader = MnistLoader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    # loader.plot_train_test_split()
