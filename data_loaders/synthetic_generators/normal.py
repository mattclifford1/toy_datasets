from __future__ import annotations

from typing import Any

import numpy as np
from data_loaders.utils import set_seed
from data_loaders.abstract_loader import AbstractLoader, DataDict


class normal_data_loader(AbstractLoader):
    def __init__(self,
                 shuffle: bool = True,
                 num_train: int = 1000,
                 num_test: int = 2000,
                 train_ratio: int = 10,
                #  test_ratio=1,
                 set_seed: bool | int = True,
                 m1: list[float] = [-1, -1],
                 m2: list[float] = [1, 1],
                 cov1: list[list[float]] = [[1, 0], [0, 1]],
                 cov2: list[list[float]] = [[1, 0], [0, 1]],
                 scale: bool = False,
                 **kwargs: Any) -> None:
        # work out the split size and ratio from the numbers
        self.total_instances = num_train + num_test
        train_size = num_train / self.total_instances
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         minority_reduce_scaler=train_ratio,
                        #  equal_test=equal_test,
                         set_seed=set_seed,
                         dataset_name='Normal Synthetic',
                         scale=scale,
                         **kwargs)
        # data sampling parameters
        self.m1 = m1
        self.m2 = m2
        self.cov1 = cov1
        self.cov2 = cov2
        # determine how many points to sample from each class
        self.N1 = num_train // (1 + train_ratio)
        self.N2 = num_train - self.N1
        # add on equal test numbers for now need to implement test ratio to have this working
        self.N1 += num_test // 2
        self.N2 += num_test // 2


    def load_data(self) -> DataDict:
        labels = [0, 1]
        X = []
        y = []
        means = [self.m1, self.m2]
        covs = [self.cov1, self.cov2]
        num_samples = [self.N1, self.N2]
        for mean, cov, num_sample, label in zip(means, covs, num_samples, labels):
            set_seed(self.set_seed)
            X.append(np.random.multivariate_normal(mean, cov, size=num_sample))
            y.append(np.ones(num_sample)*label)
        X = np.vstack(X)
        y = np.hstack(y)
        data = {
            'X': X,
            'y': y,
            'description': 'Synthetic normal distributed data',
            'mean1': self.m1,
            'mean2': self.m2,
            }
        return data


if __name__ == "__main__":
    loader = normal_data_loader()
    loader.get_train_test_split()
    loader.plot_dataset(terminal_plot=True)
    # loader.plot_train_test_split()
