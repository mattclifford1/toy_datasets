from __future__ import annotations

import os
from typing import Any

import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils import data as torch_data
from data_loaders.abstract_loader import AbstractLoader, DataDict


class mnist_loader(AbstractLoader):
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
        # X = np.empty([self.size, 1, 28, 28], dtype=np.float32)
        y = np.zeros(self.size, dtype=np.int64)
        counter = 0
        # for batch_idx, (d, t) in tqdm(enumerate(train_loader), total=self.size//256, desc='torch to numpy MNIST', leave=False):
        for batch_idx, (d, t) in enumerate(train_loader):
            d_np = d.numpy()
            # t_np = t.numpy()
            for i in range(d.shape[0]):
                X[counter, :] = d_np[i, :, :, :].reshape(-1)
                # X[counter, :, :, :] = d_np[i, :, :, :]
                y[counter] = t[i]
                counter += 1
                if counter == self.size:
                    break
            if counter == self.size:
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
    loader = mnist_loader()
    print(loader.get_info(long=True))
    # loader.plot_dataset()
    # loader.plot_train_test_split()
