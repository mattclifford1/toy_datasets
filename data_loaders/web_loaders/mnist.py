import os
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils import data as torch_data
from data_loaders.abstract_loader import AbstractLoader


class mnist_loader(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 split_size=0.1,
                 split_ratio=None,
                 size=60000,  # 60000 is full dataset
                 minority_id=[0],
                 binary=True,
                 classes_remove=[],
                 equal_test=False,
                 **kwargs):
        super().__init__(shuffle=shuffle,
                         split_size=split_size,
                         split_ratio=split_ratio,
                         dataset_name='MNIST',
                         **kwargs)
        self.size = size
        self.minority_id = minority_id
        self.binary = binary
        self.classes_remove = classes_remove
        self.equal_test = equal_test
    

    def load_data(self):
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
        for batch_idx, (d, t) in tqdm(enumerate(train_loader), total=self.size//256, desc='torch to numpy MNIST', leave=False):
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
        # formatting
        # print(len(y), sum(y))
        data = {'X':X, 'y':y}
        return data


if __name__ == '__main__':
    loader = mnist_loader()
    # loader.plot_dataset()
    loader.plot_train_test_split()
