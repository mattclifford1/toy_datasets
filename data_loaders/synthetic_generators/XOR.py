import numpy as np
import sklearn.utils
from data_loaders import utils
from data_loaders.utils import set_seed
from data_loaders.abstract_loader import AbstractLoader


class XOR_generator(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 num_samples=200,
                 **kwargs):
        self.num_samples = num_samples
         # work out the split size and ratio from the numbers
        super().__init__(shuffle=shuffle,
                         dataset_name='XOR Synthetic',
                         **kwargs)
        
    def load_data(self):
        data = self._get_XOR_single()
        return data


    def _get_XOR_single(self):
        if isinstance(self.num_samples, int):
            num_samples = [self.num_samples//2, self.num_samples//2]
        else:
            num_samples = self.num_samples
        mu = 5
        cov = [[1, 0], [0, 1]]
        covs = [cov, cov]
        top_data = self._get_two_normal_classes(means=[[-mu, -mu], [mu, -mu]],
                                        covs=covs,
                                        num_samples=[num_samples[0]//2, num_samples[1]//2])
        bot_data = self._get_two_normal_classes(means=[[mu, mu], [-mu, mu]],
                                        covs=covs,
                                        num_samples=[num_samples[0]//2, num_samples[1]//2])

        X = np.vstack([top_data['X'], bot_data['X']])
        y = np.hstack([top_data['y'], bot_data['y']])

        return {'X': X, 'y': y}


    def _get_two_normal_classes(self, means=[[0, 0], [10, 10]], 
                        covs=[[[1, 0], [0, 1]],
                            [[1, 1], [1, 1]]], 
                        num_samples=[3, 2],
                        seed=None):
        labels = [0, 1]
        X = []
        y = []
        for mean, cov, num_sample, label in zip(means, covs, num_samples, labels):
            set_seed(seed)
            X.append(np.random.multivariate_normal(mean, cov, size=num_sample))
            y.append(np.ones(num_sample)*label)
        X = np.vstack(X)
        y = np.hstack(y)
        # X, y = sklearn.utils.shuffle(X, y, random_state=seed)
        return {'X': X, 'y': y}


if __name__ == "__main__":
    loader = XOR_generator(
        num_samples=500, 
        train_size=0.5, 
        minority_reduce_scaler=10, 
        equal_test=True,
        minority_reduce_scaler_test=10, 
        )
    plot = False
    if plot:
        loader.plot_train_test_split(terminal_plot=True)
        loader.plot_dataset(terminal_plot=True)
    else:
        loader.get_train_test_split()  # print out the stats of the train/test split