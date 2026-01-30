# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generate guassian data
'''
import sklearn.utils
import numpy as np
from data_loaders.utils import set_seed
from data_loaders.abstract_loader import AbstractLoader


class gaussian_generator(AbstractLoader):
    def __init__(self,
                 shuffle=True,
                 num_samples=[200, 200],
                 means=[[0, 0], [5, 5]],
                 covs=[[[1, 0], [0, 1]],
                       [[1, 0], [0, 1]]],
                name='Gaussian Synthetic',
                 **kwargs):
        self.means = means
        self.covs = covs
        if isinstance(num_samples, int):
            self.num_samples = [num_samples//2, num_samples//2]
        else:
            self.num_samples = num_samples
        super().__init__(shuffle=shuffle,
                         dataset_name=name,
                         **kwargs)
    
    def load_data(self):
        data = self._get_two_normal_classes()
        return data


    def _get_two_normal_classes(self, 
                        seed=None):
        labels = [0, 1]
        X = []
        y = []
        for mean, cov, num_sample, label in zip(self.means, self.covs, self.num_samples, labels):
            set_seed(seed)
            X.append(np.random.multivariate_normal(mean, cov, size=num_sample))
            y.append(np.ones(num_sample)*label)
        X = np.vstack(X)
        y = np.hstack(y)

        return {'X': X, 'y': y}


def get_gaussian(samples=200,
                 gaussian_means=[[0, 0], [1, 1]],
                 gaussian_covs=[[[1, 0], [0, 1]],
                                [[1, 0], [0, 1]]],
                 test=False,
                 **kwargs):
    '''
    sample from two Gaussian dataset

    returns:
        - data: dict containing 'X', 'y'
    '''

    X = np.empty([0, 2])
    y = np.empty([0], dtype=np.int64)
    labels = [0, 1]
    for mean, cov, label in zip(gaussian_means, gaussian_covs, labels):
        # equal proportion of class samples
        class_samples = int(samples/len(labels))
        # set up current class' sampler
        gaussclass = GaussClass(mean, cov)
        # get random seed
        seed = 42+label
        if test == True:
            seed += 1
        # sample points
        gaussclass.gen_data(seed, class_samples)
        X = np.vstack([X, gaussclass.data])
        y = np.append(y, [label]*class_samples)
    X, y = sklearn.utils.shuffle(X, y, random_state=seed)
    return {'X': X, 'y':y, 'means':gaussian_means, 'covariances':gaussian_covs}


class GaussClass():
    def __init__(self, mean, covariance):
        self.mean = np.array(mean)
        self.cov = covariance

    def gen_data(self, randomseed, size):
        rng = np.random.default_rng(randomseed)
        self.data = np.array(rng.multivariate_normal(self.mean, self.cov, size))


if __name__ == "__main__":
    loader = gaussian_generator()
    # loader.plot_dataset()
    loader.plot_train_test_split(terminal_plot=True)