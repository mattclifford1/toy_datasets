'''
Generate synthetic data from sklearn datasets
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import sklearn.datasets
import sklearn.utils
from data_loaders import utils


def _generic_sklearn_loader(load_func, samples=200, test=False, **kwargs):
    '''
    sample from the a sklearn synthetic dataset
    returns:
        - data: dict containing 'X', 'y'
    '''
    seed = 42
    if test == True and load_func != sklearn.datasets.make_blobs:
        seed += 1

    X, y = load_func(
        n_samples=samples,
        random_state=seed,
        shuffle=False,
        **kwargs
        )
    
    X, y = sklearn.utils.shuffle(X, y, random_state=seed)
    data = {'X': X, 'y':y}
    return data

