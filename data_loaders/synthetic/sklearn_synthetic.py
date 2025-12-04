# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Generate synthetic data from sklearn datasets
'''

import sklearn.datasets
import sklearn.utils


def get_synthetic_sep_data_moons(N1=10000,
                           N2=10000,
                           scale=True,
                           test_nums=(10000, 10000)):

    data = get_moons((N1, N2))
    data_test = get_moons(test_nums)
    return {'data': data, 'data_test': data_test}


def _generic_loader(load_func, samples=[100, 100], test=False, **kwargs):
    '''
    sample from the a sklearn synthetic dataset
    returns:
        - data: dict containing 'X', 'y'
    '''
    seed = 0
    if test == True and load_func != sklearn.datasets.make_blobs:
        seed += 1

    X, y = load_func(n_samples=samples,
                     random_state=seed,
                     shuffle=False,
                     **kwargs)

    X, y = sklearn.utils.shuffle(X, y, random_state=seed)
    data = {'X': X, 'y': y}
    return data


def get_moons(samples=[100, 100], test=False, moons_noise=0, **kwargs):
    '''
    sample from the half moons data distribution
    returns:
        - data: dict containing 'X', 'y'
    '''
    data = _generic_loader(load_func=sklearn.datasets.make_moons,
                           samples=samples,
                           test=test,
                           noise=moons_noise)
    return data


def get_normal(samples=[100, 100], test=False, normal_dims=20, **kwargs):
    '''
    sample from the circles data distribution
    ** read docs to add more params here
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
    returns:
        - data: dict containing 'X', 'y'
    '''
    data = _generic_loader(load_func=sklearn.datasets.make_classification,
                           samples=samples,
                           test=test,
                           n_features=normal_dims
                           )
    return data


def get_circles(samples=[100, 100], test=False, circles_noise=0.2, **kwargs):
    '''
    sample from the circles data distribution
    returns:
        - data: dict containing 'X', 'y'
    '''
    data = _generic_loader(load_func=sklearn.datasets.make_circles,
                           samples=samples,
                           test=test,
                           noise=circles_noise,
                           factor=0.8)
    return data


def get_blobs(samples=[100, 100], test=False, blobs_features=2, **kwargs):
    '''
    sample from the circles data distribution
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
    returns:
        - data: dict containing 'X', 'y'
    '''
    data = _generic_loader(load_func=sklearn.datasets.make_blobs,
                           samples=samples,
                           test=test,
                           n_features=blobs_features)
    return data