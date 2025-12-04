import numpy as np
from sklearn.utils import shuffle
from data_loaders.utils import set_seed, normaliser


def get_normal_data_2_classes(m1=[-1, -1],
             m2=[1, 1],
             cov1=[[1, 0], [0, 1]],
             cov2=[[1, 0], [0, 1]],
             N1=1000,
             N2=10,
             scale=False,
             test_nums=[1000, 1000],
             seed=None):
    data = get_two_classes(means=[m1, m2],
                                  covs=[cov1, cov2],
                                  num_samples=[N1, N2],
                                  seed=seed)
    data_test = get_two_classes(means=[m1, m2],
                                       covs=[cov1, cov2],
                                       num_samples=[test_nums[0], test_nums[1]],
                                       seed=seed)

    if scale == True:
        scaler = normaliser(data)
        data = scaler(data)
        data_test = scaler(data_test)
        m1 = scaler.transform_instance(m1)
        m2 = scaler.transform_instance(m2)

    return {'data': data, 'mean1': m1, 'mean2': m2, 'data_test': data_test}


def get_two_classes(means=[[0, 0], [10, 10]], 
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
    X, y = shuffle(X, y, random_state=seed)
    return {'X': X, 'y': y}


if __name__ == '__main__':
    data = get_two_classes(num_samples=[50, 100])
    print(data)