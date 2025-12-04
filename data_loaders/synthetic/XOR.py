import numpy as np
from data_loaders import utils
from data_loaders.synthetic import normal


def get_XOR(N1, 
            N2,
            scale=True,
            test_nums=[10000, 10000]):
    # sample data
    data = get_XOR(num_samples=[N1, N2])
    data_test = get_XOR(num_samples=test_nums)
    
    #scale
    if scale == True:
        scaler = utils.normaliser(data)
        data = scaler(data)
        data_test = scaler(data_test)

    return {'data': data, 'data_test': data_test}


def get_XOR(num_samples=[100, 100]):
    mu = 5
    cov = [[1, 0], [0, 1]]
    covs = [cov, cov]
    top_data = normal.get_two_classes(means=[[-mu, -mu], [mu, -mu]],
                                      covs=covs,
                                      num_samples=[num_samples[0]//2, num_samples[1]//2])
    bot_data = normal.get_two_classes(means=[[mu, mu], [-mu, mu]],
                                      covs=covs,
                                      num_samples=[num_samples[0]//2, num_samples[1]//2])

    X = np.vstack([top_data['X'], bot_data['X']])
    y = np.hstack([top_data['y'], bot_data['y']])

    return {'X': X, 'y': y}
