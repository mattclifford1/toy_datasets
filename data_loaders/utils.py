import functools

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np



RANDOM_STATE = 42


class normaliser:
    def __init__(self, train_X):
        self.scaler = preprocessing.MinMaxScaler(
            feature_range=(-1,1)).fit(train_X)

    def __call__(self, X):
        return self.scaler.transform(X)
    
    def transform_instance(self, X):
        return self.scaler.transform([X])[0]


def set_seed(seed):
    if seed == True:
        np.random.seed(seed=RANDOM_STATE)
    elif isinstance(seed, int):
        np.random.seed(seed=seed)
    elif seed == False:
        np.random.seed(seed=None)


def shuffle_data(data, seed=True):
    if seed == True:
        seed = RANDOM_STATE
    data['X'], data['y'] = shuffle(
        data['X'], data['y'], random_state=seed)
    return data


def shuffle_dataset(data, seed=True):
    '''
    shuffle numpy arrays (data) in data dict
    '''
    instances = data['X'].shape[0]
    # get random order
    set_seed(seed)
    p = np.random.permutation(instances)
    for key in data.keys():
        # apply to all numpy arrays that are data rows
        if type(data[key]) == np.ndarray and data[key].shape[0] == instances:
            # apply the shuffle
            data[key] = data[key][p]
    return data


def stratified_subsample(X, y, n_samples, random_state=42):
    '''
    Taken from mnists_path_dataset Repo https://github.com/mattclifford1/mnist_paths_dataset/blob/main/data_utils.py
    '''
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state
    )
    return X_sub, y_sub


def proportional_split(data,
                       train_size=0.8,
                       seed=True,
                       minority_reduce_scaler=None,
                       equal_test=False,
                       minority_reduce_scaler_test=None
                       ):
    '''
    create a train, test split that preserves the class distributions
        data: data dict holder (not modified)
        train_size: size of the train set (0.5 means equal train, test size)
        minority_reduce_scaler: if not None, scale down the minority class by this factor
        equal_test: if True, balance test set classes
        minority_reduce_scaler_test: if not None, scale down minority class in test set

    returns: train_split, test_split (two new dicts)
    '''
    print('minority_reduce_scaler:', minority_reduce_scaler)
    if train_size <= 0 or train_size > 1:
        raise ValueError(
            f'train_size needs to be between 0 and 1 instead of :{train_size}')
    if not isinstance(minority_reduce_scaler, type(None)) and minority_reduce_scaler < 1:
        raise ValueError(
            f'minority_reduce_scaler needs to be greater than 1 instead of :{minority_reduce_scaler}')
    if not isinstance(minority_reduce_scaler_test, type(None)) and minority_reduce_scaler_test < 1:
        raise ValueError(
            f'minority_reduce_scaler_test needs to be greater than 1 instead of :{minority_reduce_scaler_test}')
    
    set_seed(seed)
    # get current class proportions
    classes, counts = np.unique(data['y'], return_counts=True)
    classes = sorted(classes)
    test_inds = []
    train_inds = []
    for i, cls in enumerate(classes):
        # get all the inds of current class
        cls_inds = np.where(data['y'] == cls)[0]
        # shuffle all the inds to get a random selection
        set_seed(seed)
        np.random.shuffle(cls_inds)
        set_seed(seed)
        # now split the data inds into train/test
        split_point = int(counts[i]*train_size)
        if cls == 1: # minority class
            if not isinstance(minority_reduce_scaler, type(None)) and minority_reduce_scaler != False:
                split_point = max(int(len(train_inds[0])/minority_reduce_scaler), 1)
        train_inds.append(list(cls_inds[0:split_point]))

        # reduce minority class in test set if needed
        end_point = len(cls_inds)
        if cls == 1 and not equal_test: # minority class
            if not isinstance(minority_reduce_scaler_test, type(None)) and minority_reduce_scaler_test != False:
                len_test_minority = len(cls_inds[split_point:])
                # dont take til the end
                end_point = split_point + max(int(len_test_minority/minority_reduce_scaler_test), 1)
        test_inds.append(list(cls_inds[split_point:end_point]))
        
    # concat all the inds from each class
    train_inds = np.concatenate(train_inds)
    test_inds = np.concatenate(test_inds)
    # now apply the split to all data arrays
    train_split = {}
    test_split = {}
    instances = data['X'].shape[0]
    for key, val in data.items():
        # apply to all numpy arrays that are data rows
        if isinstance(val, np.ndarray) and data[key].shape[0] == instances:
            # extract and store the splits
            train_split[key] = val[train_inds]
            test_split[key] = val[test_inds]
    if equal_test == True:
        classes, counts = np.unique(test_split['y'], return_counts=True)
        max_inst = min(counts)
        for cls in classes:
            inds = np.arange(len(test_split['y']))
            inds = inds[test_split['y']==cls]
            inds_drop = inds[max_inst:]
            test_split['y'] = np.delete(test_split['y'], inds_drop)
            test_split['X'] = np.delete(test_split['X'], inds_drop, axis=0)


    return train_split, test_split


def proportional_downsample(data, percent_of_data=1, seed=True, **kwargs):
    '''
    downsample data whilst keep the represenetaed class proportion distribution
    the same
        data: data dict holder
        percent_of_data: % of the dataset to downsample to
        seed: True, False, or random seed number
    '''
    if percent_of_data <= 0 or percent_of_data > 100:
        raise ValueError(
            f'percent_of_data needs to be between 0 and 100 instead of :{percent_of_data}')
    set_seed(seed)
    # get current class proportions
    classes, counts = np.unique(data['y'], return_counts=True)
    # now downsample
    new_data_counts = (counts*(percent_of_data/100)).astype(np.uint64)
    # make sure we have at least a sample for train/test splits
    new_data_counts[new_data_counts < 2] = 2
    new_inds = []
    for i, cls in enumerate(classes):
        # get all the inds of current class
        cls_inds = np.where(data['y'] == cls)[0]
        # shuffle all the inds to get a random selection
        np.random.shuffle(cls_inds)
        # now store a subsample of class inds
        sub_sample_of_inds = cls_inds[:new_data_counts[i]]
        new_inds.append(list(sub_sample_of_inds))
    # concat all the inds from each class
    new_inds = np.concatenate(new_inds)
    # now only take new_inds from all data arrays
    instances = data['X'].shape[0]
    for key, val in data.items():
        # apply to all numpy arrays that are data rows
        if isinstance(val, np.ndarray) and data[key].shape[0] == instances:
            # apply the shuffle
            data[key] = val[new_inds]
    return data


# def make_data_dim_reducer(data_getter):
#     @functools.wraps(data_getter)
#     def _wrapper(*args, **kwargs):
#         data_dict = data_getter(*args, **kwargs)
#         if data_dict['data']['X'].shape[1] > 2:
#             dim_reducer = get_dim_reducer(data_dict['data'])
#         else:
#             dim_reducer = None
#         data_dict['dim_reducer'] = dim_reducer
#         return data_dict
#     return _wrapper


