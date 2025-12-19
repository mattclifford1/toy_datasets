from data_loaders.main import AVAILABLE_DATASETS


def get_available_dataset_list():
    return list(AVAILABLE_DATASETS.keys())


def print_available_datasets():
    '''
    TODO: get info about each dataset and print here
    '''
    print('Available datasets:')
    for key in AVAILABLE_DATASETS.keys():
        print(f' - {key}')