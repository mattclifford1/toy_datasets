from data_loaders import utils
from data_loaders.synthetic import (
    XOR,
    normal,
    madelon,
    sklearn_synthetic
)
from data_loaders.loaders import (
    sklearn_toy, 
    diabetes, 
    Habermans_breast_cancer, 
    sonar_rocks, 
    banknote, 
    abalone_gender, 
    ionosphere, 
    wheat_seeds, 
    costcla,
    mnist,
    breast_cancer_W,
    hepititus,
    heart_disease,
    MIMIC_III,
    MIMIC_IV
)


AVAILABLE_DATASETS = {
        # 'Gaussian': sample_dataset_to_proportions(get_gaussian),
        # 'Moons': sample_dataset_to_proportions(get_moons),
        'Gaussian': normal.get_normal_data_2_classes,
        'Breast Cancer': sklearn_toy.get_breast_cancer,
        'Iris': sklearn_toy.get_iris,
        'Wine': sklearn_toy.get_wine,
        'Pima Indian Diabetes': diabetes.get_diabetes_indian,
        'Habermans breast cancer': Habermans_breast_cancer.get_Habermans_breast_cancer,
        'Sonar Rocks vs Mines': sonar_rocks.get_sonar,
        'Banknote Authentication': banknote.get_banknote,
        'Abalone Gender': abalone_gender.get_abalone,
        'Ionosphere': ionosphere.get_ionosphere,
        'Wheat Seeds': wheat_seeds.get_wheat_seeds,
        'Credit Scoring 1': costcla.costcla_dataset('CreditScoring_Kaggle2011_costcla'),
        'Credit Scoring 2': costcla.costcla_dataset('CreditScoring_PAKDD2009_costcla'),
        'Direct Marketing': costcla.costcla_dataset('DirectMarketing_costcla'),
        'MNIST': mnist.get_mnist,
        'Wisconsin Breast Cancer':  breast_cancer_W.get_Wisconsin_breast_cancer,
        'Hepatitis': hepititus.get_hepatitis,
        'Heart Disease': heart_disease.get_HD,
        'MIMIC-III': MIMIC_III.get_mortality,
        'MIMIC-III-mortality': MIMIC_III.get_mortality,
        'MIMIC-III-sepsis': MIMIC_III.get_sepsis,
        'MIMIC-IV': MIMIC_IV.get_ready_for_discharge,
        # 'Circles': sample_dataset_to_proportions(get_circles),
        # 'Blobs': sample_dataset_to_proportions(get_blobs),
        'XOR': XOR.get_XOR,
        'Madelon Separable': madelon.get_sep_datasets,
        'Madelon Non-Separable': madelon.get_non_sep_datasets,
        'Madelon High Dim Non-Separable': madelon.get_non_sep_data_high_dim,
        'Moon Separable': sklearn_synthetic.get_synthetic_sep_data_moons,
    }


def print_available_datasets():
    '''
    TODO: get info about each dataset and print here
    '''
    print('Available datasets:')
    for key in AVAILABLE_DATASETS.keys():
        print(f' - {key}')


@utils.make_data_dim_reducer
def get_dataset(dataset='Breast Cancer', _print=True, scale=False, **kwargs):
    # check input correct dataset name
    if dataset not in AVAILABLE_DATASETS.keys():
        raise ValueError(f'dataset needs to be one of:{AVAILABLE_DATASETS.keys()}')

    # load dataset
    data_set = AVAILABLE_DATASETS[dataset](**kwargs)
    if not isinstance(data_set, dict):
        # convert to dict format needed
        train_data, test_data = data_set
        data_set = {'data': train_data, 'data_test': test_data}

    # scale
    scaler = utils.normaliser(data_set['data'])
    if scale == True:
        data_set['data'] = scaler(data_set['data'])
        data_set['data_test'] = scaler(data_set['data_test'])

    train0 = len(data_set['data']['y'])-sum(data_set['data']['y'])
    train1 = sum(data_set['data']['y'])

    test0 = len(data_set['data_test']['y'])-sum(data_set['data_test']['y'])
    test1 = sum(data_set['data_test']['y'])
    if _print == True:
        print(f"{dataset}: {test0+train0+test1+train1}")
        print(f"Number of attribues: {data_set['data']['X'].shape[1]}")
        print( f"Classes total: {test0+train0} - {test1+train1}\n")
        print(f"Classes train: {train0} - {train1}")
        print(f"Classes test:  {test0} - {test1}")
    
    return data_set






@utils.make_data_dim_reducer
def get_MNIST(scale=False):
    mnist.get_mnist()



# def get_SMOTE_data(data):
#     oversample = SMOTE()
#     X, y = oversample.fit_resample(data['X'], data['y'])
#     return {'X': X, 'y': y}


if __name__ == '__main__':
    data = get_dataset(dataset='Madelon Non-Separable')
    print(data['data']['X'].shape)
    print(data['data_test']['X'].shape)
    print_available_datasets()