from data_loaders import utils
import data_loaders
from data_loaders.synthetic import (
    XOR,
    moons,
    blobs,
    circles,
    classification,
    normal,
    madelon,
    sklearn_synthetic_old,
    sklearn_synthetic_utils
)
from data_loaders.loaders import (
    abalone_gender, 
    banknote, 
    breast_cancer_W,
    cervical_cancer, 
    chronic_kidney_disease,
    costcla,
    diabetes, 
    Habermans_breast_cancer, 
    heart_disease,
    hepititus,
    ionosphere, 
    MIMIC_III,
    MIMIC_IV,
    mnist,
    sklearn_toy,
    sonar_rocks, 
    wheat_seeds, 
)


AVAILABLE_DATASETS = {
    # synthetic datasets
    # 'Gaussian': sample_dataset_to_proportions(get_gaussian),
    # 'Moons': sample_dataset_to_proportions(get_moons),

    # TODO: the ones int he synthetic folder

    # 'sklearn synthietc moons': todo
    # 'sklearn toy': todo

    # 'UCI Toy Datasets':
    # 'Gaussian': normal.get_normal_data_2_classes,
    # # 'Circles': sample_dataset_to_proportions(get_circles),
    # # 'Blobs': sample_dataset_to_proportions(get_blobs),
    'XOR': XOR.XOR_generator,
    # 'Madelon Separable': madelon.get_sep_datasets,
    # 'Madelon Non-Separable': madelon.get_non_sep_datasets,
    # 'Madelon High Dim Non-Separable': madelon.get_non_sep_data_high_dim,
    'Moons': moons.moons_generator,
    'Blobs': blobs.blobs_generator,
    'Circles': circles.circles_generator,
    'Sklearn Normal': classification.sklearn_normal_generator,
    # 'Iris': sklearn_toy.get_iris,
    # 'Wine': sklearn_toy.get_wine,
    # 'Breast Cancer': sklearn_toy.get_breast_cancer,

    # real datasets
    'Abalone Gender':               abalone_gender.abalone_gender_loader,
    'Banknote Authentication':      banknote.banknote_loader,
    'Breast Cancer Wisconsin':      breast_cancer_W.breast_cancer_W_loader,
    'Cervical Cancer':              cervical_cancer.cervical_cancer_loader,
    'Chronic Kidney Disease':       chronic_kidney_disease.chronic_kidney_disease_loader,
    'Costcla Credit Scoring Kaggle 2011': costcla.costcla_CreditScoring_Kaggle2011_loader,
    'Costcla Credit Scoring PAKDD 2009':  costcla.costcla_CreditScoring_PAKDD2009_loader,
    'Costcla Direct Marketing':     costcla.costcla_DirectMarketing_loader,
    'Diabetes Pima Indian':         diabetes.diabetes_pima_indians_loader,
    'Habermans Breast Cancer':      Habermans_breast_cancer.habermans_breast_cancer_loader,
    'Heart Disease':                heart_disease.heart_disease_loader,
    'Hepatitis':                    hepititus.hepatitis_loader,
    'Ionosphere':                   ionosphere.ionosphere_loader,
    # 'MIMIC-III Mortality': MIMIC_III.MIMIC_III_mortality_loader,
    # 'MIMIC-III Sepsis': MIMIC_III.MIMIC_III_sepsis_loader,
    # 'MIMIC-IV Ready For Discharge': MIMIC_IV.MIMIC_IV_ready_for_discharge_loader,
    'MNIST': mnist.mnist_loader,
    'Sonar Rocks vs Mines': sonar_rocks.sonar_rocks_loader,
    'Wheat Seeds': wheat_seeds.wheat_seeds_loader,
    }


def get_dataset(dataset_name, **kwargs):
    if dataset_name not in AVAILABLE_DATASETS.keys():
        raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(AVAILABLE_DATASETS.keys())}")
    loader_class = AVAILABLE_DATASETS[dataset_name]
    # initialise loader
    loader = loader_class(**kwargs)
    return loader



### OLD WAY OF DOING IT HERE ----> MAKE THIS INTO NEW METHOD ASAP
@utils.make_data_dim_reducer
def get_dataset_old(dataset='Breast Cancer', _print=True, scale=False, **kwargs):
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

    # print some info about dataset
    if _print == True:
        print(f"Dataset: {dataset}")
        print(f"    - Number of features: {data_set['data']['X'].shape[1]}")
        print(f"    - Total instances: {test0+train0+test1+train1}")
        print(f"      - Classes total: {test0+train0}:{test1+train1}")
        print(f"      -         train: {train0}:{train1}")
        print(f"      -         test:  {test0}:{test1}")
    
    return data_set



@utils.make_data_dim_reducer
def get_MNIST(scale=False):
    mnist.get_mnist()



# def get_SMOTE_data(data):
#     oversample = SMOTE()
#     X, y = oversample.fit_resample(data['X'], data['y'])
#     return {'X': X, 'y': y}


if __name__ == '__main__':
    data_loaders.print_available_datasets()


    dataset_str = "Abalone Gender"
    print(f"\n===========\nLoading {dataset_str}\n===========\n")
    dataset = data_loaders.get_dataset("Abalone Gender")
    print(dataset)
    dataset.plot_train_test_split()