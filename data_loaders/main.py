from __future__ import annotations

from collections.abc import Callable
from typing import Any

import importlib

import numpy as np
from tqdm import tqdm

from data_loaders import utils
import data_loaders


def _create_lazy_loader(module_path: str, class_name: str) -> Callable[..., Any]:
    """Create a factory function that imports the loader class on first use."""
    def loader_factory(**kwargs: Any) -> Any:
        module = importlib.import_module(module_path)
        loader_class = getattr(module, class_name)
        return loader_class(**kwargs)
    return loader_factory


AVAILABLE_DATASETS: dict[str, Callable[..., Any]] = {
    # Synthetic datasets
    'XOR': _create_lazy_loader('data_loaders.synthetic_generators.XOR', 'XOR_generator'),
    'Moons': _create_lazy_loader('data_loaders.synthetic_generators.moons', 'moons_generator'),
    'Blobs': _create_lazy_loader('data_loaders.synthetic_generators.blobs', 'blobs_generator'),
    'Circles': _create_lazy_loader('data_loaders.synthetic_generators.circles', 'circles_generator'),
    'Sklearn Normal': _create_lazy_loader('data_loaders.synthetic_generators.classification', 'sklearn_normal_generator'),
    'Gaussian': _create_lazy_loader('data_loaders.synthetic_generators.gaussian', 'gaussian_generator'),

    # Toy datasets from sklearn
    'Iris': _create_lazy_loader('data_loaders.web_loaders.iris', 'iris_loader'),
    'Wine': _create_lazy_loader('data_loaders.web_loaders.wine', 'wine_loader'),
    'Breast Cancer': _create_lazy_loader('data_loaders.web_loaders.breast_cancer', 'breast_cancer_loader'),

    # Real datasets
    'Abalone Gender': _create_lazy_loader('data_loaders.local_loaders.abalone_gender', 'abalone_gender_loader'),
    'Banknote Authentication': _create_lazy_loader('data_loaders.local_loaders.banknote', 'banknote_loader'),
    'Breast Cancer Wisconsin': _create_lazy_loader('data_loaders.local_loaders.breast_cancer_W', 'breast_cancer_W_loader'),
    'Chronic Kidney Disease': _create_lazy_loader('data_loaders.local_loaders.chronic_kidney_disease', 'chronic_kidney_disease_loader'),
    'Costcla Credit Scoring Kaggle 2011': _create_lazy_loader('data_loaders.local_loaders.costcla', 'costcla_CreditScoring_Kaggle2011_loader'),
    'Costcla Credit Scoring PAKDD 2009': _create_lazy_loader('data_loaders.local_loaders.costcla', 'costcla_CreditScoring_PAKDD2009_loader'),
    'Costcla Direct Marketing': _create_lazy_loader('data_loaders.local_loaders.costcla', 'costcla_DirectMarketing_loader'),
    'Diabetes Pima Indian': _create_lazy_loader('data_loaders.local_loaders.diabetes', 'diabetes_pima_indians_loader'),
    'Habermans Breast Cancer': _create_lazy_loader('data_loaders.local_loaders.Habermans_breast_cancer', 'habermans_breast_cancer_loader'),
    'Heart Disease': _create_lazy_loader('data_loaders.web_loaders.heart_disease', 'heart_disease_loader'),
    'Hepatitis': _create_lazy_loader('data_loaders.local_loaders.hepititus', 'hepatitis_loader'),
    'Ionosphere': _create_lazy_loader('data_loaders.local_loaders.ionosphere', 'ionosphere_loader'),
    'MNIST': _create_lazy_loader('data_loaders.web_loaders.mnist', 'mnist_loader'),
    'Sonar Rocks vs Mines': _create_lazy_loader('data_loaders.local_loaders.sonar_rocks', 'sonar_rocks_loader'),
    'Wheat Seeds': _create_lazy_loader('data_loaders.local_loaders.wheat_seeds', 'wheat_seeds_loader'),
    }


def get_dataset(dataset_name: str, **kwargs: Any) -> data_loaders.AbstractLoader:
    if dataset_name not in AVAILABLE_DATASETS.keys():
        raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(AVAILABLE_DATASETS.keys())}")
    loader_class = AVAILABLE_DATASETS[dataset_name]
    # initialise loader
    loader = loader_class(**kwargs)
    return loader



### OLD WAY OF DOING IT HERE ----> MAKE THIS INTO NEW METHOD ASAP
# @utils.make_data_dim_reducer
def get_dataset_old(
        dataset: str = 'Breast Cancer',
        _print: bool = True,
        scale: bool = False,
        **kwargs: Any,
) -> dict[str, Any]:
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



def test_available_datasets(_print: bool = False) -> None:
    infos: list[str] = []
    for dataset_name in tqdm(AVAILABLE_DATASETS.keys(), desc="Testing loading datasets"):
        if _print:
            print(dataset_name)
        dataset = data_loaders.get_dataset(dataset_name)
        X = dataset.get_X()
        y = dataset.get_y()
        if not isinstance(X, np.ndarray):
            raise ValueError(f"Dataset {dataset_name} X is not a numpy array")
        if not isinstance(y, np.ndarray):
            raise ValueError(f"Dataset {dataset_name} y is not a numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Dataset {dataset_name} X and y have different number of instances: {X.shape[0]} vs {y.shape[0]}")
        if y.shape[0] < 10:
            raise ValueError(f"Dataset {dataset_name} has less than 10 instances: {X.shape[0]}")
        infos.append(dataset.get_info(long=False))

    if _print:
        print("\n\n".join(infos))


if __name__ == '__main__':
    print(f"Testing available datasets:")
    for dataset_name in AVAILABLE_DATASETS.keys():
        print(f"    - {dataset_name}")
    test_available_datasets(_print=False)
