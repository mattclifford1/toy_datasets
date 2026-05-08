from __future__ import annotations

from collections.abc import Callable
from typing import Any
import importlib

import data_loaders


def _create_lazy_loader(module_path: str, class_name: str) -> Callable[..., Any]:
    """Create a factory function that imports a loader class on first use.

    Defers the import of heavy dependencies (e.g. torch, ucimlrepo) until the
    dataset is actually requested.

    Parameters
    ----------
    module_path : str
        Dotted module path, e.g. ``'data_loaders.web_loaders.iris'``.
    class_name : str
        Name of the loader class inside that module.

    Returns
    -------
    Callable
        A zero-argument-free factory that accepts ``**kwargs`` and returns an
        initialised loader instance.
    """
    def loader_factory(**kwargs: Any) -> Any:
        module = importlib.import_module(module_path)
        loader_class = getattr(module, class_name)
        return loader_class(**kwargs)
    return loader_factory


AVAILABLE_DATASETS: dict[str, Callable[..., Any]] = {
    # Synthetic datasets
    'XOR': _create_lazy_loader('data_loaders.loaders.synthetic_generators.XOR', 'XORGenerator'),
    'Moons': _create_lazy_loader('data_loaders.loaders.synthetic_generators.moons', 'MoonsGenerator'),
    'Blobs': _create_lazy_loader('data_loaders.loaders.synthetic_generators.blobs', 'BlobsGenerator'),
    'Circles': _create_lazy_loader('data_loaders.loaders.synthetic_generators.circles', 'CirclesGenerator'),
    'Sklearn Normal': _create_lazy_loader('data_loaders.loaders.synthetic_generators.classification', 'SklearnNormalGenerator'),
    'Gaussian': _create_lazy_loader('data_loaders.loaders.synthetic_generators.gaussian', 'GaussianGenerator'),

    # Toy datasets from sklearn
    'Iris': _create_lazy_loader('data_loaders.loaders.web_loaders.iris', 'IrisLoader'),
    'Wine': _create_lazy_loader('data_loaders.loaders.web_loaders.wine', 'WineLoader'),
    'Breast Cancer': _create_lazy_loader('data_loaders.loaders.web_loaders.breast_cancer', 'BreastCancerLoader'),

    # Real datasets
    'Abalone Gender': _create_lazy_loader('data_loaders.loaders.local_loaders.abalone_gender', 'AbaloneGenderLoader'),
    'Banknote Authentication': _create_lazy_loader('data_loaders.loaders.local_loaders.banknote', 'BanknoteLoader'),
    'Breast Cancer Wisconsin': _create_lazy_loader('data_loaders.loaders.local_loaders.breast_cancer_W', 'BreastCancerWLoader'),
    'Chronic Kidney Disease': _create_lazy_loader('data_loaders.loaders.local_loaders.chronic_kidney_disease', 'ChronicKidneyDiseaseLoader'),
    'Costcla Credit Scoring Kaggle 2011': _create_lazy_loader('data_loaders.loaders.local_loaders.costcla', 'CostclaCreditScoringKaggle2011Loader'),
    'Costcla Credit Scoring PAKDD 2009': _create_lazy_loader('data_loaders.loaders.local_loaders.costcla', 'CostclaCreditScoringPAKDD2009Loader'),
    'Costcla Direct Marketing': _create_lazy_loader('data_loaders.loaders.local_loaders.costcla', 'CostclaDirectMarketingLoader'),
    'Diabetes Pima Indian': _create_lazy_loader('data_loaders.loaders.local_loaders.diabetes', 'DiabetesPimaIndiansLoader'),
    'Habermans Breast Cancer': _create_lazy_loader('data_loaders.loaders.local_loaders.Habermans_breast_cancer', 'HabermansBreastCancerLoader'),
    'Heart Disease': _create_lazy_loader('data_loaders.loaders.web_loaders.heart_disease', 'HeartDiseaseLoader'),
    'Hepatitis': _create_lazy_loader('data_loaders.loaders.local_loaders.hepititus', 'HepatitisLoader'),
    'Ionosphere': _create_lazy_loader('data_loaders.loaders.local_loaders.ionosphere', 'IonosphereLoader'),
    'MNIST': _create_lazy_loader('data_loaders.loaders.web_loaders.mnist', 'MnistLoader'),
    'CIFAR-10': _create_lazy_loader('data_loaders.loaders.web_loaders.cifar10', 'Cifar10Loader'),
    'CIFAR-100': _create_lazy_loader('data_loaders.loaders.web_loaders.cifar100', 'Cifar100Loader'),
    'CIFAR-10N': _create_lazy_loader('data_loaders.loaders.web_loaders.cifar10n', 'Cifar10NLoader'),
    'Sonar Rocks vs Mines': _create_lazy_loader('data_loaders.loaders.local_loaders.sonar_rocks', 'SonarRocksLoader'),
    'Wheat Seeds': _create_lazy_loader('data_loaders.loaders.local_loaders.wheat_seeds', 'WheatSeedsLoader'),
    }


def get_dataset(
        dataset_name: str, 
        **kwargs: Any
        ) -> data_loaders.AbstractLoader:
    """Load and return a dataset loader by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Must be a key in ``AVAILABLE_DATASETS``.
    **kwargs
        Additional keyword arguments forwarded to the loader's constructor
        (e.g. ``train_size``, ``minority_reduce_scaler``, ``scale``).

    Returns
    -------
    AbstractLoader
        An initialised loader instance for the requested dataset.

    Raises
    ------
    ValueError
        If ``dataset_name`` is not found in ``AVAILABLE_DATASETS``.
    """
    if dataset_name not in AVAILABLE_DATASETS.keys():
        raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(AVAILABLE_DATASETS.keys())}")
    loader_class = AVAILABLE_DATASETS[dataset_name]
    # initialise loader
    loader = loader_class(**kwargs)
    return loader



if __name__ == '__main__':
    print(f"Available datasets:")
    for dataset_name in AVAILABLE_DATASETS.keys():
        print(f"    - {dataset_name}")

    print("\nExample usage:")
    dataset_name = 'Iris'
    print(f"Loading '{dataset_name}' dataset...")
    loader = get_dataset(dataset_name)
    X = loader.get_X()
    y = loader.get_y()
    print(f"Loaded '{dataset_name}' with {X.shape[0]} instances and {X.shape[1]} features.")