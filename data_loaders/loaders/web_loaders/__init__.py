from __future__ import annotations

from .iris import IrisLoader
from .wine import WineLoader
from .breast_cancer import BreastCancerLoader
from .heart_disease import HeartDiseaseLoader
from .mnist import MnistLoader
from .cifar10 import Cifar10Loader
from .cifar100 import Cifar100Loader
from .cifar10n import Cifar10NLoader
from .fashion_mnist import FashionMnistLoader
from .svhn import SVHNLoader
from .eurosat import EuroSATLoader
from .med_mnist import (
    MedMNISTLoader,
    PneumoniaMNISTLoader,
    BreastMNISTLoader,
    DermaMNISTLoader,
    BloodMNISTLoader,
    PathMNISTLoader,
    OCTMNISTLoader,
)
from .parkinsons import ParkinsonsLoader
from .liver import IndianLiverLoader
from .arrhythmia import ArrhythmiaLoader
from .thoracic import ThoracicSurgeryLoader
from .spectf import SPECTFHeartLoader
from .heart_failure import HeartFailureLoader
from .mammographic import MammographicMassLoader
from .wpbc import BreastCancerPrognosticLoader
from .breast_cancer_coimbra import BreastCancerCoimbraLoader

__all__ = [
    'IrisLoader',
    'WineLoader',
    'BreastCancerLoader',
    'HeartDiseaseLoader',
    'MnistLoader',
    'Cifar10Loader',
    'Cifar100Loader',
    'Cifar10NLoader',
    'FashionMnistLoader',
    'SVHNLoader',
    'EuroSATLoader',
    'MedMNISTLoader',
    'PneumoniaMNISTLoader',
    'BreastMNISTLoader',
    'DermaMNISTLoader',
    'BloodMNISTLoader',
    'PathMNISTLoader',
    'OCTMNISTLoader',
    'ParkinsonsLoader',
    'IndianLiverLoader',
    'ArrhythmiaLoader',
    'ThoracicSurgeryLoader',
    'SPECTFHeartLoader',
    'HeartFailureLoader',
    'MammographicMassLoader',
    'BreastCancerPrognosticLoader',
    'BreastCancerCoimbraLoader',
]
