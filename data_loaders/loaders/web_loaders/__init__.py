from __future__ import annotations

from .iris import IrisLoader
from .wine import WineLoader
from .breast_cancer import BreastCancerLoader
from .heart_disease import HeartDiseaseLoader
from .mnist import MnistLoader
from .cifar10 import Cifar10Loader
from .cifar100 import Cifar100Loader
from .cifar10n import Cifar10NLoader

__all__ = [
    'IrisLoader',
    'WineLoader',
    'BreastCancerLoader',
    'HeartDiseaseLoader',
    'MnistLoader',
    'Cifar10Loader',
    'Cifar100Loader',
    'Cifar10NLoader',
]
