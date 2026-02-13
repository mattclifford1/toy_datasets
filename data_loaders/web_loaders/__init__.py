from __future__ import annotations

from .iris import IrisLoader
from .wine import WineLoader
from .breast_cancer import BreastCancerLoader
from .heart_disease import HeartDiseaseLoader
from .mnist import MnistLoader

__all__ = [
    'IrisLoader',
    'WineLoader',
    'BreastCancerLoader',
    'HeartDiseaseLoader',
    'MnistLoader',
]
