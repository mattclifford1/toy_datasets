from __future__ import annotations

from .iris import iris_loader
from .wine import wine_loader
from .breast_cancer import breast_cancer_loader
from .heart_disease import heart_disease_loader
from .mnist import mnist_loader

__all__ = [
    'iris_loader',
    'wine_loader',
    'breast_cancer_loader',
    'heart_disease_loader',
    'mnist_loader',
]
