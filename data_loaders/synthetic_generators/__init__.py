from .sklearn_synthetic_utils import _generic_sklearn_loader
from .XOR import XOR_generator
from .moons import moons_generator
from .blobs import blobs_generator
from .circles import circles_generator
from .classification import sklearn_normal_generator
from .gaussian import gaussian_generator

__all__ = [
    'XOR_generator',
    'moons_generator',
    'blobs_generator',
    'circles_generator',
    'sklearn_normal_generator',
    'gaussian_generator',
]
