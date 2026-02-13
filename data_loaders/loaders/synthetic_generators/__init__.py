from __future__ import annotations

from .sklearn_synthetic_utils import _generic_sklearn_loader
from .XOR import XORGenerator
from .moons import MoonsGenerator
from .blobs import BlobsGenerator
from .circles import CirclesGenerator
from .classification import SklearnNormalGenerator
from .gaussian import GaussianGenerator

__all__ = [
    'XORGenerator',
    'MoonsGenerator',
    'BlobsGenerator',
    'CirclesGenerator',
    'SklearnNormalGenerator',
    'GaussianGenerator',
]
