from __future__ import annotations

from data_loaders.main import AVAILABLE_DATASETS, get_dataset
from data_loaders.loaders.abstract_loader import AbstractLoader
from data_loaders.utils import Normaliser, proportional_split
from data_loaders.resampling import (
    AbstractResampler,
    proportional_upsample,
    RandomDuplicateUpsampler,
    SMOTEUpsampler,
    proportional_downsample,
    StratifiedSubsampler,
)
from data_loaders.embeddings import DimReducer

__all__ = [
    'get_dataset',
    'AVAILABLE_DATASETS',
    'AbstractLoader',
    'Normaliser',
    'proportional_split',
    'proportional_downsample',
    'StratifiedSubsampler',
    'AbstractResampler',
    'proportional_upsample',
    'RandomDuplicateUpsampler',
    'SMOTEUpsampler',
    'DimReducer',
    'get_available_dataset_list',
    'print_available_datasets',
]


def get_available_dataset_list() -> list[str]:
    """Return a list of all registered dataset names.

    Returns
    -------
    list[str]
        Names of all datasets in ``AVAILABLE_DATASETS``.
    """
    return list(AVAILABLE_DATASETS.keys())


def print_available_datasets() -> None:
    """Print all registered dataset names to stdout."""
    print('Available datasets:')
    for key in AVAILABLE_DATASETS.keys():
        print(f' - {key}')
