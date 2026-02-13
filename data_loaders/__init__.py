from __future__ import annotations

from data_loaders.main import AVAILABLE_DATASETS, get_dataset
from data_loaders.abstract_loader import AbstractLoader
from data_loaders.utils import normaliser, proportional_split, proportional_downsample
from data_loaders.embeddings import dim_reducer

__all__ = [
    'get_dataset',
    'AVAILABLE_DATASETS',
    'AbstractLoader',
    'normaliser',
    'proportional_split',
    'proportional_downsample',
    'dim_reducer',
    'get_available_dataset_list',
    'print_available_datasets',
]


def get_available_dataset_list() -> list[str]:
    return list(AVAILABLE_DATASETS.keys())


def print_available_datasets() -> None:
    print('Available datasets:')
    for key in AVAILABLE_DATASETS.keys():
        print(f' - {key}')
