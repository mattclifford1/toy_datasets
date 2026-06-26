# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
UCI Thoracic Surgery Data:
https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data
'''
from __future__ import annotations

from typing import Any

from data_loaders.utils import binarise_labels, encode_categoricals, impute_missing
from data_loaders.loaders.abstract_loader import AbstractLoader, DataDict
from ucimlrepo import fetch_ucirepo


class ThoracicSurgeryLoader(AbstractLoader):
    """Load the UCI Thoracic Surgery dataset.

    Binary classification: predict 1-year post-operative survival of lung
    cancer surgery patients — survived (class 0) or died within one year
    (class 1) — from pre-operative risk factors (diagnosis, lung function,
    pain, dyspnoea, cough, smoking, comorbidities, tumour size, age). The
    classes are imbalanced (~15% deaths) and most features are categorical
    (boolean ``T``/``F`` flags and coded factors), which are integer-encoded.

    Dataset stats: 470 samples, 16 features.
    Source: https://archive.ics.uci.edu/dataset/277/thoracic+surgery+data

    Parameters
    ----------
    shuffle : bool, default=True
        Shuffle the dataset after loading.
    train_size : float, default=0.5
        Fraction of data used for training in train/test splits.
    **kwargs
        Additional keyword arguments forwarded to :class:`AbstractLoader`.
    """

    def __init__(self,
                 shuffle: bool = True,
                 train_size: float = 0.5,
                 **kwargs: Any) -> None:
        super().__init__(shuffle=shuffle,
                         train_size=train_size,
                         dataset_name='Thoracic Surgery',
                         short_description='Pre-op features for post-thoracotomy survival prediction — binary',
                         **kwargs)

    def load_data(self) -> DataDict:
        """Fetch and preprocess the Thoracic Surgery dataset.

        Returns
        -------
        DataDict
            Dict with keys ``'X'``, ``'y'``, ``'feature_names'``,
            ``'label_names'``, and ``'description'``.
        """
        thoracic = fetch_ucirepo(id=277)
        X = encode_categoricals(thoracic.data.features)

        # raw 'Risk1Yr': T = died within 1 year, F = survived
        y = binarise_labels(thoracic.data.targets['Risk1Yr'], {'F': 0, 'T': 1})
        return {
            'X': impute_missing(X),
            'y': y,
            'feature_names': X.columns.to_list(),
            'label_names': ['Survived', 'Died within 1yr'],
            'description': str(thoracic.metadata),
        }


if __name__ == "__main__":
    loader = ThoracicSurgeryLoader()
    print(loader.get_info(long=True))
