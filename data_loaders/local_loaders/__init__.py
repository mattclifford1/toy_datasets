from __future__ import annotations

from .abalone_gender import abalone_gender_loader
from .banknote import banknote_loader
from .breast_cancer_W import breast_cancer_W_loader
from .chronic_kidney_disease import chronic_kidney_disease_loader
from .costcla import (
    costcla_CreditScoring_Kaggle2011_loader,
    costcla_CreditScoring_PAKDD2009_loader,
    costcla_DirectMarketing_loader,
)
from .diabetes import diabetes_pima_indians_loader
from .Habermans_breast_cancer import habermans_breast_cancer_loader
from .hepititus import hepatitis_loader
from .ionosphere import ionosphere_loader
from .sonar_rocks import sonar_rocks_loader
from .wheat_seeds import wheat_seeds_loader

__all__ = [
    'abalone_gender_loader',
    'banknote_loader',
    'breast_cancer_W_loader',
    'chronic_kidney_disease_loader',
    'costcla_CreditScoring_Kaggle2011_loader',
    'costcla_CreditScoring_PAKDD2009_loader',
    'costcla_DirectMarketing_loader',
    'diabetes_pima_indians_loader',
    'habermans_breast_cancer_loader',
    'hepatitis_loader',
    'ionosphere_loader',
    'sonar_rocks_loader',
    'wheat_seeds_loader',
]
