from __future__ import annotations

from .abalone_gender import AbaloneGenderLoader
from .banknote import BanknoteLoader
from .breast_cancer_W import BreastCancerWLoader
from .cervical_cancer import CervicalCancerLoader
from .chronic_kidney_disease import ChronicKidneyDiseaseLoader
from .costcla import (
    CostclaCreditScoringKaggle2011Loader,
    CostclaCreditScoringPAKDD2009Loader,
    CostclaDirectMarketingLoader,
)
from .diabetes import DiabetesPimaIndiansLoader
from .Habermans_breast_cancer import HabermansBreastCancerLoader
from .hepititus import HepatitisLoader
from .ionosphere import IonosphereLoader
from .sonar_rocks import SonarRocksLoader
from .wheat_seeds import WheatSeedsLoader

__all__ = [
    'AbaloneGenderLoader',
    'BanknoteLoader',
    'BreastCancerWLoader',
    'CervicalCancerLoader',
    'ChronicKidneyDiseaseLoader',
    'CostclaCreditScoringKaggle2011Loader',
    'CostclaCreditScoringPAKDD2009Loader',
    'CostclaDirectMarketingLoader',
    'DiabetesPimaIndiansLoader',
    'HabermansBreastCancerLoader',
    'HepatitisLoader',
    'IonosphereLoader',
    'SonarRocksLoader',
    'WheatSeedsLoader',
]
