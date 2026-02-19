# data_loaders.loaders.local_loaders

Loaders for CSV-backed datasets bundled with the package.

## Overview

These loaders read CSV files from `data_loaders/datasets/` and wrap them with
the standard `AbstractLoader` API. All return a dict with at least `'X'` and
`'y'`; cost-sensitive datasets additionally include `'feature_names'` and
`'costs'`.

## Loaders

| Class | Task |
|---|---|
| `DiabetesPimaIndiansLoader` | Diabetes onset prediction (Pima Indians) |
| `BanknoteLoader` | Banknote authenticity classification |
| `WheatSeedsLoader` | Wheat variety classification |
| `BreastCancerWLoader` | Breast cancer (Wisconsin variant) |
| `ChronicKidneyDiseaseLoader` | Chronic kidney disease prediction |
| `HepatitisLoader` | Hepatitis survival prediction |
| `IonosphereLoader` | Radar signal classification |
| `SonarRocksLoader` | Sonar rock vs. mine classification |
| `HabermansBreastCancerLoader` | Breast cancer survival (Haberman) |
| `AbaloneGenderLoader` | Abalone sex classification |
| `CostclaCreditScoringKaggle2011Loader` | Credit scoring — Kaggle 2011 |
| `CostclaCreditScoringPAKDD2009Loader` | Credit scoring — PAKDD 2009 |
| `CostclaDirectMarketingLoader` | Direct marketing response |

## Return format

```python
{
    'X':             np.ndarray,   # feature matrix
    'y':             np.ndarray,   # label array
    'feature_names': list[str],    # feature column names
    'costs':         np.ndarray,   # misclassification costs (cost datasets only)
}
```

## Usage

```python
from data_loaders.loaders.local_loaders import DiabetesPimaIndiansLoader

loader = DiabetesPimaIndiansLoader(scale=True, train_size=0.8)

train, test = loader.get_train_test_split()
print(loader.get_feature_names())
```
