# toy_datasets

Python package providing a unified interface for loading synthetic and real-world toy datasets for ML experimentation.

Built around a **deep-module** philosophy: a tiny, consistent interface
(`get_dataset(name)` → `get_X()`, `get_train_test_split()`, `plot_dataset()`)
that works the same for every dataset, backed by a powerful base class that
handles splitting, balancing, scaling, dimensionality reduction and plotting for
you. The simple case stays simple, but deep customisation is there when you need
it.

## Features

- Unified API for 40+ datasets via `get_dataset(name)`
- Consistent dict format: `{'X': features, 'y': labels}`
- Built-in train/test splitting with class balance preservation
- MinMax normalization to [-1, 1] range
- Dimensionality reduction (PCA, UMAP, t-SNE)
- Classifier visualization: decision boundary shading and misclassification markers
- Train/test overlay mode with configurable test transparency
- Example-image previews for image datasets (`plot_class_samples()`)
- Terminal-based plotting (sixel/kitty protocols)


## Available Datasets

**Synthetic:**
- `XOR`, `Moons`, `Blobs`, `Circles`, `Gaussian`, `Sklearn Normal`

**Classic (sklearn):**
- `Iris`, `Wine`, `Breast Cancer`

**Medical (incl. imbalanced / rare-disease):**
- `Diabetes Pima Indian`, `Heart Disease`, `Breast Cancer Wisconsin`
- `Habermans Breast Cancer`, `Chronic Kidney Disease`, `Hepatitis`
- `Parkinsons`, `Indian Liver Patient`, `Cervical Cancer`, `Arrhythmia`

**Image:**
- `MNIST`, `Fashion-MNIST`, `SVHN`, `EuroSAT`, `CIFAR-10`, `CIFAR-100`, `CIFAR-10N`

**Medical image (MedMNIST):**
- `PneumoniaMNIST`, `BreastMNIST`, `DermaMNIST`
- `BloodMNIST`, `PathMNIST`, `OCTMNIST`

**Other:**
- `Banknote Authentication`, `Wheat Seeds`, `Ionosphere`
- `Sonar Rocks vs Mines`, `Abalone Gender`
- `German Credit`
- `Costcla Credit Scoring Kaggle 2011`, `Costcla Credit Scoring PAKDD 2009`
- `Costcla Direct Marketing`

List all available datasets:
```python
from data_loaders.main import AVAILABLE_DATASETS
print(list(AVAILABLE_DATASETS.keys()))
```

### Dataset Preview

The **Moons** dataset — two interleaved half-circles, a classic benchmark for non-linear classifiers:

![Moons](assets/figures/moons.png)

#### All Datasets (click to expand)
<details>
<summary><strong>Synthetic</strong></summary>

#### XOR

```
XOR Synthetic — 2D XOR-patterned binary classification (4 quadrants)

 Label Names:
    - Label 0: 0
    - Label 1: 1
    - Label 2: 2
    - Label 3: 3

 Dataset Info:
    - Number of features: 2
    - Total instances: 200
      - Class 0: 100 instances (0)
      - Class 1: 100 instances (1)
```

![XOR](assets/figures/xor.png)

#### Moons

```
Moons Synthetic — Two interleaving half-moons, non-linearly separable

 Label Names:
    - Label 0: 0
    - Label 1: 1
    - Label 2: 2
    - Label 3: 3

 Dataset Info:
    - Number of features: 2
    - Total instances: 200
      - Class 0: 100 instances (0)
      - Class 1: 100 instances (1)
```

![Moons](assets/figures/moons.png)

#### Blobs

```
Blobs Synthetic — Isotropic Gaussian blobs at random centres

 Label Names:
    - Label 0: 0
    - Label 1: 1
    - Label 2: 2
    - Label 3: 3

 Dataset Info:
    - Number of features: 2
    - Total instances: 200
      - Class 0: 100 instances (0)
      - Class 1: 100 instances (1)
```

![Blobs](assets/figures/blobs.png)

#### Circles

```
Circles Synthetic — Concentric circles, non-linearly separable

 Label Names:
    - Label 0: 0
    - Label 1: 1
    - Label 2: 2
    - Label 3: 3

 Dataset Info:
    - Number of features: 2
    - Total instances: 200
      - Class 0: 100 instances (0)
      - Class 1: 100 instances (1)
```

![Circles](assets/figures/circles.png)

#### Sklearn Normal

```
Sklearn Synthetic Classification (Normal) — Sklearn make_classification with Gaussian cluster features

 Label Names:
    - Label 0: 0
    - Label 1: 1
    - Label 2: 2
    - Label 3: 3

 Dataset Info:
    - Number of features: 20
    - Total instances: 200
      - Class 0: 102 instances (0)
      - Class 1: 98 instances (1)
```

![Sklearn Normal](assets/figures/sklearn_normal.png)

#### Gaussian

```
Gaussian Synthetic — Multi-class overlapping Gaussian distributions

 Feature Names:
    - Feature 0: Feature 1
    - Feature 1: Feature 2

 Label Names:
    - Label 0: Class 0
    - Label 1: Class 1

 Dataset Info:
    - Number of features: 2
    - Total instances: 200
      - Class 0: 100 instances (Class 0)
      - Class 1: 100 instances (Class 1)
```

![Gaussian](assets/figures/gaussian.png)

</details>

<details>
<summary><strong>Classic (sklearn)</strong></summary>

#### Iris

```
Iris — Fisher's iris flower classification — 3 species, 4 morphological features

 Feature Names:
    - Feature 0: sepal length (cm)
    - Feature 1: sepal width (cm)
    - Feature 2: petal length (cm)
    - Feature 3: petal width (cm)

 Label Names:
    - Label 0: setosa and virginica
    - Label 1: versicolor

 Dataset Info:
    - Number of features: 4
    - Total instances: 150
      - Class 0: 100 instances (setosa and virginica)
      - Class 1: 50 instances (versicolor)
```

![Iris](assets/figures/iris.png)

#### Wine

```
Wine — Wine cultivar classification from chemical analysis — 3 classes, 13 features

 Feature Names:
    - Feature 0: alcohol
    - Feature 1: malic_acid
    - Feature 2: ash
    - Feature 3: alcalinity_of_ash
    - Feature 4: magnesium
    - Feature 5: total_phenols
    - Feature 6: flavanoids
    - Feature 7: nonflavanoid_phenols
    - Feature 8: proanthocyanins
    - Feature 9: color_intensity
    - Feature 10: hue
    - Feature 11: od280/od315_of_diluted_wines
    - Feature 12: proline

 Label Names:
    - Label 0: class_0
    - Label 1: class_1 and class_2

 Dataset Info:
    - Number of features: 13
    - Total instances: 178
      - Class 0: 59 instances (class_0)
      - Class 1: 119 instances (class_1 and class_2)
```

![Wine](assets/figures/wine.png)

#### Breast Cancer

```
Breast Cancer — Digitised FNA cell-nuclei features — binary malignant/benign, 30 features

 Feature Names:
    - Feature 0: mean radius
    - Feature 1: mean texture
    - Feature 2: mean perimeter
    - Feature 3: mean area
    - Feature 4: mean smoothness
    - Feature 5: mean compactness
    - Feature 6: mean concavity
    - Feature 7: mean concave points
    - Feature 8: mean symmetry
    - Feature 9: mean fractal dimension
    - Feature 10: radius error
    - Feature 11: texture error
    - Feature 12: perimeter error
    - Feature 13: area error
    - Feature 14: smoothness error
    - Feature 15: compactness error
    - Feature 16: concavity error
    - Feature 17: concave points error
    - Feature 18: symmetry error
    - Feature 19: fractal dimension error
    - Feature 20: worst radius
    - Feature 21: worst texture
    - Feature 22: worst perimeter
    - Feature 23: worst area
    - Feature 24: worst smoothness
    - Feature 25: worst compactness
    - Feature 26: worst concavity
    - Feature 27: worst concave points
    - Feature 28: worst symmetry
    - Feature 29: worst fractal dimension

 Label Names:
    - Label 0: benign
    - Label 1: malignant

 Dataset Info:
    - Number of features: 30
    - Total instances: 569
      - Class 0: 357 instances (benign)
      - Class 1: 212 instances (malignant)
```

![Breast Cancer](assets/figures/breast_cancer.png)

</details>

<details>
<summary><strong>Medical</strong></summary>

#### Diabetes Pima Indian

```
Diabetes Pima Indians — Pima Indians 8 clinical indicators for diabetes onset — binary

 Feature Names:
    - Feature 0: Pregnancies
    - Feature 1: Glucose
    - Feature 2: BloodPressure
    - Feature 3: SkinThickness
    - Feature 4: Insulin
    - Feature 5: BMI
    - Feature 6: DiabetesPedigreeFunction
    - Feature 7: Age

 Label Names:
    - Label 0: No Diabetes
    - Label 1: Diabetes

 Dataset Info:
    - Number of features: 8
    - Total instances: 768
      - Class 0: 500 instances (No Diabetes)
      - Class 1: 268 instances (Diabetes)
```

![Diabetes Pima Indian](assets/figures/diabetes_pima_indian.png)

#### Heart Disease

```
Heart Disease — Cleveland Clinic ECG/clinical indicators for heart disease — binary

 Label Names:
    - Label 0: no heart disease
    - Label 1: heart disease

 Dataset Info:
    - Number of features: 11
    - Total instances: 212
      - Class 0: 164 instances (no heart disease)
      - Class 1: 48 instances (heart disease)
```

![Heart Disease](assets/figures/heart_disease.png)

#### Breast Cancer Wisconsin

```
Wisconsin Breast Cancer — Cell nucleus features for malignant/benign breast cancer classification — binary

 Feature Names:
    - Feature 0: radius1
    - Feature 1: texture1
    - Feature 2: perimeter1
    - Feature 3: area1
    - Feature 4: smoothness1
    - Feature 5: compactness1
    - Feature 6: concavity1
    - Feature 7: concave_points1
    - Feature 8: symmetry1
    - Feature 9: fractal_dimension1
    - Feature 10: radius2
    - Feature 11: texture2
    - Feature 12: perimeter2
    - Feature 13: area2
    - Feature 14: smoothness2
    - Feature 15: compactnes2
    - Feature 16: concavity2
    - Feature 17: concave_points2
    - Feature 18: symmetry2
    - Feature 19: fractal_dimension2
    - Feature 20: radius3
    - Feature 21: texture3
    - Feature 22: perimeter3
    - Feature 23: area3
    - Feature 24: smoothness3
    - Feature 25: compactness3
    - Feature 26: concavity3
    - Feature 27: concave_points3
    - Feature 28: symmetry3
    - Feature 29: fractal_dimension3

 Label Names:
    - Label 0: Benign
    - Label 1: Malignant

 Dataset Info:
    - Number of features: 30
    - Total instances: 569
      - Class 0: 357 instances (Benign)
      - Class 1: 212 instances (Malignant)
```

![Breast Cancer Wisconsin](assets/figures/breast_cancer_wisconsin.png)

#### Habermans Breast Cancer

```
Habermans Breast Cancer — Post-operative survival of breast cancer patients (1958-1970) — binary

 Feature Names:
    - Feature 0: Age
    - Feature 1: Operation_Year
    - Feature 2: Positive_Aux_Nodes

 Label Names:
    - Label 0: survived 5 years or longer
    - Label 1: died within 5 year

 Dataset Info:
    - Number of features: 3
    - Total instances: 306
      - Class 0: 225 instances (survived 5 years or longer)
      - Class 1: 81 instances (died within 5 year)
```

![Habermans Breast Cancer](assets/figures/habermans_breast_cancer.png)

#### Chronic Kidney Disease

```
Chronic Kidney Disease — Clinical indicators for chronic kidney disease detection — binary

 Feature Names:
    - Feature 0: age
    - Feature 1: blood pressure
    - Feature 2: specific gravity
    - Feature 3: pus cell
    - Feature 4: pus cell clumps
    - Feature 5: bacteria
    - Feature 6: blood urea
    - Feature 7: serum creatinine
    - Feature 8: hemoglobin
    - Feature 9: hypertension
    - Feature 10: diabetes mellitus
    - Feature 11: coronary artery disease
    - Feature 12: appetite
    - Feature 13: pedal edema
    - Feature 14: anemia

 Label Names:
    - Label 0: Chronic Kidney Disease
    - Label 1: Not Chronic Kidney Disease

 Dataset Info:
    - Number of features: 15
    - Total instances: 268
      - Class 0: 121 instances (Chronic Kidney Disease)
      - Class 1: 147 instances (Not Chronic Kidney Disease)
```

![Chronic Kidney Disease](assets/figures/chronic_kidney_disease.png)

#### Hepatitis

```
Hepatitis — Clinical features for hepatitis patient survival prediction — binary

 Feature Names:
    - Feature 0: AGE
    - Feature 1: SEX
    - Feature 2: STEROID
    - Feature 3: ANTIVIRALS
    - Feature 4: FATIGUE
    - Feature 5: MALAISE
    - Feature 6: ANOREXIA
    - Feature 7: LIVERBIG
    - Feature 8: LIVERFIRM
    - Feature 9: SPLEENPALPABLE
    - Feature 10: SPIDERS
    - Feature 11: ASCITES
    - Feature 12: VARICES
    - Feature 13: BILIRUBIN
    - Feature 14: SGOT
    - Feature 15: HISTOLOGY

 Label Names:
    - Label 0: Survived
    - Label 1: Died

 Dataset Info:
    - Number of features: 16
    - Total instances: 137
      - Class 0: 111 instances (Survived)
      - Class 1: 26 instances (Died)
```

![Hepatitis](assets/figures/hepatitis.png)

#### Parkinsons

```
Parkinsons — Biomedical voice measures for Parkinson's disease diagnosis — binary

 Feature Names:
    - Feature 0: MDVP:Fo
    - Feature 1: MDVP:Fhi
    - Feature 2: MDVP:Flo
    - Feature 3: MDVP:Jitter
    - Feature 4: MDVP:Jitter
    - Feature 5: MDVP:RAP
    - Feature 6: MDVP:PPQ
    - Feature 7: Jitter:DDP
    - Feature 8: MDVP:Shimmer
    - Feature 9: MDVP:Shimmer
    - Feature 10: Shimmer:APQ3
    - Feature 11: Shimmer:APQ5
    - Feature 12: MDVP:APQ
    - Feature 13: Shimmer:DDA
    - Feature 14: NHR
    - Feature 15: HNR
    - Feature 16: RPDE
    - Feature 17: DFA
    - Feature 18: spread1
    - Feature 19: spread2
    - Feature 20: D2
    - Feature 21: PPE

 Label Names:
    - Label 0: Healthy
    - Label 1: Parkinson's

 Dataset Info:
    - Number of features: 22
    - Total instances: 195
      - Class 0: 48 instances (Healthy)
      - Class 1: 147 instances (Parkinson's)
```

![Parkinsons](assets/figures/parkinsons.png)

#### Indian Liver Patient

```
Indian Liver Patient — Blood/enzyme test results for liver disease detection — binary

 Feature Names:
    - Feature 0: Age
    - Feature 1: Gender
    - Feature 2: TB
    - Feature 3: DB
    - Feature 4: Alkphos
    - Feature 5: Sgpt
    - Feature 6: Sgot
    - Feature 7: TP
    - Feature 8: ALB
    - Feature 9: A/G Ratio

 Label Names:
    - Label 0: No liver disease
    - Label 1: Liver disease

 Dataset Info:
    - Number of features: 10
    - Total instances: 583
      - Class 0: 167 instances (No liver disease)
      - Class 1: 416 instances (Liver disease)
```

![Indian Liver Patient](assets/figures/indian_liver_patient.png)

#### Cervical Cancer

```
Cervical Cancer — Risk factors and test results for cervical cancer biopsy prediction — binary

 Feature Names:
    - Feature 0: Age
    - Feature 1: Number of sexual partners
    - Feature 2: First sexual intercourse
    - Feature 3: Num of pregnancies
    - Feature 4: Smokes
    - Feature 5: Smokes (years)
    - Feature 6: Smokes (packs/year)
    - Feature 7: Hormonal Contraceptives
    - Feature 8: Hormonal Contraceptives (years)
    - Feature 9: IUD
    - Feature 10: IUD (years)
    - Feature 11: STDs
    - Feature 12: STDs (number)
    - Feature 13: STDs:condylomatosis
    - Feature 14: STDs:cervical condylomatosis
    - Feature 15: STDs:vaginal condylomatosis
    - Feature 16: STDs:vulvo-perineal condylomatosis
    - Feature 17: STDs:syphilis
    - Feature 18: STDs:pelvic inflammatory disease
    - Feature 19: STDs:genital herpes
    - Feature 20: STDs:molluscum contagiosum
    - Feature 21: STDs:AIDS
    - Feature 22: STDs:HIV
    - Feature 23: STDs:Hepatitis B
    - Feature 24: STDs:HPV
    - Feature 25: STDs: Number of diagnosis
    - Feature 26: STDs: Time since first diagnosis
    - Feature 27: STDs: Time since last diagnosis
    - Feature 28: Dx:Cancer
    - Feature 29: Dx:CIN
    - Feature 30: Dx:HPV
    - Feature 31: Dx

 Label Names:
    - Label 0: Healthy
    - Label 1: Cervical cancer

 Dataset Info:
    - Number of features: 32
    - Total instances: 858
      - Class 0: 803 instances (Healthy)
      - Class 1: 55 instances (Cervical cancer)
```

![Cervical Cancer](assets/figures/cervical_cancer.png)

#### Arrhythmia

```
Arrhythmia — ECG-derived features across cardiac arrhythmia classes (279 features)

 Label Names:
    - Label 0: Normal
    - Label 1: Arrhythmia

 Dataset Info:
    - Number of features: 279
    - Total instances: 452
      - Class 0: 245 instances (Normal)
      - Class 1: 207 instances (Arrhythmia)
```

![Arrhythmia](assets/figures/arrhythmia.png)

#### Thyroid Sick

```
Thyroid Sick — Clinical and lab measurements for thyroid disorder detection — binary

 Feature Names:
    - Feature 0: age
    - Feature 1: sex
    - Feature 2: on_thyroxine
    - Feature 3: query_on_thyroxine
    - Feature 4: on_antithyroid_medication
    - Feature 5: sick
    - Feature 6: pregnant
    - Feature 7: thyroid_surgery
    - Feature 8: I131_treatment
    - Feature 9: query_hypothyroid
    - Feature 10: query_hyperthyroid
    - Feature 11: lithium
    - Feature 12: goitre
    - Feature 13: tumor
    - Feature 14: hypopituitary
    - Feature 15: psych
    - Feature 16: TSH_measured
    - Feature 17: TSH
    - Feature 18: T3_measured
    - Feature 19: T3
    - Feature 20: TT4_measured
    - Feature 21: TT4
    - Feature 22: T4U_measured
    - Feature 23: T4U
    - Feature 24: FTI_measured
    - Feature 25: FTI
    - Feature 26: TBG_measured
    - Feature 27: referral_source

 Label Names:
    - Label 0: negative
    - Label 1: sick

 Dataset Info:
    - Number of features: 28
    - Total instances: 2800
      - Class 0: 2629 instances (negative)
      - Class 1: 171 instances (sick)
```

![Thyroid Sick](assets/figures/thyroid_sick.png)

#### Stroke Prediction

```
Stroke — Patient clinical and demographic data for stroke prediction — binary

 Feature Names:
    - Feature 0: gender
    - Feature 1: age
    - Feature 2: hypertension
    - Feature 3: heart_disease
    - Feature 4: ever_married
    - Feature 5: work_type
    - Feature 6: Residence_type
    - Feature 7: avg_glucose_level
    - Feature 8: bmi
    - Feature 9: smoking_status

 Label Names:
    - Label 0: No stroke
    - Label 1: Stroke

 Dataset Info:
    - Number of features: 10
    - Total instances: 5110
      - Class 0: 4861 instances (No stroke)
      - Class 1: 249 instances (Stroke)
```

![Stroke Prediction](assets/figures/stroke_prediction.png)

#### Framingham CHD

```
Framingham — 10-year coronary heart disease risk — Framingham Heart Study, binary

 Feature Names:
    - Feature 0: male
    - Feature 1: age
    - Feature 2: education
    - Feature 3: currentSmoker
    - Feature 4: cigsPerDay
    - Feature 5: BPMeds
    - Feature 6: prevalentStroke
    - Feature 7: prevalentHyp
    - Feature 8: diabetes
    - Feature 9: totChol
    - Feature 10: sysBP
    - Feature 11: diaBP
    - Feature 12: BMI
    - Feature 13: heartRate
    - Feature 14: glucose

 Label Names:
    - Label 0: No CHD
    - Label 1: CHD within 10yr

 Dataset Info:
    - Number of features: 15
    - Total instances: 4240
      - Class 0: 3596 instances (No CHD)
      - Class 1: 644 instances (CHD within 10yr)
```

![Framingham CHD](assets/figures/framingham_chd.png)

#### Thoracic Surgery

```
Thoracic Surgery — Pre-op features for post-thoracotomy survival prediction — binary

 Feature Names:
    - Feature 0: DGN
    - Feature 1: PRE4
    - Feature 2: PRE5
    - Feature 3: PRE6
    - Feature 4: PRE7
    - Feature 5: PRE8
    - Feature 6: PRE9
    - Feature 7: PRE10
    - Feature 8: PRE11
    - Feature 9: PRE14
    - Feature 10: PRE17
    - Feature 11: PRE19
    - Feature 12: PRE25
    - Feature 13: PRE30
    - Feature 14: PRE32
    - Feature 15: AGE

 Label Names:
    - Label 0: Survived
    - Label 1: Died within 1yr

 Dataset Info:
    - Number of features: 16
    - Total instances: 470
      - Class 0: 400 instances (Survived)
      - Class 1: 70 instances (Died within 1yr)
```

![Thoracic Surgery](assets/figures/thoracic_surgery.png)

#### SPECTF Heart

```
SPECTF Heart — SPECT cardiac imaging features for heart disease diagnosis — binary

 Feature Names:
    - Feature 0: F1R
    - Feature 1: F1S
    - Feature 2: F2R
    - Feature 3: F2S
    - Feature 4: F3R
    - Feature 5: F3S
    - Feature 6: F4R
    - Feature 7: F4S
    - Feature 8: F5R
    - Feature 9: F5S
    - Feature 10: F6R
    - Feature 11: F6S
    - Feature 12: F7R
    - Feature 13: F7S
    - Feature 14: F8R
    - Feature 15: F8S
    - Feature 16: F9R
    - Feature 17: F9S
    - Feature 18: F10R
    - Feature 19: F10S
    - Feature 20: F11R
    - Feature 21: F11S
    - Feature 22: F12R
    - Feature 23: F12S
    - Feature 24: F13R
    - Feature 25: F13S
    - Feature 26: F14R
    - Feature 27: F14S
    - Feature 28: F15R
    - Feature 29: F15S
    - Feature 30: F16R
    - Feature 31: F16S
    - Feature 32: F17R
    - Feature 33: F17S
    - Feature 34: F18R
    - Feature 35: F18S
    - Feature 36: F19R
    - Feature 37: F19S
    - Feature 38: F20R
    - Feature 39: F20S
    - Feature 40: F21R
    - Feature 41: F21S
    - Feature 42: F22R
    - Feature 43: F22S

 Label Names:
    - Label 0: Normal
    - Label 1: Abnormal

 Dataset Info:
    - Number of features: 44
    - Total instances: 267
      - Class 0: 55 instances (Normal)
      - Class 1: 212 instances (Abnormal)
```

![SPECTF Heart](assets/figures/spectf_heart.png)

#### Heart Failure

```
Heart Failure — Clinical records for heart failure event survival prediction — binary

 Feature Names:
    - Feature 0: age
    - Feature 1: anaemia
    - Feature 2: creatinine_phosphokinase
    - Feature 3: diabetes
    - Feature 4: ejection_fraction
    - Feature 5: high_blood_pressure
    - Feature 6: platelets
    - Feature 7: serum_creatinine
    - Feature 8: serum_sodium
    - Feature 9: sex
    - Feature 10: smoking
    - Feature 11: time

 Label Names:
    - Label 0: Survived
    - Label 1: Died

 Dataset Info:
    - Number of features: 12
    - Total instances: 299
      - Class 0: 203 instances (Survived)
      - Class 1: 96 instances (Died)
```

![Heart Failure](assets/figures/heart_failure.png)

#### Mammographic Mass

```
Mammographic Mass — Mammography BI-RADS attributes and patient age for mass malignancy — binary

 Feature Names:
    - Feature 0: BI-RADS
    - Feature 1: Age
    - Feature 2: Shape
    - Feature 3: Margin
    - Feature 4: Density

 Label Names:
    - Label 0: Benign
    - Label 1: Malignant

 Dataset Info:
    - Number of features: 5
    - Total instances: 961
      - Class 0: 516 instances (Benign)
      - Class 1: 445 instances (Malignant)
```

![Mammographic Mass](assets/figures/mammographic_mass.png)

#### Breast Cancer Prognostic

```
Breast Cancer Prognostic — FNA features for breast cancer recurrence prediction — binary

 Feature Names:
    - Feature 0: Time
    - Feature 1: radius1
    - Feature 2: texture1
    - Feature 3: perimeter1
    - Feature 4: area1
    - Feature 5: smoothness1
    - Feature 6: compactness1
    - Feature 7: concavity1
    - Feature 8: concave_points1
    - Feature 9: symmetry1
    - Feature 10: fractal_dimension1
    - Feature 11: radius2
    - Feature 12: texture2
    - Feature 13: perimeter2
    - Feature 14: area2
    - Feature 15: smoothness2
    - Feature 16: compactness2
    - Feature 17: concavity2
    - Feature 18: concave_points2
    - Feature 19: symmetry2
    - Feature 20: fractal_dimension2
    - Feature 21: radius3
    - Feature 22: texture3
    - Feature 23: perimeter3
    - Feature 24: area3
    - Feature 25: smoothness3
    - Feature 26: compactness3
    - Feature 27: concavity3
    - Feature 28: concave_points3
    - Feature 29: symmetry3
    - Feature 30: fractal_dimension3
    - Feature 31: tumor_size
    - Feature 32: lymph_node_status

 Label Names:
    - Label 0: Non-recurrent
    - Label 1: Recurrent

 Dataset Info:
    - Number of features: 33
    - Total instances: 198
      - Class 0: 151 instances (Non-recurrent)
      - Class 1: 47 instances (Recurrent)
```

![Breast Cancer Prognostic](assets/figures/breast_cancer_prognostic.png)

#### Breast Cancer Coimbra

```
Breast Cancer Coimbra — Anthropometric and blood biomarkers for breast cancer diagnosis — binary

 Feature Names:
    - Feature 0: Age
    - Feature 1: BMI
    - Feature 2: Glucose
    - Feature 3: Insulin
    - Feature 4: HOMA
    - Feature 5: Leptin
    - Feature 6: Adiponectin
    - Feature 7: Resistin
    - Feature 8: MCP.1

 Label Names:
    - Label 0: Healthy control
    - Label 1: Patient

 Dataset Info:
    - Number of features: 9
    - Total instances: 116
      - Class 0: 52 instances (Healthy control)
      - Class 1: 64 instances (Patient)
```

![Breast Cancer Coimbra](assets/figures/breast_cancer_coimbra.png)

#### HCC Survival

```
HCC Survival — Hepatocellular carcinoma patient survival from clinical features — binary

 Feature Names:
    - Feature 0: Gender
    - Feature 1: Symptoms
    - Feature 2: Alcohol
    - Feature 3: HBsAg
    - Feature 4: HBeAg
    - Feature 5: HBcAb
    - Feature 6: HCVAb
    - Feature 7: Cirrhosis
    - Feature 8: Endemic
    - Feature 9: Smoking
    - Feature 10: Diabetes
    - Feature 11: Obesity
    - Feature 12: Hemochromatosis
    - Feature 13: AHT
    - Feature 14: CRI
    - Feature 15: HIV
    - Feature 16: NASH
    - Feature 17: Varices
    - Feature 18: Splenomegaly
    - Feature 19: PHT
    - Feature 20: PVT
    - Feature 21: Metastasis
    - Feature 22: Hallmark
    - Feature 23: Age
    - Feature 24: Grams_per_day
    - Feature 25: Packs_per_year
    - Feature 26: PS
    - Feature 27: Encephalopathy
    - Feature 28: Ascites
    - Feature 29: INR
    - Feature 30: AFP
    - Feature 31: Hemoglobin
    - Feature 32: MCV
    - Feature 33: Leucocytes
    - Feature 34: Platelets
    - Feature 35: Albumin
    - Feature 36: Total_Bil
    - Feature 37: ALT
    - Feature 38: AST
    - Feature 39: GGT
    - Feature 40: ALP
    - Feature 41: TP
    - Feature 42: Creatinine
    - Feature 43: Nodules
    - Feature 44: Major_Dim
    - Feature 45: Dir_Bil
    - Feature 46: Iron
    - Feature 47: Sat
    - Feature 48: Ferritin

 Label Names:
    - Label 0: Lives
    - Label 1: Dies

 Dataset Info:
    - Number of features: 49
    - Total instances: 165
      - Class 0: 102 instances (Lives)
      - Class 1: 63 instances (Dies)
```

![HCC Survival](assets/figures/hcc_survival.png)

#### Z-Alizadeh Sani CAD

```
Z-Alizadeh Sani CAD — Coronary artery disease diagnosis from clinical features — binary

 Feature Names:
    - Feature 0: Age
    - Feature 1: Weight
    - Feature 2: Length
    - Feature 3: Sex
    - Feature 4: BMI
    - Feature 5: DM
    - Feature 6: HTN
    - Feature 7: Current Smoker
    - Feature 8: EX-Smoker
    - Feature 9: FH
    - Feature 10: Obesity
    - Feature 11: CRF
    - Feature 12: CVA
    - Feature 13: Airway disease
    - Feature 14: Thyroid Disease
    - Feature 15: CHF
    - Feature 16: DLP
    - Feature 17: BP
    - Feature 18: PR
    - Feature 19: Edema
    - Feature 20: Weak Peripheral Pulse
    - Feature 21: Lung rales
    - Feature 22: Systolic Murmur
    - Feature 23: Diastolic Murmur
    - Feature 24: Typical Chest Pain
    - Feature 25: Dyspnea
    - Feature 26: Function Class
    - Feature 27: Atypical
    - Feature 28: Nonanginal
    - Feature 29: Exertional CP
    - Feature 30: LowTH Ang
    - Feature 31: Q Wave
    - Feature 32: St Elevation
    - Feature 33: St Depression
    - Feature 34: Tinversion
    - Feature 35: LVH
    - Feature 36: Poor R Progression
    - Feature 37: BBB
    - Feature 38: FBS
    - Feature 39: CR
    - Feature 40: TG
    - Feature 41: LDL
    - Feature 42: HDL
    - Feature 43: BUN
    - Feature 44: ESR
    - Feature 45: HB
    - Feature 46: K
    - Feature 47: Na
    - Feature 48: WBC
    - Feature 49: Lymph
    - Feature 50: Neut
    - Feature 51: PLT
    - Feature 52: EF-TTE
    - Feature 53: Region RWMA
    - Feature 54: VHD

 Label Names:
    - Label 0: Normal
    - Label 1: CAD

 Dataset Info:
    - Number of features: 55
    - Total instances: 303
      - Class 0: 87 instances (Normal)
      - Class 1: 216 instances (CAD)
```

![Z-Alizadeh Sani CAD](assets/figures/z-alizadeh_sani_cad.png)

</details>

<details>
<summary><strong>Other</strong></summary>

#### Banknote Authentication

```
Banknote Authentication — Wavelet-transform features from banknote images — binary authentic/forged

 Feature Names:
    - Feature 0: variance of Wavelet Transformed image
    - Feature 1: skewness of Wavelet Transformed image
    - Feature 2: curtosis of Wavelet Transformed image
    - Feature 3: entropy of image

 Label Names:
    - Label 0: Authentic
    - Label 1: Counterfeit

 Dataset Info:
    - Number of features: 4
    - Total instances: 1372
      - Class 0: 762 instances (Authentic)
      - Class 1: 610 instances (Counterfeit)
```

![Banknote Authentication](assets/figures/banknote_authentication.png)

#### Wheat Seeds

```
Wheat Seeds — Geometric grain measurements for wheat variety classification — 3 classes

 Feature Names:
    - Feature 0: area
    - Feature 1: perimeter
    - Feature 2: compactness
    - Feature 3: length of kernel
    - Feature 4: width of kernel
    - Feature 5: asymmetry coefficient
    - Feature 6: length of kernel groove

 Label Names:
    - Label 0: Rosa or Canadian
    - Label 1: Kama

 Dataset Info:
    - Number of features: 7
    - Total instances: 210
      - Class 0: 140 instances (Rosa or Canadian)
      - Class 1: 70 instances (Kama)
```

![Wheat Seeds](assets/figures/wheat_seeds.png)

#### Ionosphere

```
Ionosphere — Radar signal classification for ionosphere quality — binary

 Feature Names:
    - Feature 0: Pulse 1 real
    - Feature 1: Pulse 1 imaginary
    - Feature 2: Pulse 2 real
    - Feature 3: Pulse 2 imaginary
    - Feature 4: Pulse 3 real
    - Feature 5: Pulse 3 imaginary
    - Feature 6: Pulse 4 real
    - Feature 7: Pulse 4 imaginary
    - Feature 8: Pulse 5 real
    - Feature 9: Pulse 5 imaginary
    - Feature 10: Pulse 6 real
    - Feature 11: Pulse 6 imaginary
    - Feature 12: Pulse 7 real
    - Feature 13: Pulse 7 imaginary
    - Feature 14: Pulse 8 real
    - Feature 15: Pulse 8 imaginary
    - Feature 16: Pulse 9 real
    - Feature 17: Pulse 9 imaginary
    - Feature 18: Pulse 10 real
    - Feature 19: Pulse 10 imaginary
    - Feature 20: Pulse 11 real
    - Feature 21: Pulse 11 imaginary
    - Feature 22: Pulse 12 real
    - Feature 23: Pulse 12 imaginary
    - Feature 24: Pulse 13 real
    - Feature 25: Pulse 13 imaginary
    - Feature 26: Pulse 14 real
    - Feature 27: Pulse 14 imaginary
    - Feature 28: Pulse 15 real
    - Feature 29: Pulse 15 imaginary
    - Feature 30: Pulse 16 real
    - Feature 31: Pulse 16 imaginary
    - Feature 32: Pulse 17 real
    - Feature 33: Pulse 17 imaginary

 Label Names:
    - Label 0: bad
    - Label 1: good

 Dataset Info:
    - Number of features: 34
    - Total instances: 351
      - Class 0: 126 instances (bad)
      - Class 1: 225 instances (good)
```

![Ionosphere](assets/figures/ionosphere.png)

#### Sonar Rocks vs Mines

```
Sonar Rocks vs Mines — Sonar frequency-response features for rock/mine classification — binary

 Feature Names:
    - Feature 0: 0
    - Feature 1: 1
    - Feature 2: 2
    - Feature 3: 3
    - Feature 4: 4
    - Feature 5: 5
    - Feature 6: 6
    - Feature 7: 7
    - Feature 8: 8
    - Feature 9: 9
    - Feature 10: 10
    - Feature 11: 11
    - Feature 12: 12
    - Feature 13: 13
    - Feature 14: 14
    - Feature 15: 15
    - Feature 16: 16
    - Feature 17: 17
    - Feature 18: 18
    - Feature 19: 19
    - Feature 20: 20
    - Feature 21: 21
    - Feature 22: 22
    - Feature 23: 23
    - Feature 24: 24
    - Feature 25: 25
    - Feature 26: 26
    - Feature 27: 27
    - Feature 28: 28
    - Feature 29: 29
    - Feature 30: 30
    - Feature 31: 31
    - Feature 32: 32
    - Feature 33: 33
    - Feature 34: 34
    - Feature 35: 35
    - Feature 36: 36
    - Feature 37: 37
    - Feature 38: 38
    - Feature 39: 39
    - Feature 40: 40
    - Feature 41: 41
    - Feature 42: 42
    - Feature 43: 43
    - Feature 44: 44
    - Feature 45: 45
    - Feature 46: 46
    - Feature 47: 47
    - Feature 48: 48
    - Feature 49: 49
    - Feature 50: 50
    - Feature 51: 51
    - Feature 52: 52
    - Feature 53: 53
    - Feature 54: 54
    - Feature 55: 55
    - Feature 56: 56
    - Feature 57: 57
    - Feature 58: 58
    - Feature 59: 59

 Label Names:
    - Label 0: Rock
    - Label 1: Mine

 Dataset Info:
    - Number of features: 60
    - Total instances: 208
      - Class 0: 97 instances (Rock)
      - Class 1: 111 instances (Mine)
```

![Sonar Rocks vs Mines](assets/figures/sonar_rocks_vs_mines.png)

#### Abalone Gender

```
Abalone Gender — Physical measurements for abalone sex classification — 3 classes

 Feature Names:
    - Feature 0: Length
    - Feature 1: Diameter
    - Feature 2: Height
    - Feature 3: Whole weight
    - Feature 4: Shucked weight
    - Feature 5: Viscera weight
    - Feature 6: Shell weight
    - Feature 7: Rings

 Label Names:
    - Label 0: Male
    - Label 1: Female

 Dataset Info:
    - Number of features: 8
    - Total instances: 2835
      - Class 0: 1528 instances (Male)
      - Class 1: 1307 instances (Female)
```

![Abalone Gender](assets/figures/abalone_gender.png)

</details>

<details>
<summary><strong>Image</strong></summary>

#### MNIST

```
MNIST — Handwritten digit recognition — 28×28 greyscale, 10 classes

 Label Names:
    - Label 0: Other digits
    - Label 1: Digit 0

 Dataset Info:
    - Number of features: 784
    - Total instances: 3000
      - Class 0: 2715 instances (Other digits)
      - Class 1: 285 instances (Digit 0)
```

![MNIST](assets/figures/mnist.png)

#### Fashion-MNIST

```
Fashion-MNIST — Clothing item recognition — 28×28 greyscale, 10 categories (Zalando)

 Label Names:
    - Label 0: Other classes
    - Label 1: T-shirt/top

 Dataset Info:
    - Number of features: 784
    - Total instances: 3000
      - Class 0: 2718 instances (Other classes)
      - Class 1: 282 instances (T-shirt/top)
```

![Fashion-MNIST](assets/figures/fashion-mnist.png)

#### SVHN

```
SVHN — Real-world house-number digit recognition — 32×32 RGB (Google Street View)

 Label Names:
    - Label 0: Other digits
    - Label 1: 0

 Dataset Info:
    - Number of features: 3072
    - Total instances: 3000
      - Class 0: 2806 instances (Other digits)
      - Class 1: 194 instances (0)
```

![SVHN](assets/figures/svhn.png)

#### EuroSAT

```
EuroSAT — Satellite land-use/cover classification — 64×64 RGB, 10 classes

 Label Names:
    - Label 0: Other classes
    - Label 1: Highway

 Dataset Info:
    - Number of features: 12288
    - Total instances: 3000
      - Class 0: 2694 instances (Other classes)
      - Class 1: 306 instances (Highway)
```

![EuroSAT](assets/figures/eurosat.png)

#### CIFAR-10

```
CIFAR-10 — Object recognition — 32×32 RGB, 10 classes (Krizhevsky 2009)

 Label Names:
    - Label 0: Other classes
    - Label 1: airplane

 Dataset Info:
    - Number of features: 3072
    - Total instances: 3000
      - Class 0: 2701 instances (Other classes)
      - Class 1: 299 instances (airplane)
```

![CIFAR-10](assets/figures/cifar-10.png)

#### CIFAR-100

```
CIFAR-100 — Fine-grained object recognition — 32×32 RGB, 100 classes

 Label Names:
    - Label 0: Other classes
    - Label 1: apple

 Dataset Info:
    - Number of features: 3072
    - Total instances: 3000
      - Class 0: 2976 instances (Other classes)
      - Class 1: 24 instances (apple)
```

![CIFAR-100](assets/figures/cifar-100.png)

</details>

<details>
<summary><strong>Medical Image (MedMNIST)</strong></summary>

#### PneumoniaMNIST

```
PneumoniaMNIST — Chest X-ray images for pneumonia detection — 28×28 greyscale, binary

 Label Names:
    - Label 0: Other classes
    - Label 1: pneumonia

 Dataset Info:
    - Number of features: 784
    - Total instances: 2824
      - Class 0: 728 instances (Other classes)
      - Class 1: 2096 instances (pneumonia)
```

![PneumoniaMNIST](assets/figures/pneumoniamnist.png)

#### BreastMNIST

```
BreastMNIST — Breast ultrasound images for malignancy classification — 28×28 greyscale, binary

 Label Names:
    - Label 0: Other classes
    - Label 1: malignant

 Dataset Info:
    - Number of features: 784
    - Total instances: 546
      - Class 0: 399 instances (Other classes)
      - Class 1: 147 instances (malignant)
```

![BreastMNIST](assets/figures/breastmnist.png)

#### DermaMNIST

```
DermaMNIST — Dermatoscopic skin lesion images — 28×28 RGB, 7 classes

 Label Names:
    - Label 0: Other classes
    - Label 1: melanoma

 Dataset Info:
    - Number of features: 2352
    - Total instances: 2802
      - Class 0: 2491 instances (Other classes)
      - Class 1: 311 instances (melanoma)
```

![DermaMNIST](assets/figures/dermamnist.png)

#### BloodMNIST

```
BloodMNIST — Peripheral blood cell microscopy images — 28×28 RGB, 8 classes

 Label Names:
    - Label 0: Other classes
    - Label 1: basophil

 Dataset Info:
    - Number of features: 2352
    - Total instances: 2989
      - Class 0: 2776 instances (Other classes)
      - Class 1: 213 instances (basophil)
```

![BloodMNIST](assets/figures/bloodmnist.png)

#### PathMNIST

```
PathMNIST — Colorectal cancer histology tissue patches — 28×28 RGB, 9 classes

 Label Names:
    - Label 0: Other classes
    - Label 1: colorectal adenocarcinoma epithelium

 Dataset Info:
    - Number of features: 2352
    - Total instances: 2699
      - Class 0: 2313 instances (Other classes)
      - Class 1: 386 instances (colorectal adenocarcinoma epithelium)
```

![PathMNIST](assets/figures/pathmnist.png)

#### OCTMNIST

```
OCTMNIST — Retinal optical coherence tomography images — 28×28 greyscale, 4 classes

 Label Names:
    - Label 0: Other classes
    - Label 1: choroidal neovascularization

 Dataset Info:
    - Number of features: 784
    - Total instances: 2923
      - Class 0: 1919 instances (Other classes)
      - Class 1: 1004 instances (choroidal neovascularization)
```

![OCTMNIST](assets/figures/octmnist.png)

</details>

<details>
<summary><strong>Loader options — visual demos</strong></summary>

### `get_train_test_split(train_size=...)`

Splits data into train and test sets while preserving class proportions.
`train_size` controls the fraction used for training (default `0.5`).

```python
train, test = get_dataset('Moons', train_size=0.7).get_train_test_split()
```

![train_test_split](assets/figures/options/train_test_split.png)

---

### `get_cross_validation_folds(n_splits=...)`

Folds the *train* split into `n_splits` stratified cross validation folds and
returns them alongside the untouched test split, so the test set stays a clean
estimate of generalisation. Class ratios are preserved in every fold — on
imbalanced data an unstratified fold can easily end up with no minority
instances at all. Any `get_train_test_split` option can be passed through.

```python
loader = get_dataset('Habermans Breast Cancer', train_size=0.7)
folds, test = loader.get_cross_validation_folds(n_splits=5)

for train_fold, val_fold in folds:
    clf.fit(train_fold['X'], train_fold['y'])
    score(clf, val_fold['X'], val_fold['y'])
```

Use `utils.stratified_kfold_indices(y, n_splits)` instead when you need the raw
index arrays — for example to resample the train fold only, without leaking
resampled rows into validation.

---

### `scale=True`

Applies MinMax normalisation fitted on the train set, scaling all features to `[−1, 1]`.
Shown here on the Diabetes Pima Indian dataset — note the axis ranges before and after.

```python
dataset = get_dataset('Diabetes Pima Indian', scale=True)
train, test = dataset.get_train_test_split()
```

![scale](assets/figures/options/scale.png)

---

### `percent_of_data`

Subsamples the full dataset to the given percentage while preserving class proportions.

```python
dataset = get_dataset('Moons', percent_of_data=50)  # keep 50% of data
```

![percent_of_data](assets/figures/options/percent_of_data.png)

---

### `minority_reduce_scaler`

Reduces the minority class in the **train** split by the given factor,
creating a class-imbalanced training set (useful for cost-sensitive learning).

```python
dataset = get_dataset('Moons', minority_reduce_scaler=2)
train, test = dataset.get_train_test_split()
```

![minority_reduce](assets/figures/options/minority_reduce.png)

---

### `majority_max`

Caps the majority class of the **train** split at a fixed number of instances.
The test split is untouched.

Use it when a dataset is too large to fit every model family on but you want
to keep its natural imbalance — capping the majority is much better than
subsampling both classes, which would throw away the minority data the
imbalance story depends on.

```python
# MIMIC-IV has 1.6M majority rows: fine for a linear model, hopeless for an
# O(n^2) kernel SVM. Cap the training majority and keep every minority point.
dataset = get_dataset('MIMIC-IV Ready for Discharge', majority_max=50_000)
train, test = dataset.get_train_test_split()
```

Applied **before** `minority_reduce_scaler`, so a requested imbalance ratio is
taken against the capped count rather than the original one:

```python
# 100 majority / 10 minority, not 100 / (original_majority / 10)
train, test = dataset.get_train_test_split(
    majority_max=100, minority_reduce_scaler=10)
```

---

### `dim_reducer`

Applies dimensionality reduction to the split output.
Supported methods: `PCA`, `kernelPCA`, `TSNE`, `UMAP`, `UMAP_supervised`.
Shown on the Wine dataset (13 features projected to 2D).

```python
dataset = get_dataset('Wine', dim_reducer='UMAP', reduce_to_dim=2)
train, test = dataset.get_train_test_split()
```

![dim_reducer](assets/figures/options/dim_reducer.png)

#### Reusing a reducer across plots

When comparing multiple plots (e.g. the same dataset at different downsampling rates), you need all plots to share the **same** projection. TSNE and UMAP are non-deterministic and fit-dependent, so a fresh fit per plot produces incomparable coordinate spaces.

**Option A — fit explicitly, then pass to each plot:**

```python
from data_loaders import get_dataset
from data_loaders.plotting.visualisation import plot_dataset

loader = get_dataset('Moons')

# Fit once on the full dataset
reducer = loader.fit_dim_reducer('TSNE')

# Reuse the same projection for every plot
for pct in [100, 50, 25]:
    sub = get_dataset('Moons', percent_of_data=pct)
    data = sub.get_data_dict()
    plot_dataset(data['X'], data['y'],
                 dim_reducer=reducer,
                 dataset_name=f'Moons {pct}%')
```

**Option B — extract the reducer after the first plot:**

```python
loader = get_dataset('Moons')

# First plot fits and stores the reducer automatically
fig, ax = loader.plot_dataset(dim_reducer_method='TSNE')
reducer = loader.last_dim_reducer   # retrieve the fitted reducer

# Pass it to subsequent plots so they share the same projection
loader2 = get_dataset('Moons', percent_of_data=50)
loader2.plot_dataset(dim_reducer=reducer)

# Works with plot_train_test_split too
loader2.plot_train_test_split(dim_reducer=reducer)
```

The `dim_reducer` parameter is accepted by:
- `plot_dataset()` in `data_loaders.plotting.visualisation`
- `loader.plot_dataset()`
- `loader.plot_train_test_split()`

---

### `clf` — classifier decision boundary

Pass a trained classifier to overlay its decision boundary and highlight misclassified
points (marked with **×**). The classifier must accept inputs in the **original feature
space**. Decision boundary regions are drawn when the data is already 2D (no dim
reduction needed); for high-dim data, only misclassification markers are shown.

```python
from sklearn.linear_model import LogisticRegression

train, test = get_dataset("Moons").get_train_test_split()
clf = LogisticRegression().fit(train["X"], train["y"])

# Pass clf to see boundary + misclassification markers
plot_dataset(train["X"], train["y"], clf=clf.predict)

# Or use pre-computed predictions (no boundary, just markers)
plot_dataset(train["X"], train["y"], y_pred=clf.predict(train["X"]))
```

![clf](assets/figures/options/clf.png)

---

### `overlay_train_test` — train and test on one plot

By default, train and test are shown in separate side-by-side subplots.
Set `overlay_train_test=True` to plot both on one axes: train as filled circles,
test as open circles at `test_alpha` opacity (default `0.3`).
Combine with `clf` to show the decision boundary and misclassifications together.

```python
train, test = get_dataset("Moons").get_train_test_split()
clf = LogisticRegression().fit(train["X"], train["y"])

# Overlay with custom test transparency and classifier boundary
plot_dataset(train["X"], train["y"], X_test=test["X"], y_test=test["y"],
             overlay_train_test=True, test_alpha=0.4, clf=clf.predict)

# Also available on AbstractLoader:
loader.plot_train_test_split(overlay_train_test=True, clf=clf.predict)
```

![overlay_train_test](assets/figures/options/overlay_train_test.png)

---

### Image datasets — example images per class

Image loaders (MNIST, Fashion-MNIST, SVHN, EuroSAT, CIFAR-10/100/10N and the
MedMNIST sets) flag themselves with the `is_image` class attribute and know how
to reshape a flattened sample back into a viewable image (`image_shape`,
`channels_first`). This powers a `plot_class_samples()` preview that shows a few
example images per class — handy for seeing what the raw data actually looks
like, alongside the 2D projection in the gallery above.

```python
loader = get_dataset('CIFAR-10')

# Grid of example images, one row per class
loader.plot_class_samples(n_per_class=5)

# Or reshape a single flat sample yourself for custom plots
img = loader.as_image(loader.get_X()[0])   # -> (32, 32, 3) RGB array
```

Non-image loaders have `is_image = False`; calling `as_image()` on them raises
a `ValueError`.

</details>

<details>
<summary><strong>Classifier benchmarks</strong></summary>

Balanced accuracy (test set) for 5 sklearn classifiers across all non-image datasets.
Train/test split: 50/50 (default). Seed: 42.

Classifiers: Logistic Regression, Random Forest, SVC (RBF), KNN, Gaussian NB.

### Summary heatmap

![benchmark_heatmap](assets/figures/benchmark_heatmap.png)

### Synthetic

![benchmark_synthetic](assets/figures/benchmark_synthetic.png)

![benchmark_clf_plots_synthetic](assets/figures/benchmark_clf_plots_synthetic.png)

### Classic

![benchmark_classic](assets/figures/benchmark_classic.png)

![benchmark_clf_plots_classic](assets/figures/benchmark_clf_plots_classic.png)

### Medical

![benchmark_medical](assets/figures/benchmark_medical.png)

![benchmark_clf_plots_medical](assets/figures/benchmark_clf_plots_medical.png)

### Other

![benchmark_other](assets/figures/benchmark_other.png)

![benchmark_clf_plots_other](assets/figures/benchmark_clf_plots_other.png)

</details>

## Installation

Requires Python 3.11+. Uses uv for dependency management.

```bash
# Clone and install
git clone https://github.com/your-username/toy_datasets.git
cd toy_datasets
uv sync

# Or install with dev dependencies
uv sync --group dev
```

## Quick Start

```python
import data_loaders

# Load a dataset
loader = data_loaders.get_dataset('Iris')

# Get features and labels
X = loader.get_X()
y = loader.get_y()

# Get train/test split (preserves class proportions)
train, test = loader.get_train_test_split()

# With options
loader = data_loaders.get_dataset(
    'Moons',
    scale=True,           # Normalize to [-1, 1]
    train_size=0.8,       # 80% train, 20% test
    dim_reducer='PCA',    # Apply PCA
    reduce_to_dim=2       # Reduce to 2 dimensions
)
```

## Loader API

All loaders inherit from `AbstractLoader` and provide:

```python
loader.get_X()                  # Feature array (numpy)
loader.get_y()                  # Label array (numpy)
loader.get_train_test_split()   # Returns (train_dict, test_dict)
loader.get_cross_validation_folds(n_splits=5)
                                # Returns ([(train_dict, val_dict), ...], test_dict)
loader.get_description()        # Dataset description
loader.get_feature_names()      # List of feature names
loader.get_label_names()        # List of class names
loader.get_info()               # Full dataset info string

# Visualize the full dataset (supports clf, y_pred)
loader.plot_dataset()
# Visualize train/test split (supports clf, y_pred, y_pred_test,
#                             overlay_train_test, test_alpha)
loader.plot_train_test_split()
```

## Common Options

```python
loader = data_loaders.get_dataset(
    'Iris',
    shuffle=True,            # Shuffle data (default: True)
    set_seed=42,             # Random seed for reproducibility
    train_size=0.5,          # Train set proportion (default: 0.5)
                             # The rest is used for the test set
    minority_reduce_scaler=2,# Reduce minority class in train set to 1/2 
                             # (default: None, no reduction)
    minority_reduce_scaler_test=2,# Reduce minority class in test set to 1/2 
                             # (default: None, no reduction)    
    equal_test=False,        # Whether to make the test set 
                             # perfectly balanced (overrides 
                             # minority_reduce_scaler_test)                   
    scale=True,              # Apply MinMax scaling
    percent_of_data=50,      # Use only 50% of data
    equal_test=True,         # Balance test set classes
    dim_reducer='PCA',       # 'PCA', 'UMAP', 'TSNE', 'kernelPCA'
    reduce_to_dim=2,         # Target dimensions
)
```

## Testing

```bash
uv run pytest                     # Run all tests
uv run pytest -m "not slow"       # Skip slow tests (MNIST, t-SNE)
uv run pytest --cov=data_loaders  # With coverage report
```

## Sub-packages

Each sub-package has its own README with full details.

| Sub-package | Description |
|---|---|
| [`data_loaders/utils/`](data_loaders/utils/README.md) | Normalization, shuffling, seeding, train/test splitting |
| [`data_loaders/resampling/`](data_loaders/resampling/README.md) | Upsampling, SMOTE, and downsampling for class imbalance |
| [`data_loaders/embeddings/`](data_loaders/embeddings/README.md) | PCA, kernel PCA, t-SNE, UMAP dimensionality reduction |
| [`data_loaders/plotting/`](data_loaders/plotting/README.md) | Dataset visualisation and terminal rendering |
| [`data_loaders/loaders/synthetic_generators/`](data_loaders/loaders/synthetic_generators/README.md) | XOR, Moons, Blobs, Circles, Gaussian generators |
| [`data_loaders/loaders/web_loaders/`](data_loaders/loaders/web_loaders/README.md) | Iris, Wine, Breast Cancer, Heart Disease, Parkinsons, Indian Liver, Arrhythmia, MNIST, Fashion-MNIST, SVHN, EuroSAT, CIFAR-10/100/10N, MedMNIST |
| [`data_loaders/loaders/local_loaders/`](data_loaders/loaders/local_loaders/readme.md) | CSV-backed loaders (diabetes, banknote, cervical cancer, costcla, etc.) |
| [`data_loaders/loaders/external_loaders/`](data_loaders/loaders/external_loaders/README.md) | MIMIC-III/IV medical datasets (require special access) |

## Project Structure

```
toy_datasets/
├── data_loaders/
│   ├── main.py                   # Registry and get_dataset()
│   ├── utils/                    # Normalization, splitting utilities
│   ├── resampling/               # Upsampling and downsampling
│   ├── embeddings/               # PCA, UMAP, t-SNE wrappers
│   ├── plotting/                 # Visualisation and terminal rendering
│   └── loaders/
│       ├── abstract_loader.py    # Base class for all loaders
│       ├── synthetic_generators/ # XOR, Moons, Blobs, etc.
│       ├── web_loaders/          # Iris, Wine, MNIST, CIFAR-10/100/10N, etc.
│       ├── local_loaders/        # CSV-based datasets
│       └── external_loaders/     # MIMIC (requires access)
├── data_loaders/datasets/        # Bundled CSV files
├── tests/                        # pytest test suite
└── pyproject.toml                # Project configuration
```

## License

MIT
