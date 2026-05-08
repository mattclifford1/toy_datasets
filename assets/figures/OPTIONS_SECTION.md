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

### `dim_reducer`

Applies dimensionality reduction to the split output.
Supported methods: `PCA`, `kernelPCA`, `TSNE`, `UMAP`, `UMAP_supervised`.
Shown on the Wine dataset (13 features projected to 2D).

```python
dataset = get_dataset('Wine', dim_reducer='UMAP', reduce_to_dim=2)
train, test = dataset.get_train_test_split()
```

![dim_reducer](assets/figures/options/dim_reducer.png)

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

</details>