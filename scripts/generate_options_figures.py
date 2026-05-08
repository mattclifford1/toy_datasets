"""Generate figures demonstrating loader options for the README options section.

Run with:
    uv run python scripts/generate_options_figures.py

Outputs:
    assets/figures/options/<option>.png  - one figure per loader option
    assets/figures/OPTIONS_SECTION.md    - <details> block ready to paste into README.md
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.main import get_dataset
from data_loaders.plotting.visualisation import plot_dataset as _plot_dataset
from data_loaders.utils import Normaliser

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'assets', 'figures', 'options')

COLORS = ['#3ea3e6', '#e56a6a']


def generate_train_test_split_figure() -> str | None:
    """Show train/test split with train_size=0.7 on Moons dataset."""
    print('  Generating train_test_split figure...')
    try:
        loader = get_dataset('Moons', set_seed=42, train_size=0.7)
        train, test = loader.get_train_test_split()
        label_names = loader.get_label_names()
        lnames = label_names if isinstance(label_names, list) else None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('get_train_test_split(train_size=0.7) — Moons dataset', fontsize=13)

        _plot_dataset(
            X=train['X'], y=train['y'],
            X_test=test['X'], y_test=test['y'],
            label_names=lnames,
            dim_reducer_method='PCA',
            ax=(axes[0], axes[1]),
        )

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'train_test_split.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/train_test_split.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def generate_scale_figure() -> str | None:
    """Show MinMax normalization on Diabetes Pima Indians (Pregnancies vs Insulin)."""
    print('  Generating scale figure...')
    try:
        loader = get_dataset('Diabetes Pima Indian', set_seed=42, scale=False, minority_reduce_scaler=None)
        X = loader.get_X()
        y = loader.get_y()
        feature_names = loader.get_feature_names()

        normaliser = Normaliser(X)
        X_scaled = normaliser(X)

        feat_i, feat_j = 0, 4  # Pregnancies vs Insulin — largest scale contrast
        xi = feature_names[feat_i] if isinstance(feature_names, list) else f'Feature {feat_i}'
        xj = feature_names[feat_j] if isinstance(feature_names, list) else f'Feature {feat_j}'

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('scale=True — Diabetes Pima Indians (raw vs normalized to [−1, 1])', fontsize=13)

        for ax, X_plot, title in [
            (axes[0], X, 'Raw features'),
            (axes[1], X_scaled, 'Normalized to [−1, 1]'),
        ]:
            for cls in [0, 1]:
                mask = y == cls
                ax.scatter(
                    X_plot[mask, feat_i], X_plot[mask, feat_j],
                    color=COLORS[cls], alpha=0.7, s=14,
                    label=f'Class {cls}',
                )
            ax.set_xlabel(xi)
            ax.set_ylabel(xj)
            ax.set_title(title)
            ax.legend()

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'scale.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/scale.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def generate_percent_of_data_figure() -> str | None:
    """Show Moons dataset at 100%, 50%, and 20% of data."""
    print('  Generating percent_of_data figure...')
    try:
        percents = [100, 50, 20]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('percent_of_data — Moons dataset (500 samples base)', fontsize=13)

        for ax, pct in zip(axes, percents):
            loader = get_dataset('Moons', set_seed=42, num_samples=500, percent_of_data=pct)
            X = loader.get_X()
            y = loader.get_y()
            label_names = loader.get_label_names()
            lnames = label_names if isinstance(label_names, list) else None

            _plot_dataset(
                X=X, y=y,
                label_names=lnames,
                dim_reducer_method='PCA',
                ax=ax,
            )
            ax.set_title(f'percent_of_data={pct}  (n={len(y)})')

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'percent_of_data.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/percent_of_data.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def generate_minority_reduce_figure() -> str | None:
    """Show class imbalance effect of minority_reduce_scaler=2 on Moons train split."""
    print('  Generating minority_reduce figure...')
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('minority_reduce_scaler — Moons dataset (train split)', fontsize=13)

        loader_default = get_dataset('Moons', set_seed=42, num_samples=300)
        train_default, _ = loader_default.get_train_test_split()
        label_names = loader_default.get_label_names()
        lnames = label_names if isinstance(label_names, list) else None
        n0 = int(np.sum(train_default['y'] == 0))
        n1 = int(np.sum(train_default['y'] == 1))

        _plot_dataset(
            X=train_default['X'], y=train_default['y'],
            label_names=lnames,
            dim_reducer_method='PCA',
            ax=axes[0],
        )
        axes[0].set_title(f'Default  (class 0: {n0}, class 1: {n1})')

        loader_reduced = get_dataset('Moons', set_seed=42, num_samples=300, minority_reduce_scaler=2)
        train_reduced, _ = loader_reduced.get_train_test_split()
        n0r = int(np.sum(train_reduced['y'] == 0))
        n1r = int(np.sum(train_reduced['y'] == 1))

        _plot_dataset(
            X=train_reduced['X'], y=train_reduced['y'],
            label_names=lnames,
            dim_reducer_method='PCA',
            ax=axes[1],
        )
        axes[1].set_title(f'minority_reduce_scaler=2  (class 0: {n0r}, class 1: {n1r})')

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'minority_reduce.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/minority_reduce.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def generate_clf_figure() -> str | None:
    """Show decision boundary and misclassification markers using clf on Moons (2D)."""
    print('  Generating clf figure...')
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        loader = get_dataset('Moons', set_seed=42, num_samples=300)
        train, test = loader.get_train_test_split()
        label_names = loader.get_label_names()
        lnames = label_names if isinstance(label_names, list) else None

        clf_lr = LogisticRegression().fit(train['X'], train['y'])
        clf_svm = SVC(kernel='rbf', gamma=2).fit(train['X'], train['y'])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('clf — decision boundary and misclassification markers (Moons dataset)', fontsize=13)

        # No clf
        _plot_dataset(X=train['X'], y=train['y'], label_names=lnames,
                      dim_reducer_method='PCA', ax=axes[0])
        axes[0].set_title('No classifier')

        # Linear clf (LogisticRegression) — some misclassifications on non-linear data
        _plot_dataset(X=train['X'], y=train['y'], label_names=lnames,
                      dim_reducer_method='PCA', clf=clf_lr.predict, ax=axes[1])
        axes[1].set_title("clf=LogisticRegression  (linear boundary)")

        # Non-linear clf (SVM-RBF) — fits Moons well
        _plot_dataset(X=train['X'], y=train['y'], label_names=lnames,
                      dim_reducer_method='PCA', clf=clf_svm.predict, ax=axes[2])
        axes[2].set_title("clf=SVC(rbf)  (non-linear boundary)")

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'clf.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/clf.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def generate_overlay_figure() -> str | None:
    """Show overlay_train_test mode vs side-by-side on Moons with a classifier."""
    print('  Generating overlay_train_test figure...')
    try:
        from sklearn.linear_model import LogisticRegression

        loader = get_dataset('Moons', set_seed=42, num_samples=300)
        train, test = loader.get_train_test_split()
        label_names = loader.get_label_names()
        lnames = label_names if isinstance(label_names, list) else None

        clf = LogisticRegression().fit(train['X'], train['y'])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('overlay_train_test and y_pred — Moons dataset', fontsize=13)

        # Side-by-side (default)
        _plot_dataset(X=train['X'], y=train['y'], X_test=test['X'], y_test=test['y'],
                      label_names=lnames, dim_reducer_method='PCA',
                      ax=(axes[0], axes[1]))
        axes[0].set_title('overlay_train_test=False (default)\ntrain')
        axes[1].set_title('overlay_train_test=False (default)\ntest')

        # Overlay mode with clf
        _plot_dataset(X=train['X'], y=train['y'], X_test=test['X'], y_test=test['y'],
                      label_names=lnames, dim_reducer_method='PCA',
                      overlay_train_test=True, clf=clf.predict,
                      test_alpha=0.4, ax=axes[2])
        axes[2].set_title('overlay_train_test=True\n(open markers = test, boundary = clf)')

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'overlay_train_test.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/overlay_train_test.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def generate_dim_reducer_figure() -> str | None:
    """Show Wine (13 features) projected to 2D with PCA, t-SNE, and UMAP."""
    print('  Generating dim_reducer figure...')
    try:
        loader = get_dataset('Wine', set_seed=42)
        X = loader.get_X()
        y = loader.get_y()
        label_names = loader.get_label_names()
        lnames = label_names if isinstance(label_names, list) else None

        methods = ['PCA', 'TSNE', 'UMAP']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("dim_reducer — Wine dataset (13 features → 2D)", fontsize=13)

        for ax, method in zip(axes, methods):
            print(f'    Running {method}...')
            _plot_dataset(
                X=X, y=y,
                label_names=lnames,
                dim_reducer_method=method,
                ax=ax,
            )
            ax.set_title(f"dim_reducer='{method}'")

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'dim_reducer.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'options/dim_reducer.png'
    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def build_options_markdown(results: dict[str, str | None]) -> str:
    """Build the <details> options section for README insertion."""

    def img_line(key: str) -> str:
        path = results.get(key)
        if path:
            return f'![{key}](assets/figures/{path})'
        return '_(figure unavailable)_'

    lines = [
        '<details>',
        '<summary><strong>Loader options — visual demos</strong></summary>',
        '',
        "### `get_train_test_split(train_size=...)`",
        '',
        'Splits data into train and test sets while preserving class proportions.',
        '`train_size` controls the fraction used for training (default `0.5`).',
        '',
        '```python',
        "train, test = get_dataset('Moons', train_size=0.7).get_train_test_split()",
        '```',
        '',
        img_line('train_test_split'),
        '',
        '---',
        '',
        "### `scale=True`",
        '',
        'Applies MinMax normalisation fitted on the train set, scaling all features to `[−1, 1]`.',
        'Shown here on the Diabetes Pima Indian dataset — note the axis ranges before and after.',
        '',
        '```python',
        "dataset = get_dataset('Diabetes Pima Indian', scale=True)",
        'train, test = dataset.get_train_test_split()',
        '```',
        '',
        img_line('scale'),
        '',
        '---',
        '',
        "### `percent_of_data`",
        '',
        'Subsamples the full dataset to the given percentage while preserving class proportions.',
        '',
        '```python',
        "dataset = get_dataset('Moons', percent_of_data=50)  # keep 50% of data",
        '```',
        '',
        img_line('percent_of_data'),
        '',
        '---',
        '',
        "### `minority_reduce_scaler`",
        '',
        'Reduces the minority class in the **train** split by the given factor,',
        'creating a class-imbalanced training set (useful for cost-sensitive learning).',
        '',
        '```python',
        "dataset = get_dataset('Moons', minority_reduce_scaler=2)",
        'train, test = dataset.get_train_test_split()',
        '```',
        '',
        img_line('minority_reduce'),
        '',
        '---',
        '',
        "### `dim_reducer`",
        '',
        'Applies dimensionality reduction to the split output.',
        'Supported methods: `PCA`, `kernelPCA`, `TSNE`, `UMAP`, `UMAP_supervised`.',
        'Shown on the Wine dataset (13 features projected to 2D).',
        '',
        '```python',
        "dataset = get_dataset('Wine', dim_reducer='UMAP', reduce_to_dim=2)",
        'train, test = dataset.get_train_test_split()',
        '```',
        '',
        img_line('dim_reducer'),
        '',
        '---',
        '',
        "### `clf` — classifier decision boundary",
        '',
        'Pass a trained classifier to overlay its decision boundary and highlight misclassified',
        'points (marked with **×**). The classifier must accept inputs in the **original feature',
        'space**. Decision boundary regions are drawn when the data is already 2D (no dim',
        'reduction needed); for high-dim data, only misclassification markers are shown.',
        '',
        '```python',
        'from sklearn.linear_model import LogisticRegression',
        '',
        'train, test = get_dataset("Moons").get_train_test_split()',
        'clf = LogisticRegression().fit(train["X"], train["y"])',
        '',
        '# Pass clf to see boundary + misclassification markers',
        'plot_dataset(train["X"], train["y"], clf=clf.predict)',
        '',
        '# Or use pre-computed predictions (no boundary, just markers)',
        'plot_dataset(train["X"], train["y"], y_pred=clf.predict(train["X"]))',
        '```',
        '',
        img_line('clf'),
        '',
        '---',
        '',
        "### `overlay_train_test` — train and test on one plot",
        '',
        'By default, train and test are shown in separate side-by-side subplots.',
        'Set `overlay_train_test=True` to plot both on one axes: train as filled circles,',
        'test as open circles at `test_alpha` opacity (default `0.3`).',
        'Combine with `clf` to show the decision boundary and misclassifications together.',
        '',
        '```python',
        'train, test = get_dataset("Moons").get_train_test_split()',
        'clf = LogisticRegression().fit(train["X"], train["y"])',
        '',
        '# Overlay with custom test transparency and classifier boundary',
        'plot_dataset(train["X"], train["y"], X_test=test["X"], y_test=test["y"],',
        '             overlay_train_test=True, test_alpha=0.4, clf=clf.predict)',
        '',
        '# Also available on AbstractLoader:',
        'loader.plot_train_test_split(overlay_train_test=True, clf=clf.predict)',
        '```',
        '',
        img_line('overlay_train_test'),
        '',
        '</details>',
    ]
    return '\n'.join(lines)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results: dict[str, str | None] = {}
    results['train_test_split'] = generate_train_test_split_figure()
    results['scale'] = generate_scale_figure()
    results['percent_of_data'] = generate_percent_of_data_figure()
    results['minority_reduce'] = generate_minority_reduce_figure()
    results['dim_reducer'] = generate_dim_reducer_figure()
    results['clf'] = generate_clf_figure()
    results['overlay_train_test'] = generate_overlay_figure()

    options_md = build_options_markdown(results)
    options_path = os.path.join(REPO_ROOT, 'assets', 'figures', 'OPTIONS_SECTION.md')
    with open(options_path, 'w') as f:
        f.write(options_md)

    print(f'\nDone. Options markdown written to: {options_path}')
    print(f'Figures saved to: {FIGURES_DIR}')
    print('\nNext: open assets/figures/OPTIONS_SECTION.md and paste its content into README.md')


if __name__ == '__main__':
    main()
