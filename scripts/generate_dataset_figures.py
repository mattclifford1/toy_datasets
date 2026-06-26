"""Generate dataset figures and a README gallery section for all available datasets.

Run with:
    uv run python scripts/generate_dataset_figures.py

Outputs:
    assets/figures/<dataset_name>.png  - one figure per dataset
    assets/figures/GALLERY_SECTION.md  - <details> block ready to paste into README.md
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data_loaders.main import AVAILABLE_DATASETS, get_dataset
from data_loaders.plotting.visualisation import plot_dataset as _plot_dataset
from data_loaders.plotting.visualisation import plot_class_samples as _plot_class_samples

# Example images shown per class beside the 2D projection for image datasets.
SAMPLES_PER_CLASS = 5

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'figures')

# Datasets that require special external access or remote downloads that may be flaky
SKIP_DATASETS: set[str] = {
    'Costcla Credit Scoring Kaggle 2011',
    'Costcla Credit Scoring PAKDD 2009',
    'Costcla Direct Marketing',
    'CIFAR-10N',  # requires manual label file download
}

# Dataset groupings for the gallery (order preserved)
DATASET_GROUPS: dict[str, list[str]] = {
    'Synthetic': ['XOR', 'Moons', 'Blobs', 'Circles', 'Sklearn Normal', 'Gaussian'],
    'Classic (sklearn)': ['Iris', 'Wine', 'Breast Cancer'],
    'Medical': [
        'Diabetes Pima Indian', 'Heart Disease', 'Breast Cancer Wisconsin',
        'Habermans Breast Cancer', 'Chronic Kidney Disease', 'Hepatitis',
        'Parkinsons', 'Indian Liver Patient', 'Cervical Cancer', 'Arrhythmia',
        'Thyroid Sick', 'Stroke Prediction', 'Framingham CHD', 'Thoracic Surgery',
        'SPECTF Heart', 'Heart Failure', 'Mammographic Mass',
        'Breast Cancer Prognostic', 'Breast Cancer Coimbra', 'HCC Survival',
        'Z-Alizadeh Sani CAD',
    ],
    'Other': [
        'Banknote Authentication', 'Wheat Seeds', 'Ionosphere',
        'Sonar Rocks vs Mines', 'Abalone Gender',
    ],
    'Image': ['MNIST', 'Fashion-MNIST', 'SVHN', 'EuroSAT', 'CIFAR-10', 'CIFAR-100', 'CIFAR-10N'],
    'Medical Image (MedMNIST)': [
        'PneumoniaMNIST', 'BreastMNIST', 'DermaMNIST',
        'BloodMNIST', 'PathMNIST', 'OCTMNIST',
    ],
}

# Per-dataset load overrides. Image datasets are capped to a few thousand
# samples so the 2D projection used for the gallery figure stays quick to
# compute; ``percent_of_data`` keeps class proportions for the MedMNIST sets
# whose loaders take no ``size`` argument.
DATASET_LOAD_KWARGS: dict[str, dict] = {
    'MNIST': {'size': 3000},
    'Fashion-MNIST': {'size': 3000},
    'SVHN': {'size': 3000},
    'EuroSAT': {'size': 3000},
    'CIFAR-10': {'size': 3000},
    'CIFAR-100': {'size': 3000},
    'PneumoniaMNIST': {'percent_of_data': 60},
    'BreastMNIST': {},
    'DermaMNIST': {'percent_of_data': 40},
    'BloodMNIST': {'percent_of_data': 25},
    'PathMNIST': {'percent_of_data': 3},
    'OCTMNIST': {'percent_of_data': 3},
}


def safe_filename(name: str) -> str:
    return name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')


def _build_image_figure(loader, name, X, y, label_names):
    """Combined figure: 2D projection on the left, example images per class on the right."""
    samples = loader.sample_images_per_class(n_per_class=SAMPLES_PER_CLASS)
    classes = sorted(samples)
    n_classes = len(classes)
    n_per = max((len(samples[c]) for c in classes), default=1)

    fig = plt.figure(figsize=(6 + n_per * 1.3, 6.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[6, n_per * 1.3], wspace=0.2)
    scatter_ax = fig.add_subplot(outer[0, 0])

    # vertically centre the (usually 2-row) image grid against the tall scatter
    # so the right-hand panel doesn't leave large empty margins
    cell = n_per * 1.3 / n_per  # width of one square image cell in inches
    pad = max((6.0 - n_classes * cell) / 2, 0.0)
    centred = outer[0, 1].subgridspec(3, 1, height_ratios=[pad, n_classes * cell, pad])
    inner = centred[1, 0].subgridspec(n_classes, n_per, wspace=0.08, hspace=0.15)
    sample_axes = np.empty((n_classes, n_per), dtype=object)
    for r in range(n_classes):
        for c in range(n_per):
            sample_axes[r, c] = fig.add_subplot(inner[r, c])

    _plot_dataset(
        X=X,
        y=y,
        dataset_name=None,
        label_names=label_names if isinstance(label_names, list) else None,
        dim_reducer_method=loader.default_dim_reducer,
        ax=scatter_ax,
    )
    _plot_class_samples(samples, label_names=label_names, axes=sample_axes)
    fig.suptitle(f'{name} dataset', fontsize=14)
    return fig


def generate_figure(name: str) -> str | None:
    """Load dataset, plot with PCA, save PNG. Returns relative path or None on failure."""
    print(f'  Generating figure for: {name}')
    try:
        loader = get_dataset(name, set_seed=42, **DATASET_LOAD_KWARGS.get(name, {}))
        X = loader.get_X()
        y = loader.get_y()
        label_names = loader.get_label_names()

        if getattr(loader, 'is_image', False):
            # image datasets get the scatter plus a grid of example images so
            # readers can see what the raw data actually looks like
            fig = _build_image_figure(loader, name, X, y, label_names)
        else:
            fig, _ = _plot_dataset(
                X=X,
                y=y,
                dataset_name=name,
                label_names=label_names if isinstance(label_names, list) else None,
                dim_reducer_method=loader.default_dim_reducer,
            )
        fname = f'{safe_filename(name)}.png'
        fpath = os.path.join(FIGURES_DIR, fname)
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return fname
    except Exception as e:
        print(f'  ERROR generating {name}: {e}')
        return None


def get_stats_block(name: str) -> str:
    """Return get_info(long=False) output for a dataset."""
    try:
        loader = get_dataset(name, set_seed=42, **DATASET_LOAD_KWARGS.get(name, {}))
        return loader.get_info(long=False)
    except Exception as e:
        return f'(stats unavailable: {e})'


def build_gallery_markdown(generated: dict[str, str | None]) -> str:
    """Build per-category <details> blocks for README insertion."""
    lines: list[str] = []

    for group, names in DATASET_GROUPS.items():
        lines.append('<details>')
        lines.append(f'<summary><strong>{group}</strong></summary>')
        lines.append('')
        for name in names:
            if name in SKIP_DATASETS:
                continue
            lines.append(f'#### {name}')
            lines.append('')
            lines.append('```')
            lines.append(get_stats_block(name))
            lines.append('```')
            lines.append('')
            fname = generated.get(name)
            if fname:
                rel_path = f'assets/figures/{fname}'
                lines.append(f'![{name}]({rel_path})')
            else:
                lines.append('_(figure unavailable)_')
            lines.append('')
        lines.append('</details>')
        lines.append('')

    return '\n'.join(lines)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    generated: dict[str, str | None] = {}
    all_names = [n for group in DATASET_GROUPS.values() for n in group if n not in SKIP_DATASETS]

    for name in all_names:
        if name not in AVAILABLE_DATASETS:
            print(f'  SKIP (not in AVAILABLE_DATASETS): {name}')
            continue
        fname = generate_figure(name)
        generated[name] = fname

    gallery_md = build_gallery_markdown(generated)
    gallery_path = os.path.join(FIGURES_DIR, 'GALLERY_SECTION.md')
    with open(gallery_path, 'w') as f:
        f.write(gallery_md)

    print(f'\nDone. Gallery markdown written to: {gallery_path}')
    print(f'Figures saved to: {FIGURES_DIR}')
    print('\nNext: open assets/figures/GALLERY_SECTION.md and paste its content into README.md')


if __name__ == '__main__':
    main()
