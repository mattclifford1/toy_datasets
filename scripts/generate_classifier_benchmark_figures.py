"""Generate classifier benchmark figures and a README section for all non-image datasets.

Run with:
    uv run python scripts/generate_classifier_benchmark_figures.py

Outputs:
    assets/figures/benchmark_heatmap.png              - all datasets × classifiers heatmap
    assets/figures/benchmark_synthetic.png            - bar chart for Synthetic group
    assets/figures/benchmark_classic.png              - bar chart for Classic group
    assets/figures/benchmark_medical.png              - bar chart for Medical group
    assets/figures/benchmark_other.png                - bar chart for Other group
    assets/figures/benchmark_clf_plots_synthetic.png  - overlay scatter plots for Synthetic group
    assets/figures/benchmark_clf_plots_classic.png    - overlay scatter plots for Classic group
    assets/figures/benchmark_clf_plots_medical.png    - overlay scatter plots for Medical group
    assets/figures/benchmark_clf_plots_other.png      - overlay scatter plots for Other group
    assets/figures/BENCHMARK_SECTION.md               - <details> block ready to paste into README.md
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.embeddings import DimReducer as _DimReducer
from data_loaders.main import AVAILABLE_DATASETS, get_dataset
from data_loaders.plotting.visualisation import plot_dataset as _plot_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'assets', 'figures')

SKIP_DATASETS: set[str] = {
    'MNIST', 'CIFAR-10', 'CIFAR-100', 'CIFAR-10N',
    'Costcla Credit Scoring Kaggle 2011',
    'Costcla Credit Scoring PAKDD 2009',
    'Costcla Direct Marketing',
}

DATASET_GROUPS: dict[str, list[str]] = {
    'Synthetic': ['XOR', 'Moons', 'Blobs', 'Circles', 'Sklearn Normal', 'Gaussian'],
    'Classic': ['Iris', 'Wine', 'Breast Cancer'],
    'Medical': [
        'Diabetes Pima Indian', 'Heart Disease', 'Breast Cancer Wisconsin',
        'Habermans Breast Cancer', 'Chronic Kidney Disease', 'Hepatitis',
    ],
    'Other': [
        'Banknote Authentication', 'Wheat Seeds', 'Ionosphere',
        'Sonar Rocks vs Mines', 'Abalone Gender',
    ],
}

DATASET_SHORT_NAMES: dict[str, str] = {
    'XOR': 'XOR',
    'Moons': 'Moons',
    'Blobs': 'Blobs',
    'Circles': 'Circles',
    'Sklearn Normal': 'Sklearn Normal',
    'Gaussian': 'Gaussian',
    'Iris': 'Iris',
    'Wine': 'Wine',
    'Breast Cancer': 'Breast Cancer',
    'Diabetes Pima Indian': 'Diabetes Pima',
    'Heart Disease': 'Heart Disease',
    'Breast Cancer Wisconsin': 'BC Wisconsin',
    'Habermans Breast Cancer': 'Habermans',
    'Chronic Kidney Disease': 'Kidney Disease',
    'Hepatitis': 'Hepatitis',
    'Banknote Authentication': 'Banknote',
    'Wheat Seeds': 'Wheat Seeds',
    'Ionosphere': 'Ionosphere',
    'Sonar Rocks vs Mines': 'Sonar',
    'Abalone Gender': 'Abalone',
}

CLASSIFIER_CONFIGS: list[dict] = [
    {'name': 'LogReg',     'label': 'Logistic\nRegression'},
    {'name': 'RF',         'label': 'Random\nForest'},
    {'name': 'SVC-RBF',    'label': 'SVC\n(RBF)'},
    {'name': 'KNN',        'label': 'KNN'},
    {'name': 'GaussianNB', 'label': 'Gaussian NB'},
]


def build_classifiers() -> list[tuple[str, object]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    return [
        ('LogReg',     LogisticRegression(max_iter=5000, random_state=42)),
        ('RF',         RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVC-RBF',    SVC(kernel='rbf', random_state=42)),
        ('KNN',        KNeighborsClassifier(n_neighbors=5)),
        ('GaussianNB', GaussianNB()),
    ]


def run_benchmark(name: str) -> dict[str, float] | None:
    """Fit each classifier on the train split and return balanced_accuracy_score on test."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    print(f'  Benchmarking: {name}')
    try:
        loader = get_dataset(name, set_seed=42)
        train, test = loader.get_train_test_split()
        X_train, y_train = train['X'], train['y']
        X_test, y_test = test['X'], test['y']

        scores: dict[str, float] = {}
        for clf_name, clf in build_classifiers():
            clf_instance = clone(clf)
            clf_instance.fit(X_train, y_train)
            y_pred = clf_instance.predict(X_test)
            scores[clf_name] = balanced_accuracy_score(y_test, y_pred)
        return scores
    except Exception as e:
        print(f'  ERROR on {name}: {e}')
        return None


def collect_results() -> dict[str, dict[str, float] | None]:
    results: dict[str, dict[str, float] | None] = {}
    for names in DATASET_GROUPS.values():
        for name in names:
            if name not in AVAILABLE_DATASETS:
                print(f'  SKIP (not in AVAILABLE_DATASETS): {name}')
                continue
            if name in SKIP_DATASETS:
                print(f'  SKIP (excluded): {name}')
                continue
            results[name] = run_benchmark(name)
    return results


def generate_heatmap(results: dict[str, dict[str, float] | None]) -> str | None:
    """Save benchmark_heatmap.png — all datasets × classifiers coloured by balanced accuracy."""
    print('  Generating heatmap...')
    try:
        clf_names = [c['name'] for c in CLASSIFIER_CONFIGS]
        clf_labels = [c['label'] for c in CLASSIFIER_CONFIGS]

        ordered_names = [
            n for names in DATASET_GROUPS.values() for n in names if n in results
        ]
        short_labels = [DATASET_SHORT_NAMES.get(n, n) for n in ordered_names]

        matrix = np.full((len(ordered_names), len(clf_names)), np.nan)
        for i, name in enumerate(ordered_names):
            row = results.get(name)
            if row is not None:
                for j, clf_name in enumerate(clf_names):
                    matrix[i, j] = row.get(clf_name, np.nan)

        fig, ax = plt.subplots(figsize=(10, max(6, len(ordered_names) * 0.55 + 1)))
        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad('lightgrey')

        im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.0, aspect='auto')
        fig.colorbar(im, ax=ax, label='Balanced Accuracy', shrink=0.8)

        ax.set_xticks(range(len(clf_names)))
        ax.set_xticklabels(clf_labels, fontsize=9)
        ax.set_yticks(range(len(ordered_names)))
        ax.set_yticklabels(short_labels, fontsize=9)
        ax.set_title('Classifier Benchmark — Balanced Accuracy (test set)', fontsize=12, pad=12)

        for i in range(len(ordered_names)):
            for j in range(len(clf_names)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7.5, color='black')

        # Draw white dividers between dataset groups
        cumulative = 0
        for names in list(DATASET_GROUPS.values())[:-1]:
            valid = [n for n in names if n in results]
            cumulative += len(valid)
            ax.axhline(cumulative - 0.5, color='white', linewidth=2)

        fig.tight_layout()
        fpath = os.path.join(FIGURES_DIR, 'benchmark_heatmap.png')
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return 'benchmark_heatmap.png'
    except Exception as e:
        print(f'  ERROR generating heatmap: {e}')
        return None


def generate_group_bar_chart(
    group_name: str,
    names: list[str],
    results: dict[str, dict[str, float] | None],
) -> str | None:
    """Save benchmark_{group}.png — grouped bar chart for one dataset group."""
    print(f'  Generating bar chart for {group_name}...')
    try:
        clf_names = [c['name'] for c in CLASSIFIER_CONFIGS]
        valid_names = [n for n in names if n in results and results[n] is not None]
        if not valid_names:
            return None

        x = np.arange(len(valid_names))
        n_clfs = len(clf_names)
        bar_width = 0.15
        offsets = np.linspace(-(n_clfs - 1) / 2, (n_clfs - 1) / 2, n_clfs) * bar_width

        fig, ax = plt.subplots(figsize=(max(8, len(valid_names) * 1.4), 5))
        colors = plt.cm.tab10.colors[:n_clfs]

        for clf_name, offset, color in zip(clf_names, offsets, colors):
            values = [results[n][clf_name] for n in valid_names]
            ax.bar(x + offset, values, bar_width, label=clf_name, color=color, alpha=0.85)

        short_labels = [DATASET_SHORT_NAMES.get(n, n) for n in valid_names]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'Classifier Benchmark — {group_name}', fontsize=12)
        ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Random (0.5)')
        ax.legend(loc='lower right', fontsize=8, ncol=3)

        fig.tight_layout()
        slug = group_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        fname = f'benchmark_{slug}.png'
        fpath = os.path.join(FIGURES_DIR, fname)
        fig.savefig(fpath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return fname
    except Exception as e:
        print(f'  ERROR generating bar chart for {group_name}: {e}')
        return None


def generate_group_clf_plots(
    group_name: str,
    names: list[str],
    results: dict[str, dict[str, float] | None],
) -> str | None:
    """Save benchmark_clf_plots_{group}.png — one subplot per dataset showing train+test
    overlaid with the best-performing classifier from the benchmark results."""
    from sklearn.base import clone
    print(f'  Generating clf plots for {group_name}...')
    try:
        valid_names = [n for n in names if n in results]
        if not valid_names:
            return None

        ncols = 3
        nrows = (len(valid_names) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        axes_flat = np.array(axes).flatten()

        clf_dict = dict(build_classifiers())

        for i, name in enumerate(valid_names):
            ax = axes_flat[i]
            short = DATASET_SHORT_NAMES.get(name, name)

            scores = results.get(name)
            best_clf_name = max(scores, key=scores.get) if scores else 'RF'
            best_score = scores[best_clf_name] if scores else float('nan')

            try:
                loader = get_dataset(name, set_seed=42)
                train, test = loader.get_train_test_split()

                # Project to 2D first so the clf is fitted in the same space
                # that _plot_dataset will visualise — this lets it draw the boundary.
                embedder = _DimReducer(
                    train['X'], train['y'], reducer=loader.default_dim_reducer
                )
                X_train_2d = embedder.transform(train['X'])
                X_test_2d = embedder.transform(test['X'])

                clf_instance = clone(clf_dict[best_clf_name])
                clf_instance.fit(X_train_2d, train['y'])

                # Data is now 2D so _plot_dataset won't apply any further reduction
                # and can always draw the decision boundary.
                _plot_dataset(
                    X=X_train_2d, y=train['y'],
                    X_test=X_test_2d, y_test=test['y'],
                    overlay_train_test=True,
                    clf=clf_instance.predict,
                    test_alpha=0.4,
                    show_legend=False,
                    ax=ax,
                )
                ax.set_xlabel(f'{embedder.reducer_name} 1', fontsize=7)
                ax.set_ylabel(f'{embedder.reducer_name} 2', fontsize=7)
                ax.set_title(f'{short}  [{best_clf_name}  bal-acc={best_score:.2f}]', fontsize=9)
            except Exception as e:
                ax.set_visible(False)
                print(f'    ERROR on {name}: {e}')

        for j in range(len(valid_names), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(
            f'{group_name} — train (filled) + test (open) + best classifier boundary',
            fontsize=13,
        )
        fig.tight_layout()
        slug = group_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        fname = f'benchmark_clf_plots_{slug}.png'
        fig.savefig(os.path.join(FIGURES_DIR, fname), bbox_inches='tight', dpi=100)
        plt.close(fig)
        return fname
    except Exception as e:
        print(f'  ERROR generating clf plots for {group_name}: {e}')
        return None


def print_summary(results: dict[str, dict[str, float] | None]) -> None:
    """Print a plain-text table sorted by mean balanced accuracy descending."""
    clf_names = [c['name'] for c in CLASSIFIER_CONFIGS]
    col_w = 10
    name_w = 30

    header = f"{'Dataset':<{name_w}} | " + ' '.join(f'{c:>{col_w}}' for c in clf_names) + f" | {'Mean':>{col_w}}"
    print('\n' + header)
    print('-' * len(header))

    rows = []
    for name, scores in results.items():
        if scores is None:
            continue
        vals = [scores.get(c, float('nan')) for c in clf_names]
        mean = float(np.nanmean(vals))
        rows.append((name, vals, mean))

    for name, vals, mean in sorted(rows, key=lambda r: r[2], reverse=True):
        val_str = ' '.join(f'{v:>{col_w}.3f}' for v in vals)
        print(f'{name:<{name_w}} | {val_str} | {mean:>{col_w}.3f}')


def build_benchmark_markdown(
    heatmap_fname: str | None,
    group_bar_fnames: dict[str, str | None],
    group_clf_fnames: dict[str, str | None],
) -> str:
    """Build the <details> benchmark section for README insertion."""

    def img_line(fname: str | None, alt: str) -> str:
        if fname:
            return f'![{alt}](assets/figures/{fname})'
        return '_(figure unavailable)_'

    clf_labels = ', '.join(c['label'].replace('\n', ' ') for c in CLASSIFIER_CONFIGS)

    lines = [
        '<details>',
        '<summary><strong>Classifier benchmarks</strong></summary>',
        '',
        'Balanced accuracy (test set) for 5 sklearn classifiers across all non-image datasets.',
        'Train/test split: 50/50 (default). Seed: 42.',
        '',
        f'Classifiers: {clf_labels}.',
        '',
        '### Summary heatmap',
        '',
        img_line(heatmap_fname, 'benchmark_heatmap'),
        '',
    ]

    for group_name in group_bar_fnames:
        bar_fname = group_bar_fnames.get(group_name)
        clf_fname = group_clf_fnames.get(group_name)
        lines += [
            f'### {group_name}',
            '',
            img_line(bar_fname, f'benchmark_{group_name.lower()}'),
            '',
            img_line(clf_fname, f'benchmark_clf_plots_{group_name.lower()}'),
            '',
        ]

    lines.append('</details>')
    return '\n'.join(lines)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print('Running classifier benchmarks...')
    results = collect_results()

    print('\nGenerating figures...')
    heatmap_fname = generate_heatmap(results)

    group_bar_fnames: dict[str, str | None] = {}
    group_clf_fnames: dict[str, str | None] = {}
    for group_name, names in DATASET_GROUPS.items():
        group_bar_fnames[group_name] = generate_group_bar_chart(group_name, names, results)
        group_clf_fnames[group_name] = generate_group_clf_plots(group_name, names, results)

    print_summary(results)

    benchmark_md = build_benchmark_markdown(heatmap_fname, group_bar_fnames, group_clf_fnames)
    md_path = os.path.join(FIGURES_DIR, 'BENCHMARK_SECTION.md')
    with open(md_path, 'w') as f:
        f.write(benchmark_md)

    print(f'\nDone. Benchmark markdown written to: {md_path}')
    print(f'Figures saved to: {FIGURES_DIR}')
    print('\nNext: open assets/figures/BENCHMARK_SECTION.md and paste its content into README.md')


if __name__ == '__main__':
    main()
