# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Visualisation functions for each dataset
'''
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _plot_decision_boundary(
        ax: Any,
        clf: Callable[[np.ndarray], np.ndarray],
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        colors: list[str],
        resolution: int = 200,
        fill_alpha: float = 0.25,
) -> None:
    """Draw decision boundary regions on ax using clf in the 2D visualization space."""
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], resolution),
        np.linspace(ylim[0], ylim[1], resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = np.array(clf(grid)).reshape(xx.shape)

    unique = sorted(np.unique(Z).astype(int))
    region_colors = [colors[c] for c in unique]
    levels = [v - 0.5 for v in unique] + [unique[-1] + 0.5]

    ax.contourf(xx, yy, Z, levels=levels, colors=region_colors, alpha=fill_alpha)
    if len(unique) > 1:
        boundary_level = (unique[0] + unique[-1]) / 2
        ax.contour(xx, yy, Z, levels=[boundary_level],
                   colors=['black'], linewidths=1.0, linestyles='--')


def _scatter_predictions(
        ax: Any,
        X_2d: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        colors: list[str],
        alpha: float = 0.8,
        open_markers: bool = False,
        label_suffix: str = '',
) -> None:
    """Scatter data colored by true label; mark misclassified points with 'X'."""
    misclassified = y_true != y_pred
    for cls in [0, 1]:
        mask_correct = (y_true == cls) & ~misclassified
        mask_wrong = (y_true == cls) & misclassified
        fc = 'none' if open_markers else colors[cls]
        ec = colors[cls]

        if mask_correct.any():
            ax.scatter(
                X_2d[mask_correct, 0], X_2d[mask_correct, 1],
                facecolor=fc, edgecolor=ec, alpha=alpha, s=20,
                linewidths=0.8 if open_markers else 0,
            )
        if mask_wrong.any():
            ax.scatter(
                X_2d[mask_wrong, 0], X_2d[mask_wrong, 1],
                marker='X', facecolor=fc, edgecolor='black',
                alpha=alpha, s=40, linewidths=0.5, zorder=5,
                label=f'Misclassified{label_suffix}' if mask_wrong.any() else None,
            )


def plot_dataset(
        X: np.ndarray,
        y: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        dataset_name: str | None = None,
        label_names: list[str | int] | None = None,
        terminal_plot: bool = False,
        dim_reducer_method: str = 'TSNE',
        ax: Any = None,
        clf: Callable[[np.ndarray], np.ndarray] | None = None,
        y_pred: np.ndarray | None = None,
        y_pred_test: np.ndarray | None = None,
        overlay_train_test: bool = False,
        test_alpha: float = 0.3,
        show_legend: bool = True,
) -> tuple[Any, Any] | None:
    """
    Plot dataset in 2D using dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for training/full dataset.
    y : np.ndarray
        Labels for training/full dataset.
    X_test : np.ndarray, optional
        Feature matrix for test dataset.
    y_test : np.ndarray, optional
        Labels for test dataset.
    dataset_name : str, optional
        Name of dataset for title.
    label_names : list, optional
        Names for each class label.
    terminal_plot : bool, default=False
        If True, render plot in terminal (requires ax=None).
    dim_reducer_method : str, default='TSNE'
        Dimensionality reduction method: 'PCA', 'TSNE', or 'UMAP'.
    ax : matplotlib.axes.Axes or tuple of 2 Axes, optional
        Axes to plot on. For train/test split with overlay_train_test=False,
        provide a tuple of 2 Axes. If None: create a new figure.
    clf : callable, optional
        Classifier callable in the original feature space: ``clf(X) -> labels``.
        Used to mark misclassified points on the scatter. If the data is already
        2D (no dim reduction applied), also draws shaded decision regions and
        boundary line. For high-dim data with dim reduction, only point marking
        is shown — wrap clf with your dim reducer to get boundary visualization.
    y_pred : np.ndarray, optional
        Pre-computed predictions for X. If provided alongside clf, these are used
        for point coloring (clf is still used for the decision boundary if applicable).
    y_pred_test : np.ndarray, optional
        Pre-computed predictions for X_test.
    overlay_train_test : bool, default=False
        If True and X_test is provided, plot train and test on one axes instead
        of side-by-side subplots. Train points use filled markers; test points
        use open markers at test_alpha opacity.
    test_alpha : float, default=0.3
        Alpha for test data scatter points when overlay_train_test=True.
    show_legend : bool, default=True
        If False, suppress the legend on all axes.

    Returns
    -------
    tuple or None
        If ax=None: returns (fig, ax) for single/overlay, or (fig, [ax1, ax2])
        for side-by-side split. If ax provided: returns None.

    Raises
    ------
    ValueError
        If terminal_plot=True and ax is provided.
        If overlay_train_test=True and ax is a tuple/list.
        If ax format doesn't match the expected layout.
    """
    import matplotlib.pyplot as plt
    from data_loaders.plotting.terminal_plots import terminal_show
    from data_loaders.embeddings import DimReducer

    if terminal_plot and ax is not None:
        raise ValueError(
            "Cannot use terminal_plot=True with provided ax. "
            "Terminal plotting requires creating its own figure."
        )
    if overlay_train_test and isinstance(ax, (list, tuple)):
        raise ValueError(
            "overlay_train_test=True requires a single Axes, not a tuple. "
            "Pass a single ax or ax=None to create a new figure."
        )

    has_test = X_test is not None and y_test is not None
    side_by_side = has_test and not overlay_train_test
    created_fig = ax is None

    if created_fig:
        if side_by_side:
            fig, axes_array = plt.subplots(1, 2, figsize=(12, 6))
            axes = list(axes_array)
        else:
            fig, single_ax = plt.subplots(figsize=(6, 6))
            axes = [single_ax]
    else:
        fig = None
        if side_by_side:
            if not isinstance(ax, (list, tuple)) or len(ax) != 2:
                raise ValueError("For train/test split, ax should be a tuple/list of 2 Axes")
            axes = list(ax)
        else:
            if isinstance(ax, (list, tuple)):
                raise ValueError("For single/overlay plot, ax should be a single Axes")
            axes = [ax]

    colors = ["#3ea3e6", "#e56a6a"]  # blue, red

    embedder = DimReducer(X, y, reducer=dim_reducer_method)
    X_2d = embedder.transform(X)
    X_2d_test = embedder.transform(X_test) if has_test else None

    # global axis limits across all data
    all_2d = np.vstack([X_2d, X_2d_test]) if X_2d_test is not None else X_2d
    x_min, x_max = all_2d[:, 0].min(), all_2d[:, 0].max()
    y_min, y_max = all_2d[:, 1].min(), all_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    # resolve predictions for point coloring
    y_pred_train = y_pred if y_pred is not None else (np.array(clf(X)) if clf is not None else None)
    y_pred_test_resolved = (
        y_pred_test if y_pred_test is not None
        else (np.array(clf(X_test)) if clf is not None and has_test else None)
    )

    # decision boundary — only when clf is provided and no dim reduction was applied
    can_draw_boundary = clf is not None and embedder.reducer_model is None
    plot_ax = axes[0]

    if can_draw_boundary:
        _plot_decision_boundary(plot_ax, clf, xlim, ylim, colors)

    # --- train scatter ---
    if y_pred_train is not None:
        _scatter_predictions(plot_ax, X_2d, y, y_pred_train, colors, alpha=0.8)
        # legend entries for class counts
        for cls in [0, 1]:
            n = int(np.sum(y == cls))
            lbl = f"Class {cls}: {label_names[cls]} (n={n})" if label_names else f"Class {cls} (n={n})"
            plot_ax.scatter([], [], color=colors[cls], s=20, label=lbl)
        if np.any(y != y_pred_train):
            plot_ax.scatter([], [], marker='X', color='grey', edgecolor='black',
                            s=40, linewidths=0.5, label='Misclassified (train)')
    else:
        for cls in [0, 1]:
            n = int(np.sum(y == cls))
            lbl = f"Class {cls}: {label_names[cls]} (n={n})" if label_names else f"Class {cls} (n={n})"
            plot_ax.scatter(
                X_2d[y == cls, 0], X_2d[y == cls, 1],
                color=colors[cls], alpha=0.8, s=20, label=lbl,
            )

    # --- test scatter (overlay mode) ---
    if has_test and overlay_train_test:
        if y_pred_test_resolved is not None:
            _scatter_predictions(plot_ax, X_2d_test, y_test, y_pred_test_resolved,
                                 colors, alpha=test_alpha, open_markers=True)
            if np.any(y_test != y_pred_test_resolved):
                plot_ax.scatter([], [], marker='X', facecolor='none', edgecolor='black',
                                s=40, linewidths=0.5, label='Misclassified (test)')
        else:
            for cls in [0, 1]:
                plot_ax.scatter(
                    X_2d_test[y_test == cls, 0], X_2d_test[y_test == cls, 1],
                    facecolor='none', edgecolor=colors[cls],
                    alpha=test_alpha, s=20, linewidths=0.8,
                )
        # legend markers for test
        for cls in [0, 1]:
            n = int(np.sum(y_test == cls))
            lbl = f"Class {cls} test (n={n})"
            plot_ax.scatter([], [], facecolor='none', edgecolor=colors[cls],
                            s=20, linewidths=0.8, label=lbl)

    title = (
        f"Train + Test in {embedder.reducer_name} space"
        if (has_test and overlay_train_test)
        else f"Dataset in {embedder.reducer_name} space"
        if not side_by_side
        else f"Train set in {embedder.reducer_name} space"
    )
    plot_ax.set_title(title)
    plot_ax.set_xlabel(embedder.feature_names[0])
    plot_ax.set_ylabel(embedder.feature_names[1])
    if show_legend:
        plot_ax.legend()
    plot_ax.set_xlim(*xlim)
    plot_ax.set_ylim(*ylim)

    # --- test scatter (side-by-side mode) ---
    if side_by_side:
        test_ax = axes[1]
        if can_draw_boundary:
            _plot_decision_boundary(test_ax, clf, xlim, ylim, colors)

        if y_pred_test_resolved is not None:
            _scatter_predictions(test_ax, X_2d_test, y_test, y_pred_test_resolved, colors, alpha=0.8)
            for cls in [0, 1]:
                n = int(np.sum(y_test == cls))
                lbl = f"Class {cls}: {label_names[cls]} (n={n})" if label_names else f"Class {cls} (n={n})"
                test_ax.scatter([], [], color=colors[cls], s=20, label=lbl)
            if np.any(y_test != y_pred_test_resolved):
                test_ax.scatter([], [], marker='X', color='grey', edgecolor='black',
                                s=40, linewidths=0.5, label='Misclassified')
        else:
            for cls in [0, 1]:
                n = int(np.sum(y_test == cls))
                lbl = f"Class {cls}: {label_names[cls]} (n={n})" if label_names else f"Class {cls} (n={n})"
                test_ax.scatter(
                    X_2d_test[y_test == cls, 0], X_2d_test[y_test == cls, 1],
                    color=colors[cls], alpha=0.8, s=20, label=lbl,
                )
        test_ax.set_title(f"Test set in {embedder.reducer_name} space")
        test_ax.set_xlabel(embedder.feature_names[0])
        test_ax.set_ylabel(embedder.feature_names[1])
        if show_legend:
            test_ax.legend()
        test_ax.set_xlim(*xlim)
        test_ax.set_ylim(*ylim)

    if dataset_name is not None and created_fig:
        fig.suptitle(f"{dataset_name} dataset", fontsize=14)

    if created_fig:
        plt.tight_layout()
        if terminal_plot:
            terminal_show()
        else:
            plt.show()
        if side_by_side:
            return (fig, axes)
        return (fig, axes[0])
    return None


if __name__ == "__main__":
    import data_loaders
    dataset_name = 'Abalone Gender'
    dataset = data_loaders.get_dataset(dataset_name)
    train, test = dataset.get_train_test_split()
    plot_dataset(
        train['X'],
        train['y'],
        X_test=test['X'],
        y_test=test['y'],
        dataset_name='Abalone Gender',
        terminal_plot=True,
        dim_reducer_method='PCA'
    )
