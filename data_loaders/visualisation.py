# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Visualisation functions for each dataset
'''
import numpy as np
import matplotlib.pyplot as plt

from data_loaders.terminal_plots import terminal_show
from data_loaders.embeddings import dim_reducer


def plot_dataset(X,
                 y,
                 X_test=None,
                 y_test=None,
                 dataset_name=None,
                 label_names=None,
                 terminal_plot=False,
                 dim_reducer_method='TSNE',
                 ax=None):
    """
    Plot dataset in 2D using dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for training/full dataset
    y : np.ndarray
        Labels for training/full dataset
    X_test : np.ndarray, optional
        Feature matrix for test dataset (creates 2-subplot layout)
    y_test : np.ndarray, optional
        Labels for test dataset
    dataset_name : str, optional
        Name of dataset for title
    label_names : list, optional
        Names for each class label
    terminal_plot : bool, default=False
        If True, render plot in terminal (requires ax=None)
    dim_reducer_method : str, default='TSNE'
        Dimensionality reduction method: 'PCA', 'TSNE', or 'UMAP'
    ax : matplotlib.axes.Axes or tuple of 2 Axes, optional
        If single dataset (X_test=None): single Axes to plot on
        If train/test split (X_test provided): tuple of 2 Axes (train, test)
        If None (default): create new figure and call plt.show()

    Returns
    -------
    tuple or None
        If ax=None (created new figure): returns (fig, ax) or (fig, [ax1, ax2])
        If ax provided: returns None

    Raises
    ------
    ValueError
        If terminal_plot=True and ax is provided (incompatible options)
        If ax format doesn't match expected (single vs tuple)
    """

    # Validation: terminal_plot and ax are incompatible
    if terminal_plot and ax is not None:
        raise ValueError(
            "Cannot use terminal_plot=True with provided ax. "
            "Terminal plotting requires creating its own figure."
        )

    # Determine if we're creating new figure
    created_fig = ax is None

    if created_fig:
        # Current figure creation logic
        if X_test is None or y_test is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            axes = [ax]
        else:
            fig, axes_array = plt.subplots(1, 2, figsize=(12, 6))
            axes = list(axes_array)  # Convert numpy array to list
    else:
        # Use provided axes
        fig = None  # We don't own the figure
        if X_test is None or y_test is None:
            # Single dataset: ax should be a single Axes
            if isinstance(ax, (list, tuple)):
                raise ValueError("For single dataset, ax should be a single Axes, not a sequence")
            axes = [ax]
        else:
            # Train/test split: ax should be tuple of 2 Axes
            if not isinstance(ax, (list, tuple)) or len(ax) != 2:
                raise ValueError("For train/test split, ax should be a tuple/list of 2 Axes")
            axes = list(ax)

    # Setup data lists
    if X_test is None or y_test is None:
        Xs = [X]
        ys = [y]
        titles = ["Dataset"]
    else:
        Xs = [X, X_test]
        ys = [y, y_test]
        titles = ["Train set", "Test set"]

    colors = ["#3ea3e6", "#e56a6a"]  # blue, red

    # get 2d embedder fitted on train data
    embedder = dim_reducer(X, y, reducer=dim_reducer_method)

    # transform all datasets first to determine global axis limits
    embeddings = [embedder.transform(Xd) for Xd in Xs]

    # find global min/max across all embeddings for consistent scaling
    all_embeddings = embeddings[0] if len(embeddings) == 1 else np.vstack(embeddings)
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()

    # add padding (5% of range)
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    # plot each dataset (X_train, X_test)
    for ax, X_embed_2d, yd, title in zip(axes, embeddings, ys, titles):
        for cls in [0, 1]:
            if label_names is not None:
                class_label = f"Class {cls}: {label_names[cls]}"
            else:
                class_label = f"Class {cls}"
            ax.scatter(
                X_embed_2d[yd == cls, 0],
                X_embed_2d[yd == cls, 1],
                color=colors[cls],
                alpha=0.8,
                s=12,
                label=class_label
            )
        ax.set_title(f"{title} in {embedder.reducer_name} space")
        ax.set_xlabel(embedder.feature_names[0])
        ax.set_ylabel(embedder.feature_names[1])
        ax.legend()

        # set consistent axis limits across all subplots
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)

    if dataset_name is not None and created_fig:
        fig.suptitle(f"{dataset_name} dataset", fontsize=14)

    if created_fig:
        plt.tight_layout()
        if terminal_plot:
            terminal_show()
        else:
            plt.show()
        # Return tuple for single dataset, or tuple with list of axes for split
        if X_test is None or y_test is None:
            return (fig, axes[0])  # Single axes
        else:
            return (fig, axes)  # List of 2 axes
    else:
        # Don't call show() - caller controls display
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