# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Visualisation functions for each dataset
'''
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
                 dim_reducer_method='TSNE'):
    
    # determine layout: one plot if test not provided, else two side by side
    if X_test is None or y_test is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        axes = [ax]
        Xs = [X]
        ys = [y]
        titles = ["Dataset"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        Xs = [X, X_test]
        ys = [y, y_test]
        titles = ["Train set", "Test set"]

    colors = ["#3ea3e6", "#e56a6a"]  # blue, red

    # get 2d embedder fitted on train data
    embedder = dim_reducer(X, y, reducer=dim_reducer_method)

    # plot each dataset (X_train, X_test)
    for ax, Xd, yd, title in zip(axes, Xs, ys, titles):
        X_embed_2d = embedder.transform(Xd)
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

    if dataset_name is not None:
        fig.suptitle(f"{dataset_name} dataset", fontsize=14)

    plt.tight_layout()
    if terminal_plot:
        terminal_show()
    else:
        plt.show()
    
    
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