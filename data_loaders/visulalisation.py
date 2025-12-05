# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Visualisation functions for each dataset
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_dataset(X, y, X_test=None, y_test=None, dataset_name=None):
    def embed(X):
        if X.shape[1] > 2:
            return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
        return X

    # determine layout: one plot if test not provided, else two side by side
    if X_test is None or y_test is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        axes = [ax]
        Xs = [X]
        ys = [y]
        titles = ["Train"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        Xs = [X, X_test]
        ys = [y, y_test]
        titles = ["Train", "Test"]

    colors = ["#3ea3e6", "#e56a6a"]  # blue, red

    for ax, Xd, yd, title in zip(axes, Xs, ys, titles):
        X_embed_2d = embed(Xd)
        for cls in [0, 1]:
            ax.scatter(
                X_embed_2d[yd == cls, 0],
                X_embed_2d[yd == cls, 1],
                color=colors[cls],
                alpha=0.8,
                s=12,
                label=f"Class {cls}"
            )
        ax.set_title(f"{title} set in TSNE space")
        ax.set_xlabel("TSNE Dim 1")
        ax.set_ylabel("TSNE Dim 2")
        ax.legend()

    if dataset_name is not None:
        fig.suptitle(f"{dataset_name} dataset", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from data_loaders.main import get_dataset
    train = get_dataset(dataset='Breast Cancer', scale=True)
    plot_dataset(
        train['data']['X'], train['data']['y'],
        X_test=train['data_test']['X'], y_test=train['data_test']['y'],
        dataset_name='Breast Cancer'
    )