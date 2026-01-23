# author: Matt Clifford <matt.clifford@bristol.ac.uk>
'''
Visualisation functions for each dataset
'''
from openTSNE import TSNE as OpenTSNE
import matplotlib.pyplot as plt


def plot_dataset(X, 
                 y, 
                 X_test=None, 
                 y_test=None, 
                 dataset_name=None,
                 label_names=None):

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

    # get 2d embedder fitted on train data
    embeder = two_dim_embedder(X)

    # plot each dataset (X_train, X_test)
    for ax, Xd, yd, title in zip(axes, Xs, ys, titles):
        X_embed_2d = embeder.get_transform(Xd)
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
        ax.set_title(f"{title} set in TSNE space")
        ax.set_xlabel("TSNE Dim 1")
        ax.set_ylabel("TSNE Dim 2")
        ax.legend()

    if dataset_name is not None:
        fig.suptitle(f"{dataset_name} dataset", fontsize=14)

    plt.tight_layout()
    plt.show()


class two_dim_embedder:
    def __init__(self, 
                 X_train,  # use the train to fit
                 perplexity=30, 
                 random_state=42,
                 n_iter=1000,
                 n_jobs=-1):
        self.perplexity = perplexity
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_jobs = n_jobs

        # fit TSNE on training data
        if X_train.shape[1] > 2:
            self.model = OpenTSNE(
                n_components=2,
                perplexity=self.perplexity,
                random_state=self.random_state,
                n_iter=self.n_iter,
                n_jobs=self.n_jobs,
            )
            self.embedding_ = self.model.fit(X_train)
            self.transform = self.embedding_.transform
        else:
            # data is already 2d, no embedding needed
            self.embedding_ = None
            self.transform = lambda X: X  # identity function


    def get_transform(self, X):
        # Transform ONLY (no fit) for test/other data
        return self.transform(X)
    
    
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
        dataset_name='Abalone Gender'
    )