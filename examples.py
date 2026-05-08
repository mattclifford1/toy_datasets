"""
Examples demonstrating how to use the toy_datasets package.

Run with:
    uv run python examples.py           # Core examples
    uv run python examples.py --plot    # Include terminal plotting (requires sixel/kitty)
"""
import numpy as np
import data_loaders


def basic_usage():
    """Basic dataset loading and access."""
    print("=" * 60)
    print("BASIC USAGE")
    print("=" * 60)

    # Load a dataset
    loader = data_loaders.get_dataset('Iris')

    # Get features and labels
    X = loader.get_X()
    y = loader.get_y()

    print(f"Dataset: {loader.name}")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")


def train_test_split():
    """Demonstrate train/test splitting."""
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)

    # Load with custom split size
    loader = data_loaders.get_dataset('Banknote Authentication', train_size=0.8)

    # Get train/test split (preserves class proportions)
    train, test = loader.get_train_test_split()

    print(f"Train samples: {len(train['y'])}")
    print(f"Test samples: {len(test['y'])}")

    # Check class distribution
    train_classes, train_counts = np.unique(train['y'], return_counts=True)
    test_classes, test_counts = np.unique(test['y'], return_counts=True)

    print(f"Train class distribution: {dict(zip(train_classes, train_counts))}")
    print(f"Test class distribution: {dict(zip(test_classes, test_counts))}")


def scaling_and_normalization():
    """Demonstrate data scaling."""
    print("\n" + "=" * 60)
    print("SCALING AND NORMALIZATION")
    print("=" * 60)

    # Without scaling
    loader_raw = data_loaders.get_dataset('Wine', scale=False)
    train_raw, _ = loader_raw.get_train_test_split()

    # With scaling (MinMax to [-1, 1])
    loader_scaled = data_loaders.get_dataset('Wine', scale=True)
    train_scaled, _ = loader_scaled.get_train_test_split()

    print(f"Raw data range: [{train_raw['X'].min():.2f}, {train_raw['X'].max():.2f}]")
    print(f"Scaled data range: [{train_scaled['X'].min():.2f}, {train_scaled['X'].max():.2f}]")


def dimensionality_reduction():
    """Demonstrate dimensionality reduction options."""
    print("\n" + "=" * 60)
    print("DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # Original high-dimensional data
    loader_orig = data_loaders.get_dataset('Breast Cancer Wisconsin')
    X_orig = loader_orig.get_X()
    print(f"Original dimensions: {X_orig.shape[1]}")

    # With PCA reduction
    loader_pca = data_loaders.get_dataset(
        'Breast Cancer Wisconsin',
        dim_reducer='PCA',
        reduce_to_dim=2
    )
    train_pca, _ = loader_pca.get_train_test_split()
    print(f"After PCA: {train_pca['X'].shape[1]} dimensions")

    # With UMAP reduction
    loader_umap = data_loaders.get_dataset(
        'Breast Cancer Wisconsin',
        dim_reducer='UMAP',
        reduce_to_dim=2
    )
    train_umap, _ = loader_umap.get_train_test_split()
    print(f"After UMAP: {train_umap['X'].shape[1]} dimensions")


def synthetic_datasets():
    """Demonstrate synthetic dataset generators."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATASETS")
    print("=" * 60)

    # XOR dataset
    xor_loader = data_loaders.get_dataset('XOR', num_samples=[100, 100])
    print(f"XOR: {xor_loader.get_X().shape[0]} samples, {xor_loader.get_X().shape[1]} features")

    # Moons dataset with custom noise
    moons_loader = data_loaders.get_dataset('Moons', num_samples=500, moons_noise=0.1)
    print(f"Moons: {moons_loader.get_X().shape[0]} samples")

    # Blobs with multiple centers
    blobs_loader = data_loaders.get_dataset('Blobs', n_samples=300, centers=3)
    print(f"Blobs: {blobs_loader.get_X().shape[0]} samples, {len(np.unique(blobs_loader.get_y()))} classes")


def dataset_metadata():
    """Demonstrate accessing dataset metadata."""
    print("\n" + "=" * 60)
    print("DATASET METADATA")
    print("=" * 60)

    loader = data_loaders.get_dataset('Diabetes Pima Indian')

    print(f"Name: {loader.name}")
    print(f"Feature names: {loader.get_feature_names()}")
    print(f"Label names: {loader.get_label_names()}")
    print(f"\nDescription (first 200 chars):\n{loader.get_description()[:200]}...")


def downsampling():
    """Demonstrate data downsampling."""
    print("\n" + "=" * 60)
    print("DOWNSAMPLING")
    print("=" * 60)

    # Full dataset
    loader_full = data_loaders.get_dataset('Heart Disease')
    print(f"Full dataset: {loader_full.get_X().shape[0]} samples")

    # Downsampled to 50%
    loader_half = data_loaders.get_dataset('Heart Disease', percent_of_data=50)
    print(f"50% downsample: {loader_half.get_X().shape[0]} samples")

    # Downsampled to 25%
    loader_quarter = data_loaders.get_dataset('Heart Disease', percent_of_data=25)
    print(f"25% downsample: {loader_quarter.get_X().shape[0]} samples")


def class_balancing():
    """Demonstrate class balancing options."""
    print("\n" + "=" * 60)
    print("CLASS BALANCING")
    print("=" * 60)

    # Normal split (preserves original imbalance)
    loader_normal = data_loaders.get_dataset('Habermans Breast Cancer')
    _, test_normal = loader_normal.get_train_test_split()

    # With equal_test (balances test set)
    loader_balanced = data_loaders.get_dataset('Habermans Breast Cancer', equal_test=True)
    _, test_balanced = loader_balanced.get_train_test_split()

    print("Normal test set:")
    classes, counts = np.unique(test_normal['y'], return_counts=True)
    print(f"  Class distribution: {dict(zip(classes, counts))}")

    print("Balanced test set (equal_test=True):")
    classes, counts = np.unique(test_balanced['y'], return_counts=True)
    print(f"  Class distribution: {dict(zip(classes, counts))}")


def list_all_datasets():
    """List all available datasets."""
    print("\n" + "=" * 60)
    print("ALL AVAILABLE DATASETS")
    print("=" * 60)

    from data_loaders.main import AVAILABLE_DATASETS

    for i, name in enumerate(AVAILABLE_DATASETS.keys(), 1):
        print(f"  {i:2}. {name}")


def reproducibility():
    """Demonstrate reproducible results with seeds."""
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY")
    print("=" * 60)

    # Same seed = same results
    loader1 = data_loaders.get_dataset('Moons', set_seed=42)
    loader2 = data_loaders.get_dataset('Moons', set_seed=42)

    X1 = loader1.get_X()
    X2 = loader2.get_X()

    print(f"Same seed (42): arrays equal = {np.array_equal(X1, X2)}")

    # Different seed = different results
    loader3 = data_loaders.get_dataset('Moons', set_seed=123)
    X3 = loader3.get_X()

    print(f"Different seed (123): arrays equal = {np.array_equal(X1, X3)}")


def terminal_plotting():
    """Demonstrate terminal-based plotting (requires sixel/kitty support)."""
    print("\n" + "=" * 60)
    print("TERMINAL PLOTTING")
    print("=" * 60)
    print("(Requires terminal with sixel/kitty graphics support)")
    print()

    # Plot a 2D synthetic dataset
    print("Moons dataset:")
    moons = data_loaders.get_dataset('Moons', num_samples=200, moons_noise=0.15)
    moons.plot_dataset(terminal_plot=True)

    print("\nXOR dataset:")
    xor = data_loaders.get_dataset('XOR', num_samples=[100, 100])
    xor.plot_dataset(terminal_plot=True)

    print("\nCircles dataset:")
    circles = data_loaders.get_dataset('Circles', n_samples=200)
    circles.plot_dataset(terminal_plot=True)


def terminal_plotting_train_test():
    """Demonstrate train/test split visualization in terminal."""
    print("\n" + "=" * 60)
    print("TERMINAL PLOTTING - TRAIN/TEST SPLIT")
    print("=" * 60)
    print("(Requires terminal with sixel/kitty graphics support)")
    print()

    # Plot train/test split for Blobs
    print("Blobs train/test split:")
    blobs = data_loaders.get_dataset('Blobs', n_samples=300, centers=3, train_size=0.7)
    blobs.plot_train_test_split(terminal_plot=True)


def terminal_plotting_with_reduction():
    """Demonstrate plotting high-dimensional data with PCA reduction."""
    print("\n" + "=" * 60)
    print("TERMINAL PLOTTING - WITH DIMENSIONALITY REDUCTION")
    print("=" * 60)
    print("(Requires terminal with sixel/kitty graphics support)")
    print()

    # Plot Iris with PCA reduction to 2D
    print("Iris dataset (PCA reduced to 2D):")
    iris = data_loaders.get_dataset(
        'Iris',
        dim_reducer='PCA',
        reduce_to_dim=2
    )
    iris.plot_train_test_split(terminal_plot=True)

    # Plot Wine with PCA reduction
    print("\nWine dataset (PCA reduced to 2D):")
    wine = data_loaders.get_dataset(
        'Wine',
        dim_reducer='PCA',
        reduce_to_dim=2
    )
    wine.plot_train_test_split(terminal_plot=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run toy_datasets examples')
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Skip terminal plotting examples (requires sixel/kitty terminal)'
    )
    args = parser.parse_args()

    # Core examples (always run)
    basic_usage()
    train_test_split()
    scaling_and_normalization()
    dimensionality_reduction()
    synthetic_datasets()
    dataset_metadata()
    downsampling()
    class_balancing()
    reproducibility()
    list_all_datasets()

    # Terminal plotting examples (optional)
    if not args.no_plots:
        terminal_plotting()
        terminal_plotting_train_test()
        terminal_plotting_with_reduction()
    else:
        print("\n" + "-" * 60)
        print("Tip: Run with --no_plots flag to skip terminal plotting examples")
        print("  uv run python examples.py --no_plots")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
