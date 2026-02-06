"""
Example demonstrating the optional axes parameter for plotting.

This shows how to:
1. Use default behavior (creates figure and shows it)
2. Plot on custom single axes
3. Plot multiple datasets in a grid
4. Plot train/test split on custom axes
5. Use return values for further customization
"""
import matplotlib.pyplot as plt
from data_loaders import get_dataset


def example_1_default_behavior():
    """Example 1: Default behavior (unchanged from before)."""
    print("Example 1: Default behavior")
    loader = get_dataset("Moons")
    fig, ax = loader.plot_dataset()
    print(f"  Created figure: {fig}")
    print(f"  Created axes: {ax}")


def example_2_custom_single_subplot():
    """Example 2: Plot on custom single axes."""
    print("\nExample 2: Custom single subplot")
    fig, ax = plt.subplots(figsize=(8, 8))

    loader = get_dataset("Moons")
    loader.plot_dataset(ax=ax)

    # Customize after plotting
    ax.set_title("My Custom Title for Moons Dataset", fontsize=16, color='navy')
    ax.set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig('/tmp/custom_single_plot.png', dpi=150)
    print("  Saved to /tmp/custom_single_plot.png")
    plt.close()


def example_3_multiple_datasets_in_grid():
    """Example 3: Multiple datasets in a 2x2 grid."""
    print("\nExample 3: Multiple datasets in 2x2 grid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    datasets = ['XOR', 'Moons', 'Circles', 'Blobs']
    titles = ['XOR Pattern', 'Two Moons', 'Concentric Circles', 'Blob Clusters']

    for ax, dataset_name, title in zip(axes.flat, datasets, titles):
        loader = get_dataset(dataset_name)
        loader.plot_dataset(ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')

    fig.suptitle("Comparison of Synthetic Datasets", fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/grid_comparison.png', dpi=150)
    print("  Saved to /tmp/grid_comparison.png")
    plt.close()


def example_4_train_test_custom_layout():
    """Example 4: Train/test split on custom axes."""
    print("\nExample 4: Train/test split on custom layout")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    loader = get_dataset("Gaussian")
    loader.plot_train_test_split(ax=(ax1, ax2))

    # Customize both axes
    ax1.set_facecolor('#ffe6e6')  # Light red background
    ax2.set_facecolor('#e6f7ff')  # Light blue background

    fig.suptitle("Custom Train/Test Visualization", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/train_test_custom.png', dpi=150)
    print("  Saved to /tmp/train_test_custom.png")
    plt.close()


def example_5_return_values_for_customization():
    """Example 5: Use return values for further customization."""
    print("\nExample 5: Using return values for customization")
    loader = get_dataset("Circles")

    # Get the figure and axes for customization
    fig, ax = loader.plot_dataset()

    # Now we can customize before showing
    ax.set_facecolor('#f5f5dc')  # Beige background
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add custom annotation
    ax.text(0.02, 0.98, 'Circles Dataset\nPerfect for testing\nnon-linear classifiers',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save instead of showing
    fig.savefig('/tmp/customized_plot.png', dpi=150, facecolor='white')
    print("  Saved to /tmp/customized_plot.png")
    plt.close()


def example_6_mixed_comparison():
    """Example 6: Compare same dataset with different splits."""
    print("\nExample 6: Compare dataset with different train/test ratios")
    fig = plt.figure(figsize=(16, 10))

    # Create a grid with 2 rows (different split ratios), 2 columns (train/test)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 50/50 split
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    loader1 = get_dataset("Blobs", split_size=0.5)
    loader1.plot_train_test_split(ax=(ax1, ax2))
    ax1.set_title("50/50 Split - Train", fontsize=12)
    ax2.set_title("50/50 Split - Test", fontsize=12)

    # 80/20 split
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    loader2 = get_dataset("Blobs", split_size=0.8)
    loader2.plot_train_test_split(ax=(ax3, ax4))
    ax3.set_title("80/20 Split - Train", fontsize=12)
    ax4.set_title("80/20 Split - Test", fontsize=12)

    fig.suptitle("Impact of Different Train/Test Split Ratios", fontsize=16, fontweight='bold')
    plt.savefig('/tmp/split_comparison.png', dpi=150)
    print("  Saved to /tmp/split_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Demonstrating optional axes parameter for plotting\n")
    print("=" * 60)

    # Run all examples
    # Note: Example 1 will call plt.show() by default
    # Uncomment to run: example_1_default_behavior()

    example_2_custom_single_subplot()
    example_3_multiple_datasets_in_grid()
    example_4_train_test_custom_layout()
    example_5_return_values_for_customization()
    example_6_mixed_comparison()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("Check /tmp/ for generated PNG files.")
