import numpy as np
import matplotlib.pyplot as plt

def random_gray_grid(n=10, save_path=None):
    # 1. Create 0..1 values, linearly spaced
    values = np.linspace(0.0, 1.0, n * n)

    # 2. Shuffle them and reshape to n×n
    np.random.shuffle(values)
    grid = values.reshape((n, n))

    # 3. Plot grid
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(grid, cmap='gray', vmin=0, vmax=1, origin='upper')

    # 4. Draw grid lines
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=1)

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Optional: save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
    return grid  # in case you want the numeric values

if __name__ == "__main__":
    grid_values = random_gray_grid(n=10, save_path="gray_grid_10x10.png")
    print(grid_values)