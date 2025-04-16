import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_interior_voxel_centers(interior_mask, step, min_bounds):
    # Get indices of interior voxels
    interior_indices = np.argwhere(interior_mask)

    # Compute center point of each voxel
    centers = min_bounds + (interior_indices + 0.5) * step

    d = centers.shape[1]

    if d == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(centers[:, 0], centers[:, 1], s=1, c='blue')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Interior voxel centers (2D)')
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    elif d == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=1, c='blue')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Interior voxel centers (3D)')
        plt.show()
    else:
        print(f"Plotting is only supported for 2D and 3D. Got dimension: {d}")




def plot_bar_diagram(x, y, xlabel='Category', ylabel='Value', title='Bar Diagram', show_values=True):
    plt.plot(x, y, marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simple Line Plot')
    plt.grid(True)
    plt.show()