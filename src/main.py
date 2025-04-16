import network as nw
import matplotlib.pyplot as plt
import marching_hypercubes as mh
import filling_hypercubes as fh
import plot as p
import numpy as np

import json

def main():
    """dim = 2
    wrapper = nw.ModelWrapper(dim)

    start_coord = wrapper.get_current_params()
    isovalue = wrapper.evaluate_loss(start_coord)
    stepwidth = 0.001

    isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

    filename = '2d.json'
    with open(filename, 'w') as file:
        json.dump(isosurface.tolist(), file)"""

    filename = '2d.json'
    with open(filename, 'r') as file:
        isosurface =  np.array(json.load(file))

    plt.figure(figsize=(6, 6))
    plt.scatter(isosurface[:, 0], isosurface[:, 1], s=1, c="blue")
    plt.title("Isosurface for Circle Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    for step in [0.05, 0.025, 0.005, 0.0025, 0.0005]:

        #step = 0.005
        interior_count, grid_shape, surface_mask, exterior_mask = fh.count_interior_from_iso_points_hull_first(
            isosurface, step, margin=2)

        print("Grid shape (cells):", grid_shape)
        print("Interior voxel count:", interior_count)

        # TODO work with interior mask directly
        p.plot_interior_voxel_centers(~surface_mask & ~exterior_mask, step, isosurface.min(axis=0) - 2 * step)

if __name__ == "__main__":
    main()
