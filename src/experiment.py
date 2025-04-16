import math

import network as nw
import marching_hypercubes as mh
import filling_hypercubes as fh
import plot as p
import numpy as np

def experiment():

    dimensions = [2, 3, 4, 5]
    steps = np.round(np.linspace(0.05, 0.005, 20), 3)
    #steps = [0.05, 0.025, 0.005, 0.0025]

    for dim in dimensions:

        interior_counts = []

        wrapper = nw.ModelWrapper(dim)

        start_coord = wrapper.get_current_params()
        isovalue = wrapper.evaluate_loss(start_coord)
        stepwidth = 0.1

        isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

        for step in steps:

            interior_count, grid_shape, surface_mask, exterior_mask  = fh.count_interior_from_iso_points_hull_first(
                isosurface, step, margin=2)

            print(dim, step, interior_count)
            interior_counts.append(math.log(interior_count))

        p.plot_bar_diagram(steps, interior_counts)


def three_d_tets():


    wrapper = nw.ModelWrapper(3)

    start_coord = wrapper.get_current_params()
    isovalue = wrapper.evaluate_loss(start_coord)
    stepwidth = 0.1

    isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

    """

    filename = '3d.json'
    with open(filename, 'w') as file:
        json.dump(isosurface.tolist(), file)

    filename = '3d.json'
    with open(filename, 'r') as file:
        isosurface =  np.array(json.load(file))
        
    """


    """fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(isosurface[:, 0], isosurface[:, 1], isosurface[:, 2], s=1, c="blue")
    ax.set_title("Isosurface for Sphere Function")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()"""

    interior_count, _, _, _ = fh.count_interior_from_iso_points_hull_first(
        isosurface, 0.05, margin=2)

    print(interior_count)


if __name__ == "__main__":
    experiment()
