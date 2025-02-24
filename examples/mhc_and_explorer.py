import matplotlib.pyplot as plt
import numpy as np

import src.explorer as e
import src.helper as h
import src.marching_hypercubes as mh
import src.network as nw


def explore_2d_nn():
    """Example usage of mhc and explorer."""
    h.set_enviroment(42)

    wrapper = nw.ModelWrapper(2)

    start_coord = wrapper.get_current_params()
    stepwidth = 0.01
    isovalue = wrapper.evaluate_loss(start_coord)

    isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

    plt.figure(figsize=(6, 6))
    plt.scatter(isosurface[:, 0], isosurface[:, 1], s=1, c="blue")

    for i in range(len(isosurface)):
        direction = wrapper.get_direction(isosurface[i])
        l = 0.8  # TODO: find minima
        a, b, c, d = np.concatenate([isosurface[i], -l * direction])
        plt.quiver(a, b, c, d, angles='xy', scale_units='xy', scale=1, color='b')

    plt.title("Isosurface for Network")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    data = []
    plane = []
    for i in np.arange(0, 0.7, 0.01):
        for j in np.arange(0.0, 0.7, 0.01):
            data.append((i, j, wrapper.evaluate_loss((i, j))))
            plane.append((i, j, isovalue))

    x, y, z = zip(*data)
    a, b, c = zip(*plane)
    crosspoints = [(i, j, z_val) for (i, j, z_val) in zip(x, y, z) if np.isclose(z_val, isovalue, atol=0.0001)]
    cx, cy, cz = zip(*crosspoints)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, alpha=0.1)
    ax.scatter(a, b, c, alpha=0.1)
    ax.scatter(cx, cy, cz, color='red', label='Crosspoints')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    t = 0
    for i in [0.1, 0.01, 0.001, 0.0001, 0.000001, 0.000001]:
        y = e.explore(wrapper, isosurface[t], wrapper.get_direction(isosurface[t]), i)

        plt.plot(list(range(len(y))), y)

        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.title('Graph of Y Values')
        plt.show()


def explore_3d_nn():
    h.set_enviroment(465)

    wrapper = nw.ModelWrapper(3)

    start_coord = wrapper.get_current_params()
    stepwidth = 0.1
    isovalue = wrapper.evaluate_loss(start_coord)

    isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(isosurface[:, 0], isosurface[:, 1], isosurface[:, 2], s=1, c="blue")

    for i in range(len(isosurface)):
        if i % 250 != 0:
            continue
        direction = wrapper.get_direction(isosurface[i])
        l = 0.8  # TODO: find minima
        a, b, c = isosurface[i]
        x, y, z = -l * direction
        ax.quiver(a, b, c, x, y, z, color='r', arrow_length_ratio=0.1)

    plt.show()


if __name__ == "__main__":
    explore_2d_nn()
    # explore_3d_nn()
