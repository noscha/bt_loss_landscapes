import matplotlib.pyplot as plt

import helper as h
import marching_hypercubes as mh
import network as nw


def main():
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
    main()
