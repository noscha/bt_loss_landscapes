import matplotlib.pyplot as plt

import src.marching_hypercubes as mh


def test_circle():
    def circle_function(*coords):
        x, y = coords
        return x ** 2 + y ** 2

    stepwidth = 0.05
    isovalue = 1.0
    start_coord = (1.0, 0.0)

    isosurface = mh.marching_hypercubes(circle_function, stepwidth, isovalue, start_coord)

    assert isosurface.size != 0

    plt.figure(figsize=(6, 6))
    plt.scatter(isosurface[:, 0], isosurface[:, 1], s=1, c="blue")
    plt.title("Isosurface for Circle Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def test_sphere():
    def sphere_function(*coords):
        x, y, z = coords
        return x ** 2 + y ** 2 + z ** 2

    # Parameters (example for 2D)
    stepwidth = 0.05
    isovalue = 1.0
    start_coord = (1.0, 0.0, 0.0)  # This is the center of the first hypercube

    isosurface = mh.marching_hypercubes(sphere_function, stepwidth, isovalue, start_coord)

    assert isosurface.size != 0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(isosurface[:, 0], isosurface[:, 1], isosurface[:, 2], s=1, c="blue")
    ax.set_title("Isosurface for Sphere Function")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def all_tests():
    test_circle()
    test_sphere()


if __name__ == "__main__":
    all_tests()
