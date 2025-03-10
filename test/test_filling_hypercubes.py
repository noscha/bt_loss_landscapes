import src.marching_hypercubes as mh
import src.filling_hypercubes as fh
import matplotlib.pyplot as plt

def test_fill():

    def circle_function(*coords):
        x, y = coords
        return max(abs(x), abs(y))

    stepwidth = 0.05
    isovalue = 0.5
    start_coord = (0.5, 0.0)

    isosurface = mh.marching_hypercubes(circle_function, stepwidth, isovalue, start_coord)

    plt.figure(figsize=(6, 6))
    plt.scatter(isosurface[:, 0], isosurface[:, 1], s=1, c="blue")
    plt.title("Isosurface for Circle Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    step = 0.05
    interior_count, grid_shape, surface_mask, exterior_mask, interior_iso_sum = fh.count_interior_from_iso_points(isosurface, step, margin=2, func=circle_function)

    assert interior_count == 399  # TODO tolerance

    print("Grid shape (cells):", grid_shape)
    print("Interior voxel count:", interior_count)
    print("Sum of interior iso values:", interior_iso_sum)

def all_tests():
    test_fill()


if __name__ == "__main__":
    all_tests()