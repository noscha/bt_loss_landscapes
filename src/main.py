import network as nw
import matplotlib.pyplot as plt
import marching_hypercubes as mh
import filling_hypercubes as fh

def main():
    dim = 2
    wrapper = nw.ModelWrapper(dim)

    start_coord = wrapper.get_current_params()
    isovalue = wrapper.evaluate_loss(start_coord)
    stepwidth = 0.05

    isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

    plt.figure(figsize=(6, 6))
    plt.scatter(isosurface[:, 0], isosurface[:, 1], s=1, c="blue")
    plt.title("Isosurface for Circle Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    step = 0.05
    interior_count, grid_shape, surface_mask, exterior_mask, interior_iso_sum = fh.count_interior_from_iso_points(
        isosurface, step, margin=2, func=wrapper.func)

    print("Grid shape (cells):", grid_shape)
    print("Interior voxel count:", interior_count)
    print("Sum of interior iso values:", interior_iso_sum)

if __name__ == "__main__":
    main()
