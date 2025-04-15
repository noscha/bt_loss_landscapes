import network as nw
import marching_hypercubes as mh
import filling_hypercubes as fh

def experiment():

    dimensions = [2, 3, 4, 5]
    steps = [0.25, 0.05, 0.01, 0.025]

    for dim in dimensions:

        wrapper = nw.ModelWrapper(dim)

        start_coord = wrapper.get_current_params()
        isovalue = wrapper.evaluate_loss(start_coord)
        stepwidth = 0.001

        isosurface = mh.marching_hypercubes(wrapper.func, stepwidth, isovalue, start_coord)

        for step in steps:

            interior_count, _, _, _ = fh.count_interior_from_iso_points(
                isosurface, step, margin=2)

            print(interior_count, dim, step)

if __name__ == "__main__":
    experiment()
