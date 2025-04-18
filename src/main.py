import experiment as e

import plot as p
import network as n
import grid_traversal as gt

if __name__ == "__main__":
    dim = 2
    wrapper = n.ModelWrapper(dim)

    stepwidth, size_volume = 0.1, 3
    data, start_coord, _ = gt.grid_traversal(wrapper, stepwidth, size_volume)

    p.plot_2d_coord_system(data, start_coord)

    #e.experiment()
