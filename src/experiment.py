import math

import numpy as np

import grid_traversal as gt
import network as n
import plot as p


def experiment():
    dimensions = [2, 3, 4, 5]
    size_volumes = [3, 4, 5, 6, 7, 8]
    stepwidths = np.round(np.linspace(0.1, 0.01, num=20), 3)

    for dim in dimensions:

        wrapper = n.ModelWrapper(dim)
        counts_global = []

        for stepwidth in stepwidths:

            counts_local = []

            for size_volume in size_volumes:
                data, _ = gt.grid_traversal(wrapper, stepwidth, size_volume)

                count = math.log(len(data))  # todo , STEP???
                print(size_volume, size_volume, dim, count)
                counts_local.append(count)

            counts_global.append(counts_local)

        p.plot_bar_diagram(size_volumes, counts_global, labels=stepwidths, name=dim)
