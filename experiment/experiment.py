import math

import numpy as np

import src.grid_traversal as gt
import src.network as n
import src.plot as p


def look_for_modes():
    dimensions = [1]  # [2, 3, 4, 5]
    stepwidths = np.round(np.linspace(0.1, 0.01, num=5), 3)  # np.round(np.linspace(0.1, 0.01, num=20), 3)
    size_volumes = [2, 3, 4, 5]  # [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for dim in dimensions:

        wrapper = n.ModelWrapper(dim)
        counts_global = []

        for stepwidth in stepwidths:

            counts_local = []
            data, next_start = None, None

            for size_volume in size_volumes:
                data, _, next_start = gt.grid_traversal(wrapper, stepwidth, size_volume, queue=next_start, visited=data)

                count = round(math.log(len(data)), 3)
                print(dim, stepwidth, size_volume, count)
                counts_local.append(count)

            counts_global.append(counts_local)

        p.plot_bar_diagram(size_volumes, counts_global, labels=stepwidths, name=dim)

    def bfs_with_dropout():

        pass
