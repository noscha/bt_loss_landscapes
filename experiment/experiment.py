import math

import numpy as np

import src.grid_traversal as gt
import src.network as n
import src.plot as p


def look_for_modes():
    dimensions = [-1]  # [2, 3, 4, 5]
    stepwidths = np.round(np.linspace(0.1, 0.01, num=10), 3)  # np.round(np.linspace(0.1, 0.01, num=20), 3)
    size_volumes = [1.1, 1.2, 1.3, 1.4]  # [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for dim in dimensions:

        wrapper = n.ModelWrapper(dim)
        b = gt.subspace(wrapper, subspace_dim=3)
        counts_global = []

        for stepwidth in stepwidths:

            counts_local = []
            data, next_start = None, None

            for size_volume in size_volumes:
                data, _, next_start = gt.grid_traversal(wrapper, stepwidth, size_volume=size_volume, B=b,
                                                        queue=next_start, visited=data)

                count = round(math.log(len(data)), 3)
                print(dim, stepwidth, size_volume, count)
                counts_local.append(count)

            counts_global.append(counts_local)

        p.plot_bar_diagram(size_volumes, counts_global, labels=stepwidths, name=dim)

def bfs_with_dropout():

    pass

def multiple_minima():

    stepwidths = np.round(np.linspace(0.1, 0.05, num=5), 3)
    size_volumes = [1.1, 1.2, 1.3, 1.4, 1.5]

    wrapper = n.ModelWrapper(-1)
    b = gt.subspace(wrapper, subspace_dim=2)

    for minima in range(4):

        wrapper.train() # retrain model for new minima
        counts_global = []

        for stepwidth in stepwidths:

            counts_local = []
            data, next_start = None, None

            for size_volume in size_volumes:
                data, _, next_start = gt.grid_traversal(wrapper, stepwidth, size_volume=size_volume, B=b,
                                                        queue=next_start, visited=data)

                count = round(math.log(len(data)), 3)
                print(minima, stepwidth, size_volume, count)
                counts_local.append(count)

            counts_global.append(counts_local)

        p.plot_bar_diagram(size_volumes, counts_global, labels=stepwidths, name=minima)

def distance():
    size_volumes = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

    wrapper = n.ModelWrapper(-1)
    b = gt.subspace(wrapper, subspace_dim=2)

    dist_global = []
    for minima in range(5):


        data, next_start = None, None
        dists_local = []

        for size_volume in size_volumes:

            data, _, next_start = gt.grid_traversal(wrapper, 0.05, size_volume=size_volume, B=b,
                                                    queue=next_start, visited=data)

            print(minima, size_volume)

            sp = np.zeros(2)
            max_dist = max(math.dist(sp, point) for point in data)
            dists_local.append(max_dist)

        dist_global.append(dists_local)

    p.plot_bar_diagram(size_volumes, dist_global, labels=[0, 1, 2, 3, 4], name=0)


def different_basis():

    stepwidths = np.round(np.linspace(0.1, 0.01, num=10), 3)
    size_volumes = [1.1, 1.2, 1.3, 1.4, 1.5]

    wrapper = n.ModelWrapper(-1)


    for base in [0, 1]:

        b = gt.subspace(wrapper, subspace_dim=2, orthogonal=base)
        counts_global = []

        for stepwidth in stepwidths:

            counts_local = []
            data, next_start = None, None

            for size_volume in size_volumes:
                data, _, next_start = gt.grid_traversal(wrapper, stepwidth, size_volume=size_volume, B=b,
                                                        queue=next_start, visited=data)

                count = round(math.log(len(data)), 3)
                print(base, stepwidth, size_volume)
                counts_local.append(count)

            counts_global.append(counts_local)

        p.plot_bar_diagram(size_volumes, counts_global, labels=stepwidths, name=base)


if __name__ == "__main__":
    look_for_modes()