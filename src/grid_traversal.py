import random

import numpy as np


def grid_traversal(model, stepwidth, size_volume, queue=None, visited=None, accuracy=3, limit=10_000_000):
    iterations = 0
    start_coord = tuple(np.float64(round(i, accuracy)) for i in model.get_current_params())
    c = size_volume * model.evaluate_loss(start_coord)

    if queue is None and visited is None:
        queue, visited = [start_coord], {start_coord}

    future_queue = []

    # bfs
    while queue and iterations < limit:

        element_to_pop = random.randrange(len(queue))
        curr_coord = queue.pop(element_to_pop)

        for coord in neighboring_coords(curr_coord, stepwidth, accuracy):

            if coord in visited or random.random() > 0.5:
                pass
                # what to do if not included

            elif model.evaluate_loss(coord) <= c:
                queue.append(coord)
                visited.add(coord)

            else:
                future_queue.append(coord)
                visited.add(coord)

        iterations += 1

    if iterations == limit:
        print('Iteration limit was reached')

    return visited, start_coord, future_queue


def neighboring_coords(coord, stepwidth, accuracy):
    coord = np.array(coord)
    dim = len(coord)

    offsets = np.array([-stepwidth, stepwidth])
    all_offsets = np.array([np.array([o if i == j else 0 for i in range(dim)]) for j in range(dim) for o in offsets])
    all_offsets = all_offsets.reshape(2 * dim, dim)

    neighbors = [tuple(np.round(coord + offset, decimals=accuracy)) for offset in all_offsets]

    return neighbors
