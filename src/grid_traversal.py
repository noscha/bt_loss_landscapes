import random

import numpy as np


# TODO: make class

def grid_traversal(model, stepwidth, size_volume, B=None, queue=None, visited=None, accuracy=3, dropout=0.0,
                   limit=10_000_000):
    iterations = 0
    # TODO: type cast !!!!!!!!!!!!!!!!!
    start_coord = tuple(np.float64(round(i, accuracy)) for i in model.get_current_params())  # theta_star
    relative_grid_position = tuple(np.zeros(len(start_coord)))  # a
    c = size_volume * model.evaluate_loss(start_coord)

    # TODO: wip
    #####################################
    if B is not None:
        relative_grid_position = tuple(np.zeros(3))
    ######################

    if queue is None and visited is None:
        queue, visited = [relative_grid_position], {relative_grid_position}

    future_queue = []

    # bfs
    while queue and iterations < limit:

        element_to_pop = random.randrange(len(queue)) if iterations > 1000 else 0
        curr_rel_coord = queue.pop(element_to_pop)

        for rel_coord in neighboring_coords(curr_rel_coord, accuracy):

            # no dropout of first 1000 elements for smooth start
            if rel_coord in visited or (random.random() < dropout and iterations > 1000):
                pass

            elif model.evaluate_loss(relative_to_true(rel_coord, B, start_coord, stepwidth, accuracy)) <= c:
                queue.append(rel_coord)
                visited.add(rel_coord)

            else:
                future_queue.append(rel_coord)
                visited.add(rel_coord)

        iterations += 1

    if iterations == limit:
        print('Iteration limit was reached')

    return visited, start_coord, future_queue


def neighboring_coords(coord, accuracy):
    dim = len(coord)
    neighbors = []

    for i in range(dim):
        for delta in [-1, 1]:
            neighbor = list(coord)
            neighbor[i] = round(neighbor[i] + delta, accuracy)
            neighbors.append(tuple(neighbor))

    return neighbors

def relative_to_true(relative_coord, B, start_coord, stepwidth, accuracy):
    relative_coord = np.asarray(relative_coord)

    #################################
    if B is not None:
        coord = B @ relative_coord

        relative_coord = coord
    ###################################

    start_coord = np.asarray(start_coord)

    # theta = theta_star + B(a) * s
    new_coord = start_coord + relative_coord * stepwidth

    return new_coord


def subspace(model, subspace_dim=3, orthogonal=True):
    vectors = []
    basis = []

    for i in range(subspace_dim + 1):
        model.train()
        vectors.append(np.asarray(model.get_current_params()))

    for i in range(1, subspace_dim + 1):
        basis.append(vectors[0] - vectors[i])

    basis = np.column_stack(basis)

    if np.linalg.matrix_rank(basis) != subspace_dim:
        print("Linear dependence")
        return -1

    if orthogonal:
        Q, R = np.linalg.qr(basis, mode='reduced')
        return Q

    return basis
