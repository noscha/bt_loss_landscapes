import numpy as np
from itertools import product, combinations
from collections import deque


def generate_lookup_table(dim):
    """
    Generate a lookup table for isosurfacing in a d-dimensional hypercube.
    """
    num_vertices = 2 ** dim
    cases = 2 ** num_vertices
    lookup_table = {}

    for case_index in range(cases):
        signs = [(case_index >> i) & 1 for i in range(num_vertices)]
        edges = []
        for i, j in combinations(range(num_vertices), 2):
            if bin(i ^ j).count('1') == 1 and signs[i] != signs[j]:
                edges.append((i, j))
        lookup_table[case_index] = edges
    return lookup_table


def interpolate_edge(p1, p2, v1, v2, isovalue):
    """
    Linearly interpolate between two points p1 and p2 given scalar values v1 and v2.
    """
    t = (isovalue - v1) / (v2 - v1)
    return p1 + t * (p2 - p1)


def get_hypercube_vertices(dim, origin, stepwidth):
    """
    Generate the vertices of a hypercube in d dimensions.

    The vertices are computed as:
        vertex = origin + stepwidth * vertex_offset,
    where vertex_offset is one of the tuples in {0, 1}^d.
    """
    return [
        origin + stepwidth * np.array(vertex)
        for vertex in product([0, 1], repeat=dim)
    ]


def marching_hypercubes(func, stepwidth, isovalue, start_coord):
    """
    Generalized Marching Hypercubes algorithm for any dimension.

    This version places the first hypercube so that its center is at 'start_coord'.
    To do so, we define the grid origin as:
        grid_origin = start_coord - (stepwidth/2)
    so that the initial cell (with indices (0, 0, ..., 0)) has lower corner at
        start_coord - (stepwidth/2).

    The algorithm then marches through neighboring cells (adding/subtracting stepwidth)
    and collects isosurface intersection points.
    """
    dim = len(start_coord)
    lookup_table = generate_lookup_table(dim)

    isosurface_points = []
    visited = set()

    # Set grid origin so that the first cell is centered at start_coord.
    grid_origin = np.array(start_coord) - (stepwidth / 2)

    # Starting cell indices (0,0,...,0) corresponds to a cell whose lower corner is grid_origin.
    start_indices = (0,) * dim

    queue = deque([start_indices])

    # Compute a full mask for a cell with 2^dim vertices.
    full_mask = (2 ** (2 ** dim)) - 1

    while queue:
        indices = queue.popleft()
        if indices in visited:
            continue
        visited.add(indices)

        # Compute cell's lower corner from indices relative to grid_origin.
        origin = grid_origin + stepwidth * np.array(indices)
        vertices = get_hypercube_vertices(dim, origin, stepwidth)
        values = [func(*vertex) for vertex in vertices]
        case_index = sum(2 ** i for i, v in enumerate(values) if v > isovalue)

        # Process cell if it is neither entirely above nor entirely below the isovalue.
        if case_index != 0 and case_index != full_mask:
            for edge in lookup_table[case_index]:
                v1, v2 = edge
                p1, p2 = vertices[v1], vertices[v2]
                val1, val2 = values[v1], values[v2]
                point = interpolate_edge(p1, p2, val1, val2, isovalue)
                isosurface_points.append(point)

            # Explore neighbors in each dimension.
            for i in range(dim):
                for delta in [-1, 1]:
                    neighbor_indices = list(indices)
                    neighbor_indices[i] += delta
                    neighbor_indices = tuple(neighbor_indices)

                    if neighbor_indices in visited:
                        continue

                    neighbor_origin = grid_origin + stepwidth * np.array(neighbor_indices)
                    neighbor_vertices = get_hypercube_vertices(dim, neighbor_origin, stepwidth)
                    neighbor_values = [func(*vertex) for vertex in neighbor_vertices]
                    neighbor_case_index = sum(2 ** i for i, v in enumerate(neighbor_values) if v > isovalue)

                    if neighbor_case_index != 0 and neighbor_case_index != full_mask:
                        queue.append(neighbor_indices)

    return np.array(isosurface_points)
