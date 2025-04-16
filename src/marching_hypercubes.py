from collections import deque
from itertools import product, combinations
from typing import Callable, List, Tuple, Sequence, Optional, Dict

import numpy as np

import matplotlib.pyplot as plt


def generate_lookup_table(dim: int) -> Dict[int, List[Tuple[int, int]]]:
    """
    Generate a lookup table for isosurfacing in a d-dimensional hypercube.

    For each possible configuration (a bit mask of 2^(dim) vertices),
    this table lists the edges (as vertex index pairs) that are intersected
    by an isosurface. An edge is intersected if its two endpoints have
    different signs with respect to the isovalue.

    Parameters:
        dim (int): The dimension of the hypercube.

    Returns:
        Dict[int, List[Tuple[int, int]]]:
            A dictionary mapping each configuration (as an integer) to a list
            of edges (each edge represented by a tuple of vertex indices).
    """
    num_vertices = 2 ** dim
    cases = 2 ** num_vertices
    lookup_table: Dict[int, List[Tuple[int, int]]] = {}

    for case_index in range(cases):
        # Determine whether each vertex is above (1) or below (0) the isovalue.
        signs = [(case_index >> i) & 1 for i in range(num_vertices)]
        edges: List[Tuple[int, int]] = []
        # Check each pair of vertices to see if they form an edge (differ in one bit)
        # and if they straddle the isosurface.
        for i, j in combinations(range(num_vertices), 2):
            if bin(i ^ j).count('1') == 1 and signs[i] != signs[j]:
                edges.append((i, j))
        lookup_table[case_index] = edges
    return lookup_table


def interpolate_edge(p1: np.ndarray, p2: np.ndarray, v1: float, v2: float, isovalue: float) -> np.ndarray:
    """
    Linearly interpolate between two points p1 and p2 given scalar values v1 and v2,
    returning the point where the scalar function equals isovalue.

    Parameters:
        p1 (np.ndarray): Coordinates of the first point.
        p2 (np.ndarray): Coordinates of the second point.
        v1 (float): Function value at p1.
        v2 (float): Function value at p2.
        isovalue (float): The target isovalue.

    Returns:
        np.ndarray: The interpolated point on the edge where the function equals isovalue.
    """
    # Avoid division by zero by checking if the two values are nearly equal.
    if np.isclose(v1, v2):
        return p1  # or alternatively return (p1+p2)/2
    t = (isovalue - v1) / (v2 - v1)
    return p1 + t * (p2 - p1)


def get_hypercube_vertices(dim: int, origin: np.ndarray, stepwidth: float) -> List[np.ndarray]:
    """
    Generate the vertices of a hypercube in d dimensions.

    Each vertex is computed as:
        vertex = origin + stepwidth * vertex_offset,
    where vertex_offset is one of the tuples in {0, 1}^d.

    Parameters:
        dim (int): The dimension of the hypercube.
        origin (np.ndarray): The lower corner of the hypercube.
        stepwidth (float): The side length of the hypercube.

    Returns:
        List[np.ndarray]: A list of vertex coordinates as NumPy arrays.
    """
    return [
        origin + stepwidth * np.array(vertex)
        for vertex in product([0, 1], repeat=dim)
    ]


def marching_hypercubes(func: Callable[..., float],
                        stepwidth: float,
                        isovalue: float,
                        start_coord: Sequence[float],
                        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        max_cells: Optional[int] = None) -> np.ndarray:
    """
    Generalized Marching Hypercubes algorithm for any dimension.

    This function marches through grid cells (hypercubes) in a d-dimensional space,
    collects intersection points with the isosurface (where func == isovalue), and
    supports optional boundary limits and a maximum number of cells to process.

    The first cell is placed so that its center is at 'start_coord'. To do so, the grid
    origin is defined as:
        grid_origin = start_coord - (stepwidth/2)

    Parameters:
        func (Callable[..., float]):
            The scalar function to evaluate; expects d arguments (one per dimension).
        stepwidth (float):
            The side length of each hypercube cell.
        isovalue (float):
            The threshold value defining the isosurface.
        start_coord (Sequence[float]):
            The coordinate at which to center the initial cell.
        bounds (Optional[Tuple[np.ndarray, np.ndarray]]):
            An optional tuple (lower_bound, upper_bound) that defines the valid region
            for the hypercube grid. Each is a NumPy array of length d.
        max_cells (Optional[int]):
            An optional maximum number of cells to process. This acts as a safeguard
            against unbounded exploration.

    Returns:
        np.ndarray: An array of points (as NumPy arrays) where the isosurface intersects the grid.
    """
    dim = len(start_coord)
    lookup_table = generate_lookup_table(dim)

    isosurface_points: List[np.ndarray] = []
    visited: set[Tuple[int, ...]] = set()

    # Set grid origin so that the first cell (indices (0,...,0)) is centered at start_coord.
    grid_origin = np.array(start_coord) - (stepwidth / 2)
    start_indices = (0,) * dim
    queue = deque([start_indices])

    # full_mask corresponds to the case when all vertices are above the isovalue.
    full_mask = (2 ** (2 ** dim)) - 1

    while queue:
        indices = queue.popleft()
        if indices in visited:
            continue
        visited.add(indices)

        # Optionally stop if we've processed too many cells.
        if max_cells is not None and len(visited) >= max_cells:
            break
        if len(visited) % 1000 == 0:
            print(len(visited))


            """fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            isosurface = np.array(isosurface_points)
            ax.scatter(isosurface[:, 0], isosurface[:, 1], isosurface[:, 2], s=1, c="blue")
            ax.set_title("Isosurface for Sphere Function")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            name = "images/" + str(len(visited)) + ".png"
            #plt.savefig(name, dpi=300, bbox_inches='tight', transparent=True)
            plt.show()"""


        # Compute the lower corner of the current cell.
        origin = grid_origin + stepwidth * np.array(indices)
        # If bounds are provided, check that the cell is within the allowed region.
        if bounds is not None:
            lower_bound, upper_bound = bounds
            # Ensure that the entire cell is within bounds.
            if np.any(origin < lower_bound) or np.any(origin + stepwidth > upper_bound):
                continue

        vertices = get_hypercube_vertices(dim, origin, stepwidth)
        values = [func(*vertex) for vertex in vertices]

        # Early rejection: if the cell's values do not span the isovalue, skip further processing.
        cell_min = min(values)
        cell_max = max(values)
        if cell_max <= isovalue or cell_min > isovalue:
            continue

        # Compute the case index based on whether each vertex's value is above the isovalue.
        case_index = sum(2 ** i for i, v in enumerate(values) if v > isovalue)

        # Process cell if it is not completely inside or outside the isosurface.
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

                    # Check the neighbor's cell against the bounds, if provided.
                    neighbor_origin = grid_origin + stepwidth * np.array(neighbor_indices)
                    if bounds is not None:
                        lower_bound, upper_bound = bounds
                        if np.any(neighbor_origin < lower_bound) or np.any(neighbor_origin + stepwidth > upper_bound):
                            continue

                    # Evaluate neighbor cell's vertices.
                    neighbor_vertices = get_hypercube_vertices(dim, neighbor_origin, stepwidth)
                    neighbor_values = [func(*vertex) for vertex in neighbor_vertices]
                    neighbor_min = min(neighbor_values)
                    neighbor_max = max(neighbor_values)
                    if neighbor_max <= isovalue or neighbor_min > isovalue:
                        continue

                    neighbor_case_index = sum(2 ** i for i, v in enumerate(neighbor_values) if v > isovalue)

                    if neighbor_case_index != 0 and neighbor_case_index != full_mask:
                        queue.append(neighbor_indices)

    return 0 if len(isosurface_points) == 0 else np.array(isosurface_points)
