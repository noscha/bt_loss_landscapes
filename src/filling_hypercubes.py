import numpy as np
from collections import deque
from itertools import product


def count_interior_from_iso_points(iso_points, step, margin=1):
    """
    Given an iso point cloud (points on the isosurface) in high dimensions,
    this function creates a grid, marks cells that contain an iso point as surface,
    flood-fills from the boundary to mark exterior cells, and then counts cells
    that are neither surface nor exterior (i.e. interior)

    Parameters:
      iso_points : np.ndarray
          Array of shape (N, d) containing the iso point cloud.
      step : float
          The side length of each hypercube (voxel).
      margin : int, optional
          Number of extra cells to pad around the bounding box.
      func : Callable, optional
          A scalar function f(x1, ..., xd) defined in the d-dimensional space.
          If provided, f is evaluated at the center of each interior voxel and
          these values are summed.

    Returns:
      interior_count : int
          Count of cells inside the closed isosurface.
      grid_shape : tuple
          The shape (number of cells per dimension) of the grid.
      surface_mask : np.ndarray (bool)
          Boolean array with True for cells that contain an iso point.
      exterior_mask : np.ndarray (bool)
          Boolean array with True for cells that are flood-filled as exterior.
      interior_iso_sum : float or None
          The sum of f(x) over the interior voxels (evaluated at their centers)
          if func is provided; otherwise, None.
    """
    # Determine the bounding box of the iso points.
    d = iso_points.shape[1]
    min_bounds = iso_points.min(axis=0) - margin * step
    max_bounds = iso_points.max(axis=0) + margin * step
    grid_shape = np.ceil((max_bounds - min_bounds) / step).astype(int)

    # Create a grid (as a boolean array) to mark surface cells.
    surface_mask = np.zeros(grid_shape, dtype=bool)

    # Map each iso point to a cell index and mark that cell as surface.
    indices = np.floor((iso_points - min_bounds) / step).astype(int)
    for idx in indices:
        surface_mask[tuple(idx)] = True

    # Flood-fill to mark exterior cells.
    exterior_mask = np.zeros(grid_shape, dtype=bool)
    queue = deque()

    # Add boundary cells that are not surface.
    for idx in np.ndindex(*grid_shape):
        if any(i == 0 or i == grid_shape[dim] - 1 for dim, i in enumerate(idx)):
            if not surface_mask[idx] and not exterior_mask[idx]:
                exterior_mask[idx] = True
                queue.append(idx)

    # Define neighbor offsets (1-step moves in each dimension).
    offsets = []
    for i in range(d):
        for delta in (-1, 1):
            offset = np.zeros(d, dtype=int)
            offset[i] = delta
            offsets.append(offset)

    # Perform flood-fill: propagate exterior label through cells that are not surface.
    while queue:
        current = np.array(queue.popleft())
        for offset in offsets:
            neighbor = current + offset
            if np.all(neighbor >= 0) and np.all(neighbor < grid_shape):
                neighbor_tuple = tuple(neighbor)
                if not surface_mask[neighbor_tuple] and not exterior_mask[neighbor_tuple]:
                    exterior_mask[neighbor_tuple] = True
                    queue.append(neighbor_tuple)

    # Cells that are neither surface nor exterior are considered interior.
    interior_mask = ~surface_mask & ~exterior_mask
    interior_count = np.sum(interior_mask)

    return interior_count, tuple(grid_shape), surface_mask, exterior_mask