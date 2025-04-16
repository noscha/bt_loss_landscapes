import numpy as np
from collections import deque
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def count_interior_from_iso_points_hull_first(iso_points, step, margin=1):

    d = iso_points.shape[1]

    # Bounding box
    min_bounds = iso_points.min(axis=0) - margin * step
    max_bounds = iso_points.max(axis=0) + margin * step
    grid_shape = np.ceil((max_bounds - min_bounds) / step).astype(int)

    # Surface mask
    surface_mask = np.zeros(grid_shape, dtype=bool)
    indices = np.floor((iso_points - min_bounds) / step).astype(int)
    for idx in indices:
        surface_mask[tuple(idx)] = True

    # Compute voxel centers
    grid_coords = np.stack(np.meshgrid(*[np.arange(s) for s in grid_shape], indexing='ij'), axis=-1)
    voxel_centers = grid_coords * step + min_bounds + step / 2
    voxel_centers_flat = voxel_centers.reshape(-1, d)

    # Use ConvexHull to mask flood-fill area
    try:
        hull = ConvexHull(iso_points)
        delaunay = Delaunay(iso_points[hull.vertices])
        inside_hull_flat = delaunay.find_simplex(voxel_centers_flat) >= 0
        inside_hull_mask = inside_hull_flat.reshape(grid_shape)

        """fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(iso_points[:, 0], iso_points[:, 1], iso_points[:, 2], color='blue', s=5, alpha=0.5)
        for simplex in hull.simplices:
            tri = Poly3DCollection([iso_points[simplex]], color='orange', alpha=0.4)
            ax.add_collection3d(tri)
        for s in hull.simplices:
            s = np.append(s, s[0])  # cycle back to start
            ax.plot(iso_points[s, 0], iso_points[s, 1], iso_points[s, 2], color='k', linewidth=0.5)
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
        ax.set_title("3D Convex Hull of Iso-Surface Points")
        plt.tight_layout()
        plt.show()"""

    except Exception as e:
        print("ConvexHull failed:", e)
        return 0
        inside_hull_mask = np.ones(grid_shape, dtype=bool)  # fallback: everything inside

    # Flood-fill exterior — only allow flood through cells outside the hull
    exterior_mask = np.zeros(grid_shape, dtype=bool)
    queue = deque()

    for idx in np.ndindex(*grid_shape):
        if any(i == 0 or i == grid_shape[dim] - 1 for dim, i in enumerate(idx)):
            if not surface_mask[idx] and not exterior_mask[idx] and not inside_hull_mask[idx]:
                exterior_mask[idx] = True
                queue.append(idx)

    offsets = []
    for i in range(d):
        for delta in (-1, 1):
            offset = np.zeros(d, dtype=int)
            offset[i] = delta
            offsets.append(offset)

    while queue:
        current = np.array(queue.popleft())
        for offset in offsets:
            neighbor = current + offset
            if np.all(neighbor >= 0) and np.all(neighbor < grid_shape):
                neighbor_tuple = tuple(neighbor)
                if (
                    not surface_mask[neighbor_tuple]
                    and not exterior_mask[neighbor_tuple]
                    and not inside_hull_mask[neighbor_tuple]
                ):
                    exterior_mask[neighbor_tuple] = True
                    queue.append(neighbor_tuple)

    # Interior = inside hull AND not surface AND not exterior
    interior_mask = inside_hull_mask & ~surface_mask & ~exterior_mask
    interior_count = np.sum(interior_mask)

    return interior_count, tuple(grid_shape), surface_mask, exterior_mask