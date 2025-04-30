import numpy as np

import network as n
import grid_traversal as gt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    wrappper = n.ModelWrapper(3)
    o = np.asarray(wrappper.get_current_params())
    b = gt.subspace(wrappper, subspace_dim=2)
    visited, o, _ = gt.grid_traversal(wrappper, 0.1, 3, B=b)

    u = b[:, 0:1]
    v = b[:, 1:2]

    s = np.linspace(-1, 1, 20)
    t = np.linspace(-1, 1, 20)
    s_grid, t_grid = np.meshgrid(s, t)

    x_plane = o[0] + s_grid * u[0] + t_grid * v[0]
    y_plane = o[1] + s_grid * u[1] + t_grid * v[1]
    z_plane = o[2] + s_grid * u[2] + t_grid * v[2]

    points = []
    for point in visited:
        relative_coord = np.asarray(point)
        relative_coord = b @ relative_coord
        new_coord = o + relative_coord * 0.1
        points.append(new_coord)

    points = np.vstack(points)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot plane
    ax.plot_surface(x_plane, y_plane, z_plane,
                    alpha=0.5, color='cyan', edgecolor='k')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color='red', s=100, label='Data Points')

    # Add vectors for reference
    ax.quiver(*o, *u, color='blue', label='Vector u')
    ax.quiver(*o, *v, color='green', label='Vector v')

    # Customize plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('3D Plane with Data Points')
    plt.tight_layout()
    plt.show()

    """wrappper = n.ModelWrapper(-1)
    b = gt.subspace(wrappper)
    v, _, _ = gt.grid_traversal(wrappper, 0.01, 3, B=b, dropout=0.0)

    p.plot_2d_coord_system(v, tuple(np.ones(2)))"""

    #e.look_for_modes()

    """wrapper = n.ModelWrapper(4)

    v1 = wrapper.get_current_params()
    wrapper.train()
    v2 = wrapper.get_current_params()
    wrapper.train()
    v3 = wrapper.get_current_params()

    v1 = np.asarray([1, 0, 0, 1])
    v2 = np.asarray([0, 0, 1, 3])
    v3 = np.asarray([0, 1, 0, 2])

    A = np.column_stack((v1, v2, v3))

    print(A)
    print("--------------")

    Q, R = np.linalg.qr(A, mode='reduced')

    print("Orthonormal basis (columns of Q):")
    print(Q)"""

    # data, _, next_start = gt.grid_traversal(wrapper, 0.1, 1.5, queue=None, visited=None)
    # print(len(data))

    # e.look_for_modes()

""" 
TODO:
        tests
        experoment bfs runtime
        sub room
        
        
        abgleichen experiment mit einzelläüfen in tets
"""
