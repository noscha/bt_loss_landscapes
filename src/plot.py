import os

import matplotlib.pyplot as plt


def plot_2d_coord_system(data, start_coord):
    data.remove(start_coord)
    x_vals = [point[0] for point in data]
    y_vals = [point[1] for point in data]
    plt.scatter(start_coord[0], start_coord[1], color='red', label='First point')
    plt.scatter(x_vals, y_vals, color='blue', marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot')
    plt.grid(True)
    plt.show()


def plot_bar_diagram(x, y, labels, name):
    colors = [
        'blue', 'green', 'red', 'orange', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan'
    ]

    for idx, y_dataset in enumerate(y):
        plt.plot(
            x, y_dataset,
            marker='o',
            label=labels[idx],
            color=colors[idx % len(colors)]
        )

    plt.xlabel('c: multiple of epsilon')
    plt.ylabel('log count of interior voxels')
    plt.title(f'Plot for Iris')
    plt.legend()
    plt.grid(True)
    os.makedirs("images", exist_ok=True)
    filename = "images/" + str(name) + ".png"
    plt.savefig(filename)
    plt.show()
