import numpy as np


def explore(wrapper, start, direction, stepwidth):
    """Explore loss landscape in direction of derivative with stepwidth"""

    goal = start + 0.8 * direction  # TODO: find minima
    iso_values = []

    while np.linalg.norm(start - goal) > stepwidth:  # stop when within one step of the goal
        start += direction * stepwidth  # Move one step in the direction

        iso_values.append(wrapper.evaluate_loss(start))

        if np.linalg.norm(start - goal) < stepwidth:
            break

    return np.array(iso_values)
