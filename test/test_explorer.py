from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np

import src.explorer as e


def test_explore():
    wrapper = MagicMock()
    wrapper.evaluate_loss.side_effect = lambda x: np.sin(x)

    iso_values = e.explore(wrapper, 0.0, 10.0, 0.01)

    plt.plot(list(range(len(iso_values))), iso_values)

    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Graph of Y Values')
    plt.show()


def all_tests():
    test_explore()


if __name__ == "__main__":
    all_tests()
