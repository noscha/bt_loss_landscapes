import src.network as nw


def test_network():
    dim = 2
    wrapper = nw.ModelWrapper(dim)
    assert wrapper.num_params == dim

    start_coord = wrapper.get_current_params()
    isovalue = wrapper.evaluate_loss(start_coord)

    assert len(start_coord) == dim
    assert isovalue != 0

    direction = wrapper.get_direction((0, 0))
    assert len(direction) == dim


def all_tests():
    test_network()


if __name__ == "__main__":
    all_tests()
