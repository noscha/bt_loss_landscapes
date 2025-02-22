import torch
import src.network as nw

def test_network():

    torch.manual_seed(42)
    dim = 2
    wrapper = nw.ModelWrapper(dim)
    assert wrapper.num_params == dim

    start_coord = wrapper.get_current_params()
    isovalue = wrapper.evaluate_loss(start_coord)

    assert len(start_coord) == dim
    assert isovalue != 0

def all_tests():
    test_network()

if __name__ == "__main__":
    all_tests()