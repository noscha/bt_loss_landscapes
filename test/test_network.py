import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import src.network as nw

def test_network():

    torch.manual_seed(42)
    X = torch.rand(50, 2)
    y = torch.rand(50, 1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    model = nw.TinyNN()
    trainer = pl.Trainer(max_epochs=5, accelerator="cpu", enable_progress_bar=False)
    trainer.fit(model, dataloader)

    assert sum(p.numel() for p in model.parameters()) > 0

    wrapper = nw.ModelWrapper(model, dataloader)

    start_coord = wrapper.get_current_params()
    isovalue = wrapper.evaluate_loss(start_coord)

    assert len(start_coord) == sum(p.numel() for p in model.parameters())
    assert isovalue != 0

def all_tests():
    test_network()

if __name__ == "__main__":
    all_tests()