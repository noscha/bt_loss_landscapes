import pytorch_lightning as pl
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset


class ModelWrapper:
    """
    Wrapper to access model and manipulate it
    """

    def __init__(self, dim=2):
        random.seed(42)
        self.model = TinyNN(dim)
        self.input_dim = dim  # TODO: might change later on
        self.dataloader = self.create_data(self.input_dim)
        self.num_params = sum(p.numel() for p in self.model.parameters())  # Total number of weights
        self.train()

    @staticmethod
    def create_data(dim):
        """Create random data"""
        X = torch.rand(50, dim)
        y = torch.rand(50, 1)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        return dataloader

    def train(self):
        """Train model"""
        trainer = pl.Trainer(max_epochs=5, accelerator="cpu", enable_progress_bar=False)
        trainer.fit(self.model, self.dataloader)

    def get_current_params(self):
        """Returns the current model parameters as a tuple."""
        return tuple(torch.cat([p.view(-1) for p in self.model.parameters()]).tolist())

    def set_model_params(self, new_params):
        """Update model parameters from a tuple."""
        with torch.no_grad():
            idx = 0
            for param in self.model.parameters():
                param_length = param.numel()
                param.copy_(torch.tensor(new_params[idx:idx + param_length], dtype=torch.float32).view_as(param))
                idx += param_length

    def evaluate_loss(self, new_params):
        """Compute the loss after updating model parameters."""
        self.set_model_params(new_params)

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def func(self, *params):
        """Function interface that takes n parameters, updates the model, and returns loss."""
        assert len(params) == self.num_params, f"Expected {self.num_params} parameters, got {len(params)}"
        return self.evaluate_loss(params)

    def get_direction(self, new_params):
        """ Get normalized derivative for given parameters"""

        self.set_model_params(new_params)
        self.model.zero_grad()  # Ensure gradients are fresh

        total_loss = 0.0
        for inputs, targets in self.dataloader:
            outputs = self.model(inputs)
            loss = self.model.criterion(outputs, targets)
            total_loss += loss

        total_loss /= len(self.dataloader)
        total_loss.backward()
        direction_vector = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        self.model.zero_grad()  # Reset gradients after extraction

        direction_vector = direction_vector.detach().numpy()
        # TODO: normalization ???
        # direction_vector /= np.linalg.norm(direction_vector)
        return direction_vector


class TinyNN(pl.LightningModule):
    """
    Small network fot testing of algos
    """

    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 1, bias=False)
            # nn.Linear(1, 2),
            # nn.ReLU(),
            # nn.Linear(2, 1)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
