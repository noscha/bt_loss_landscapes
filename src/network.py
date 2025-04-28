import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from torch.utils.data import Dataset, DataLoader, TensorDataset


class ModelWrapper:
    """
    Wrapper to access model and manipulate it
    """

    def __init__(self, dim=2):

        seed = 24
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if dim == -1:

            self.model = IrisModel()
            dataset = IrisDataset()
            self.dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        else:

            self.model = TinyNN(dim)
            self.input_dim = dim  # TODO: might change later on
            self.dataloader = self.create_data(self.input_dim)

        self.num_params = sum(p.numel() for p in self.model.parameters())  # Total number of weights
        self.train()

    @staticmethod
    def create_data(dim):
        # TODO auslagern
        """Create random data"""
        X = torch.rand(50, dim)
        y = torch.rand(50, 1)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=50, shuffle=True)

        return dataloader

    def train(self):
        """Train model"""
        trainer = pl.Trainer(max_epochs=100, accelerator="cpu", enable_progress_bar=True)
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
        old_params = self.get_current_params()
        self.set_model_params(new_params)

        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dataloader:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, targets)
                total_loss += loss.item()

        self.set_model_params(old_params)
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


##################################################################


class TinyNN(pl.LightningModule):
    """
    Small network fot testing of algos
    """

    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            # nn.Linear(dim, 1, bias=False)
            nn.Linear(dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
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


#################################################################################

class IrisDataset(Dataset):
    def __init__(self):
        iris = load_iris()
        self.X = torch.tensor(iris.data, dtype=torch.float32)
        self.y = torch.tensor(iris.target, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IrisModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
