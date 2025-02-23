# Function with no logic

import torch


def set_enviroment(seed=42):
    """Make code deterministic"""
    torch.manual_seed(seed)
