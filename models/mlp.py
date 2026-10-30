from typing import List, Tuple
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    Multilayer Perceptron (Feedforward Neural Network) for MNIST digit classification.

    Architecture:
        Input: Flattened 28x28 image -> 784 features
        Hidden Layer 1: Linear(784, 512) -> BatchNorm1d -> ReLU -> Dropout(0.2)
        Hidden Layer 2: Linear(512, 256) -> BatchNorm1d -> ReLU -> Dropout(0.2)
        Output Layer: Linear(256, 10) -> 10 class logits
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = None,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Automatically flatten images of shape (B, 1, 28, 28) or (B, 28, 28) to (B, 784)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
