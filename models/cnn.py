from typing import Tuple
import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    Lightweight Convolutional Neural Network (CNN) for MNIST digit classification.

    Architecture:
        Input: (B, 1, 28, 28)
        Conv Block 1: Conv2d(1, 16, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) -> (B, 16, 14, 14)
        Conv Block 2: Conv2d(16, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2, 2) -> (B, 32, 7, 7)
        Classifier: Flatten -> Linear(32 * 7 * 7, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 10)
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        fc_dim: int = 128,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            # Block 2
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
        )

        flat_features = conv2_channels * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure 4D shape (B, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
