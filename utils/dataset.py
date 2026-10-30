import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic, reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def denormalize(tensor: torch.Tensor, mean: float = 0.1307, std: float = 0.3081) -> torch.Tensor:
    """Reverses MNIST normalization and clamps pixels to [0.0, 1.0] for plotting."""
    return torch.clamp((tensor * std) + mean, 0.0, 1.0)


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Downloads and prepares MNIST DataLoaders for train, validation, and test sets.

    Args:
        data_dir: Directory where the dataset will be downloaded/cached.
        batch_size: Batch size for DataLoader.
        val_split: Fraction of the training set to reserve for validation (default: 0.1 = 6,000 samples).
        seed: Random seed for reproducible splitting.
        num_workers: Number of subprocesses for data loading (0 recommended on Windows).

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Standard MNIST transformation pipeline: Tensor conversion and Z-score normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Download / Load full training set and test set
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Train / Validation split
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
