import argparse
import io
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.optim import Adam

from models import get_model
from utils.dataset import get_mnist_dataloaders, set_seed
from utils.plots import plot_training_curves


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one training epoch and returns (mean_loss, accuracy_percentage)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs one evaluation/validation epoch and returns (mean_loss, accuracy_percentage)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def train_model(
    model_name: str,
    args: argparse.Namespace,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, List[float]], float]:
    """Full training pipeline for a given model architecture."""
    print(f"\n{'='*60}")
    print(f">> Initializing Training for: {model_name.upper()}")
    print(f"{'='*60}")

    model = get_model(model_name).to(device)
    total_params, trainable_params = model.count_parameters()
    print(f"Parameters: Total={total_params:,} | Trainable={trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_time": [],
    }

    best_val_acc = 0.0
    best_weights_path = os.path.join(args.save_dir, f"{model_name}_best.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(ep_duration)

        saved_flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "model_name": model_name,
                },
                best_weights_path,
            )
            saved_flag = " * [Saved Best]"

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
            f"({ep_duration:.2f}s){saved_flag}"
        )

    total_duration = time.time() - start_time
    print(f"\n[DONE] Finished training {model_name.upper()} in {total_duration:.2f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Model Weights Saved: {best_weights_path}")

    return model, history, total_duration


def main():
    parser = argparse.ArgumentParser(description="Train MNIST Digit Classifiers (MLP / CNN)")
    parser.add_argument("--model", type=str, default="all", choices=["mlp", "cnn", "all"],
                        help="Model architecture to train (default: 'all')")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to MNIST dataset directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to train on ('auto' detects CUDA)")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    # Data loaders
    print("\n[INFO] Loading MNIST dataset (90% train, 10% val, separate test set)...")
    train_loader, val_loader, _ = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        seed=args.seed,
    )
    print(f"Data batches: Train={len(train_loader)} | Val={len(val_loader)}")

    models_to_train = ["mlp", "cnn"] if args.model == "all" else [args.model]
    combined_histories = {}
    train_durations = {}

    for m_name in models_to_train:
        _, hist, duration = train_model(m_name, args, train_loader, val_loader, device)
        combined_histories[m_name.upper()] = hist
        train_durations[m_name.upper()] = duration

    # Save histories to JSON
    history_file = os.path.join(args.save_dir, "history.json")
    with open(history_file, "w") as f:
        json.dump(
            {
                "histories": combined_histories,
                "durations": train_durations,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(f"\n[INFO] Training history saved to: {history_file}")

    # Generate and save training curves
    plot_path = os.path.join(".", "training_curves.png")
    plot_training_curves(combined_histories, save_path=plot_path)
    print(f"[INFO] Training curves saved to: {plot_path}")


if __name__ == "__main__":
    main()
