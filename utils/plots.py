import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from .dataset import denormalize


def plot_training_curves(
    histories: Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]],
    save_path: str = "training_curves.png",
) -> str:
    """
    Plots training and validation loss and accuracy curves across epochs.

    Supports either a single model history:
        {'train_loss': [...], 'val_loss': [...], 'train_acc': [...], 'val_acc': [...]}
    or multi-model histories:
        {'MLP': {'train_loss': [...], ...}, 'CNN': {'train_loss': [...], ...}}

    Args:
        histories: History dictionary or nested dictionary of models.
        save_path: Destination path to save the image.

    Returns:
        save_path (str)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Normalize histories to multi-model dictionary format
    if "train_loss" in histories:
        histories_dict = {"Model": histories}
    else:
        histories_dict = histories

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # 1. Loss Subplot
    ax_loss = axes[0]
    for idx, (m_name, hist) in enumerate(histories_dict.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax_loss.plot(
            epochs,
            hist["train_loss"],
            label=f"{m_name} Train Loss",
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax_loss.plot(
            epochs,
            hist["val_loss"],
            label=f"{m_name} Val Loss",
            color=color,
            linestyle="--",
            marker="s",
            linewidth=2,
            markersize=4,
        )

    ax_loss.set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss (Cross-Entropy)", fontsize=11)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(frameon=True, fontsize=10)

    # 2. Accuracy Subplot
    ax_acc = axes[1]
    for idx, (m_name, hist) in enumerate(histories_dict.items()):
        color = colors[idx % len(colors)]
        epochs = range(1, len(hist["train_acc"]) + 1)
        ax_acc.plot(
            epochs,
            hist["train_acc"],
            label=f"{m_name} Train Acc",
            color=color,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=4,
        )
        ax_acc.plot(
            epochs,
            hist["val_acc"],
            label=f"{m_name} Val Acc",
            color=color,
            linestyle="--",
            marker="s",
            linewidth=2,
            markersize=4,
        )

    ax_acc.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_misclassified_samples(
    images: Union[List[torch.Tensor], torch.Tensor],
    true_labels: Union[List[int], torch.Tensor],
    pred_labels: Union[List[int], torch.Tensor],
    save_path: str = "misclassified_samples.png",
    grid_size: Tuple[int, int] = (4, 4),
    confidences: Optional[Union[List[float], torch.Tensor]] = None,
) -> str:
    """
    Renders and saves a grid of misclassified test samples with True vs Pred annotations.

    Args:
        images: List or tensor of images.
        true_labels: Ground truth class integers.
        pred_labels: Predicted class integers.
        save_path: Output file path.
        grid_size: (rows, cols) for the subplot grid (default: (4, 4) = 16 samples).
        confidences: Optional predicted class probability/confidence.

    Returns:
        save_path (str)
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    rows, cols = grid_size
    num_samples = rows * cols
    available_samples = min(len(images), num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        if i < available_samples:
            img = images[i]
            if isinstance(img, torch.Tensor):
                img_denorm = denormalize(img).detach().cpu()
                if img_denorm.dim() == 3 and img_denorm.size(0) == 1:
                    img_np = img_denorm.squeeze(0).numpy()
                elif img_denorm.dim() == 2:
                    img_np = img_denorm.numpy()
                else:
                    img_np = img_denorm.view(28, 28).numpy()
            else:
                img_np = np.array(img).reshape(28, 28)

            true_val = int(true_labels[i])
            pred_val = int(pred_labels[i])

            ax.imshow(img_np, cmap="gray", interpolation="nearest")
            title_str = f"True: {true_val} | Pred: {pred_val}"
            if confidences is not None:
                title_str += f"\n({confidences[i]*100:.1f}%)"

            ax.set_title(title_str, fontsize=10, fontweight="bold", color="#d62728", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    plt.suptitle("Misclassified Test Digits (Error Analysis)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path
