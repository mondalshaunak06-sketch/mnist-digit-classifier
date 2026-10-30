import argparse
import io
import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from tabulate import tabulate

from models import get_model
from utils.dataset import get_mnist_dataloaders, set_seed
from utils.plots import plot_misclassified_samples


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[torch.Tensor], List[int], List[int], List[float], float]:
    """
    Evaluates model performance on the test set and collects misclassified samples.

    Returns:
        (test_loss, test_acc, mis_images, mis_trues, mis_preds, mis_confs, avg_batch_latency_ms)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    misclassified_images = []
    misclassified_trues = []
    misclassified_preds = []
    misclassified_confs = []

    batch_latencies = []

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            start_t = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_latencies.append((time.perf_counter() - start_t) * 1000.0)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Detect misclassifications
            mask = preds != targets
            if mask.any():
                mis_imgs = images[mask].cpu()
                mis_t = targets[mask].cpu().tolist()
                mis_p = preds[mask].cpu().tolist()
                mis_c = confs[mask].cpu().tolist()

                for img, t, p, c in zip(mis_imgs, mis_t, mis_p, mis_c):
                    misclassified_images.append(img)
                    misclassified_trues.append(t)
                    misclassified_preds.append(p)
                    misclassified_confs.append(c)

    test_loss = running_loss / total
    test_acc = (correct / total) * 100.0
    avg_latency = float(sum(batch_latencies) / len(batch_latencies))

    return (
        test_loss,
        test_acc,
        misclassified_images,
        misclassified_trues,
        misclassified_preds,
        misclassified_confs,
        avg_latency,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MNIST digit classifiers")
    parser.add_argument("--mlp_weights", type=str, default="./checkpoints/mlp_best.pt",
                        help="Path to MLP best checkpoint")
    parser.add_argument("--cnn_weights", type=str, default="./checkpoints/cnn_best.pt",
                        help="Path to CNN best checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST data directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoints directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--grid_rows", type=int, default=4, help="Grid rows for error analysis")
    parser.add_argument("--grid_cols", type=int, default=4, help="Grid cols for error analysis")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Evaluation device: {device}")

    # Load test loader
    _, _, test_loader = get_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        seed=args.seed,
    )
    print(f"Test samples: {len(test_loader.dataset):,} across {len(test_loader)} batches")

    criterion = nn.CrossEntropyLoss()
    results = {}
    misclassified_data = {}

    models_to_eval = [
        ("MLP (Baseline)", "mlp", args.mlp_weights),
        ("CNN (Lightweight)", "cnn", args.cnn_weights),
    ]

    # Load training durations if available from history.json
    history_file = os.path.join(args.save_dir, "history.json")
    training_metadata = {}
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            training_metadata = json.load(f)

    for display_name, m_key, weight_path in models_to_eval:
        if not os.path.exists(weight_path):
            print(f"[WARN] Warning: Checkpoint not found at {weight_path}. Skipping {display_name}.")
            continue

        print(f"\n[EVAL] Evaluating {display_name}...")
        model = get_model(m_key).to(device)
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        total_p, _ = model.count_parameters()

        test_loss, test_acc, mis_imgs, mis_t, mis_p, mis_c, avg_latency = evaluate_model(
            model, test_loader, criterion, device
        )

        val_acc = checkpoint.get("val_acc", None)
        durations = training_metadata.get("durations", {})
        train_time = durations.get(m_key.upper(), None)

        throughput = (args.batch_size / (avg_latency / 1000.0)) if avg_latency > 0 else 0.0

        results[m_key] = {
            "name": display_name,
            "params": total_p,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "mis_count": len(mis_imgs),
            "latency": avg_latency,
            "throughput": throughput,
            "train_time": train_time,
        }

        misclassified_data[m_key] = (mis_imgs, mis_t, mis_p, mis_c)

        print(f"  -> Test Loss: {test_loss:.4f}")
        print(f"  -> Test Accuracy: {test_acc:.2f}% ({10000 - len(mis_imgs)}/10,000 correct)")
        print(f"  -> Misclassified Count: {len(mis_imgs)}")
        print(f"  -> Avg Latency: {avg_latency:.2f} ms/batch ({throughput:.1f} imgs/s)")

    # Produce Misclassified Samples Plot
    # Prefer CNN for error analysis since its errors are subtle edge cases, or fall back to MLP
    primary_key = "cnn" if "cnn" in misclassified_data else ("mlp" if "mlp" in misclassified_data else None)
    if primary_key:
        mis_imgs, mis_t, mis_p, mis_c = misclassified_data[primary_key]
        grid_save_path = "misclassified_samples.png"
        plot_misclassified_samples(
            mis_imgs,
            mis_t,
            mis_p,
            save_path=grid_save_path,
            grid_size=(args.grid_rows, args.grid_cols),
            confidences=mis_c,
        )
        print(f"\n[INFO] Saved misclassified sample grid ({primary_key.upper()}): {grid_save_path}")

        # Also save specific individual plots if both exist
        for m_key, (m_imgs, m_t, m_p, m_c) in misclassified_data.items():
            indiv_path = f"misclassified_samples_{m_key}.png"
            plot_misclassified_samples(
                m_imgs,
                m_t,
                m_p,
                save_path=indiv_path,
                grid_size=(args.grid_rows, args.grid_cols),
                confidences=m_c,
            )

    # Print Formatted Comparison Table
    if "mlp" in results and "cnn" in results:
        mlp = results["mlp"]
        cnn = results["cnn"]

        table_data = [
            ["Architecture Type", "Fully-Connected (MLP)", "Convolutional (CNN)"],
            ["Total Parameters", f"{mlp['params']:,}", f"{cnn['params']:,}"],
            ["Validation Accuracy (Best)", f"{mlp['val_acc']:.2f}%" if mlp['val_acc'] else "N/A",
             f"{cnn['val_acc']:.2f}%" if cnn['val_acc'] else "N/A"],
            ["Test Accuracy", f"{mlp['test_acc']:.2f}%", f"{cnn['test_acc']:.2f}%"],
            ["Test Loss", f"{mlp['test_loss']:.4f}", f"{cnn['test_loss']:.4f}"],
            ["Misclassified Digits", f"{mlp['mis_count']} / 10,000", f"{cnn['mis_count']} / 10,000"],
            ["Error Reduction", "-", f"{((mlp['mis_count'] - cnn['mis_count']) / mlp['mis_count']) * 100:.1f}% fewer errors"],
            ["Avg Batch Latency (ms)", f"{mlp['latency']:.2f} ms", f"{cnn['latency']:.2f} ms"],
            ["Throughput (imgs/sec)", f"{mlp['throughput']:.1f}", f"{cnn['throughput']:.1f}"],
            ["Training Time", f"{mlp['train_time']:.2f}s" if mlp['train_time'] else "N/A",
             f"{cnn['train_time']:.2f}s" if cnn['train_time'] else "N/A"],
        ]

        print(f"\n{'='*75}")
        print("MODEL PERFORMANCE COMPARISON (Baseline MLP vs. CNN)")
        print(f"{'='*75}")
        formatted_table = tabulate(
            table_data,
            headers=["Metric", "Baseline MLP", "Lightweight CNN"],
            tablefmt="grid",
        )
        print(formatted_table)

        # Save comparison results JSON
        eval_json_path = os.path.join(args.save_dir, "evaluation_summary.json")
        with open(eval_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Evaluation summary saved to: {eval_json_path}")


if __name__ == "__main__":
    main()
