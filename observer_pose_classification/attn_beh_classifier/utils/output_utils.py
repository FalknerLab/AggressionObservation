"""Output utilities for logging, checkpoint saving, and training curve plotting."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .metrics_utils import idx_to_name_dict


def setup_logging(
    output_dir: Path, script_name: str = "action_recognition"
) -> logging.Logger:
    """Configure file and console logging.

    Args:
        output_dir: Directory where the log file is written.
        script_name: Used for the log file name and logger name.

    Returns:
        Configured Logger instance.
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"{script_name}_{timestamp}.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def save_validation_outputs(
    predictions: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    segments: List[Dict],
    epoch: int,
    output_dir: Path,
    split: str = "val",
    filename: str = None,
) -> None:
    """Save predictions, labels, logits, and softmax probabilities to CSV.

    Args:
        predictions: Predicted class indices [N].
        labels: True class indices [N].
        logits: Raw logits before softmax [N, num_classes].
        segments: Segment metadata list (must contain 'session_id' and 'segment_idx').
        epoch: Current epoch (used in filename).
        output_dir: Directory where the CSV is saved.
        split: Split name ('val', 'test', etc.).
        filename: Override the auto-generated filename.
    """
    outputs_dir = output_dir / "validation_outputs"
    outputs_dir.mkdir(exist_ok=True, parents=True)

    data: Dict = {
        "epoch": [epoch] * len(predictions),
        "prediction": predictions,
        "true_label": labels,
        "session_id": [seg["session_id"] for seg in segments],
        "segment_idx": [seg.get("segment_idx", 0) for seg in segments],
    }

    num_classes = logits.shape[1]
    for i in range(num_classes):
        data[f"logit_{i}"] = logits[:, i]

    softmax_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    for i in range(num_classes):
        data[f"prob_{i}"] = softmax_probs[:, i]

    data["prediction_name"] = [
        idx_to_name_dict.get(p, f"Class_{p}") for p in predictions
    ]
    data["true_name"] = [idx_to_name_dict.get(t, f"Class_{t}") for t in labels]

    df = pd.DataFrame(data)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{split}_outputs_epoch_{epoch:03d}_{timestamp}.csv"

    df.to_csv(outputs_dir / filename, index=False)


def plot_training_curves(
    train_metrics: dict,
    val_metrics: dict,
    save_path: Path,
    warmup_epochs: int = None,
    test_metrics: dict = None,
):
    """Plot loss and accuracy curves for train, val, and optional OOD test splits.

    Args:
        train_metrics: Dict with 'loss' and 'accuracy' lists.
        val_metrics: Dict with 'loss' and 'accuracy' lists.
        save_path: Directory to save 'training_curves.png'.
        warmup_epochs: If provided, draws a vertical line marking warmup end.
        test_metrics: Optional dict with 'loss', 'accuracy', and 'epochs' lists.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_metrics["loss"], label="Train", linewidth=2)
    ax1.plot(val_metrics["loss"], label="Val", linewidth=2)
    if test_metrics and test_metrics.get("loss"):
        test_epochs = test_metrics.get("epochs", [])
        if len(test_epochs) == len(test_metrics["loss"]):
            ax1.plot(
                test_epochs, test_metrics["loss"], "o-", label="OOD Test", linewidth=2
            )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_metrics["accuracy"], label="Train", linewidth=2)
    ax2.plot(val_metrics["accuracy"], label="Val", linewidth=2)
    if test_metrics and test_metrics.get("accuracy"):
        test_epochs = test_metrics.get("epochs", [])
        if len(test_epochs) == len(test_metrics["accuracy"]):
            ax2.plot(
                test_epochs,
                test_metrics["accuracy"],
                "o-",
                label="OOD Test",
                linewidth=2,
            )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if warmup_epochs is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=warmup_epochs, color="orange", linestyle="--", alpha=0.7)
            ylim = ax.get_ylim()
            ax.text(
                warmup_epochs + 1,
                ylim[0] + 0.1 * (ylim[1] - ylim[0]),
                "Warmup End",
                color="orange",
                rotation=90,
            )

    plt.tight_layout()
    plt.savefig(save_path / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_checkpoint(
    model,
    optimizer,
    lr_scheduler,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    args: dict,
    filepath: Path,
    val_accuracy: float = None,
    test_accuracy: float = None,
):
    """Save a full training checkpoint including model, optimizer, and scheduler state.

    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        lr_scheduler: ProgressiveLRScheduler instance.
        epoch: Current epoch.
        train_metrics: Training metrics dict.
        val_metrics: Validation metrics dict.
        test_metrics: OOD test metrics dict (may be None).
        args: Argument dict (from vars(args)).
        filepath: Destination .pt file.
        val_accuracy: Best validation accuracy so far.
        test_accuracy: Best OOD test accuracy so far.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "args": args,
    }

    if hasattr(lr_scheduler, "current_epoch"):
        checkpoint["lr_scheduler_state"] = {
            "current_epoch": lr_scheduler.current_epoch,
            "classifier_lr": lr_scheduler.classifier_lr,
            "encoder_start_lr": lr_scheduler.encoder_start_lr,
            "encoder_target_lr": lr_scheduler.encoder_target_lr,
            "warmup_epochs": lr_scheduler.warmup_epochs,
            "total_epochs": lr_scheduler.total_epochs,
            "schedule": lr_scheduler.schedule,
        }

    if val_accuracy is not None:
        checkpoint["val_accuracy"] = val_accuracy
    if test_accuracy is not None:
        checkpoint["test_accuracy"] = test_accuracy

    torch.save(checkpoint, filepath)


def save_best_model(
    model,
    optimizer,
    epoch: int,
    val_accuracy: float,
    test_accuracy: float,
    num_classes: int,
    args: dict,
    filepath: Path,
):
    """Save the best model checkpoint.

    Args:
        model: PyTorch model.
        optimizer: Optimizer instance.
        epoch: Epoch at which this best model was achieved.
        val_accuracy: Validation accuracy at this checkpoint.
        test_accuracy: OOD test accuracy (may be None).
        num_classes: Number of output classes.
        args: Argument dict.
        filepath: Destination .pt file.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "num_classes": num_classes,
            "args": args,
        },
        filepath,
    )


def load_checkpoint(filepath: Path, model, optimizer=None, device: str = "cpu"):
    """Load a model checkpoint.

    Args:
        filepath: Path to the .pt checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore state into.
        device: Device to map tensors to.

    Returns:
        The full checkpoint dictionary.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
