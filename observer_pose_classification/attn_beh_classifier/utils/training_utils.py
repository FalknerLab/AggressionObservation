"""Training utilities: training/evaluation loops and progressive LR scheduler."""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class ProgressiveLRScheduler:
    """Progressive learning rate scheduler for stable encoder adaptation.

    Applies a two-phase schedule to the encoder:
    - **Warmup phase** (epochs 0..warmup_epochs): encoder LR increases from
      encoder_start_lr to encoder_target_lr using the selected schedule.
    - **Decay phase** (epochs warmup_epochs..total_epochs): encoder LR decays
      from encoder_target_lr via cosine annealing.

    The classifier LR follows cosine annealing throughout training, starting
    from classifier_lr.

    Args:
        optimizer: AdamW optimizer with two param groups (classifier first, encoder second).
        classifier_lr: Initial classifier learning rate.
        encoder_start_lr: Encoder LR at epoch 0.
        encoder_target_lr: Encoder LR after warmup.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        schedule: Warmup curve shape ('linear', 'cosine', or 'exponential').
    """

    def __init__(
        self,
        optimizer,
        classifier_lr: float,
        encoder_start_lr: float,
        encoder_target_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        schedule: str = "cosine",
    ):
        self.optimizer = optimizer
        self.classifier_lr = classifier_lr
        self.encoder_start_lr = encoder_start_lr
        self.encoder_target_lr = encoder_target_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.current_epoch = 0

        self.optimizer.param_groups[0]["lr"] = classifier_lr
        self.optimizer.param_groups[1]["lr"] = encoder_start_lr

    def step(self):
        """Advance one epoch and update both param group LRs.

        Returns:
            Tuple of (classifier_lr, encoder_lr) for the current epoch.
        """
        progress = self.current_epoch / self.total_epochs
        classifier_lr = self.classifier_lr * 0.5 * (1 + np.cos(np.pi * progress))

        if self.current_epoch < self.warmup_epochs:
            wp = self.current_epoch / self.warmup_epochs
            if self.schedule == "linear":
                encoder_lr = self.encoder_start_lr + wp * (
                    self.encoder_target_lr - self.encoder_start_lr
                )
            elif self.schedule == "cosine":
                encoder_lr = self.encoder_start_lr + (
                    self.encoder_target_lr - self.encoder_start_lr
                ) * 0.5 * (1 - np.cos(np.pi * wp))
            elif self.schedule == "exponential":
                ratio = self.encoder_target_lr / self.encoder_start_lr
                encoder_lr = self.encoder_start_lr * (ratio**wp)
            else:
                encoder_lr = self.encoder_start_lr + wp * (
                    self.encoder_target_lr - self.encoder_start_lr
                )
        else:
            post_wp = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            encoder_lr = self.encoder_target_lr * 0.5 * (1 + np.cos(np.pi * post_wp))

        self.optimizer.param_groups[0]["lr"] = classifier_lr
        self.optimizer.param_groups[1]["lr"] = encoder_lr
        self.current_epoch += 1
        return classifier_lr, encoder_lr


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    scaler: GradScaler = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train for one epoch with optional AMP and gradient accumulation.

    Args:
        model: Model to train.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.
        epoch: Current epoch number (used for logging).
        logger: Logger instance.
        scaler: GradScaler for AMP; None disables AMP scaling.
        use_amp: Whether automatic mixed precision is enabled.
        gradient_accumulation_steps: Number of mini-batches per optimizer step.
        max_grad_norm: Gradient clipping threshold.

    Returns:
        Dict with 'loss' (average) and 'accuracy' (percent).
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    accumulated_loss = 0
    steps_since_update = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", mininterval=1.0)

    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp and scaler is not None:
            with autocast("cuda"):
                logits = model(videos)
                loss = criterion(logits, labels) / gradient_accumulation_steps
        else:
            logits = model(videos)
            loss = criterion(logits, labels) / gradient_accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item()
        steps_since_update += 1

        if steps_since_update >= gradient_accumulation_steps:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            steps_since_update = 0

        with torch.no_grad():
            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            total_loss += accumulated_loss * gradient_accumulation_steps
            accumulated_loss = 0

        if batch_idx % 10 == 0:
            pbar.set_postfix(
                loss=f"{loss.item() * gradient_accumulation_steps:.4f}",
                acc=f"{100 * correct / total:.2f}%",
            )

    if steps_since_update > 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    logger.info(f"Epoch {epoch} Train — Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    split: str = "val",
    use_amp: bool = False,
    save_outputs: bool = False,
    output_dir: Path = None,
    segments: List[Dict] = None,
) -> dict:
    """Evaluate the model for one epoch with per-class metrics.

    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        criterion: Loss function.
        device: Compute device.
        epoch: Current epoch number.
        logger: Logger instance.
        split: Split name ('val' or 'test') used for logging and file naming.
        use_amp: Whether to use automatic mixed precision.
        save_outputs: Save predictions and logits to CSV.
        output_dir: Directory for CSV output (required when save_outputs=True).
        segments: Segment metadata list (required when save_outputs=True).

    Returns:
        Dict with 'loss', 'accuracy', 'per_class_metrics',
        'predictions', 'labels', 'logits'.
    """
    from .metrics_utils import calculate_per_class_metrics
    from .output_utils import save_validation_outputs

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_logits = []

    pbar = tqdm(
        dataloader, desc=f"Epoch {epoch} [{split.capitalize()}]", mininterval=1.0
    )

    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast("cuda"):
                logits = model(videos)
                loss = criterion(logits, labels)
        else:
            logits = model(videos)
            loss = criterion(logits, labels)

        _, predicted = logits.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.extend(logits.cpu().numpy())

        if batch_idx % 10 == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    per_class_metrics = calculate_per_class_metrics(all_predictions, all_labels)

    if save_outputs and output_dir is not None and segments is not None:
        try:
            save_validation_outputs(
                predictions=all_predictions,
                labels=all_labels,
                logits=all_logits,
                segments=segments,
                epoch=epoch,
                output_dir=output_dir,
                split=split,
            )
        except Exception as e:
            logger.warning(f"Failed to save {split} outputs: {e}")

    logger.info(
        f"Epoch {epoch} {split.capitalize()} — Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%"
    )
    for class_name in sorted(per_class_metrics):
        m = per_class_metrics[class_name]
        logger.info(
            f"  {class_name}: Acc={m['accuracy']:.1f}% | "
            f"P={m['precision']:.1f}% | R={m['recall']:.1f}% | "
            f"F1={m['f1']:.1f}% | n={m['count']}"
        )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class_metrics": per_class_metrics,
        "predictions": all_predictions,
        "labels": all_labels,
        "logits": all_logits,
    }


def val_epoch(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    logger,
    use_amp=False,
    save_outputs=False,
    output_dir=None,
    segments=None,
):
    """Convenience wrapper: evaluate on the validation split."""
    return evaluate_epoch(
        model,
        dataloader,
        criterion,
        device,
        epoch,
        logger,
        "val",
        use_amp,
        save_outputs,
        output_dir,
        segments,
    )


def test_epoch(
    model,
    dataloader,
    criterion,
    device,
    epoch,
    logger,
    use_amp=False,
    save_outputs=False,
    output_dir=None,
    segments=None,
):
    """Convenience wrapper: evaluate on the test split."""
    return evaluate_epoch(
        model,
        dataloader,
        criterion,
        device,
        epoch,
        logger,
        "test",
        use_amp,
        save_outputs,
        output_dir,
        segments,
    )
