"""Metrics utilities for action recognition evaluation."""

import numpy as np
from typing import Dict

# Class index to name mapping (0-indexed labels from preload pipeline)
idx_to_name_dict = {
    0: "grooming",
    1: "investigate",
    2: "rearing",
    3: "scratching",
    4: "sniffing",
    5: "still",
    6: "turning",
}


def calculate_per_class_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """Calculate per-class precision, recall, F1, accuracy, and sample counts.

    Args:
        predictions: Array of predicted class indices.
        labels: Array of true class indices.

    Returns:
        Dictionary mapping class names to dicts with keys
        'accuracy', 'precision', 'recall', 'f1', 'count' (all as percentages
        except 'count').
    """
    from sklearn.metrics import precision_recall_fscore_support

    unique_classes = np.unique(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=unique_classes, average=None, zero_division=0
    )

    per_class_metrics = {}
    for i, class_idx in enumerate(unique_classes):
        class_name = idx_to_name_dict.get(class_idx, f"Class_{class_idx}")
        class_mask = labels == class_idx
        class_acc = (
            (predictions[class_mask] == class_idx).mean() * 100
            if class_mask.sum() > 0
            else 0.0
        )
        per_class_metrics[class_name] = {
            "accuracy": class_acc,
            "precision": precision[i] * 100,
            "recall": recall[i] * 100,
            "f1": f1[i] * 100,
            "count": int(support[i]),
        }

    return per_class_metrics
