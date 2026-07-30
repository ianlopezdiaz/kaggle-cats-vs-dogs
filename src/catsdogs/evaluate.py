"""Evaluation beyond accuracy: per-class metrics, confusion matrix, and
error analysis on the most confidently wrong predictions.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return softmax class probabilities, shape (N, num_classes), in loader order."""
    model.eval()
    probs = []
    for batch in loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs)


def compute_confusion_matrix(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix, rows are true labels, columns are predictions."""
    return confusion_matrix(labels, preds, labels=[0, 1])


def per_class_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall, and F1, one value per class."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, labels=[0, 1])
    return {"precision": precision, "recall": recall, "f1": f1}


def most_confident_errors(labels: np.ndarray, probs: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the ``k`` misclassified examples with the highest predicted
    probability for the (wrong) predicted class, most confident first.
    """
    preds = probs.argmax(axis=1)
    wrong = np.flatnonzero(labels != preds)
    confidence = probs[wrong, preds[wrong]]
    order = np.argsort(-confidence)
    return wrong[order][:k]
