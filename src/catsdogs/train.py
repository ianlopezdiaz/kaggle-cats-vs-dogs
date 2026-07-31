"""Training loop, with the best checkpoint (by validation loss) saved to disk."""

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and torch so a training run reproduces.

    Worth calling before building the model rather than only before ``fit``,
    since the replaced classification head is randomly initialized. DataLoader
    workers inherit a seed derived from torch's generator, so the augmentation
    draws reproduce too, provided the worker count is unchanged.

    cuDNN is pinned to deterministic kernels, which costs a little throughput
    and buys repeatable gradients.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; returns the mean training loss."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss_and_accuracy(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one evaluation pass; returns (mean loss, accuracy)."""
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    checkpoint_path: Path | None = None,
) -> list[dict[str, float]]:
    """Train for a fixed number of epochs, saving the best checkpoint (by
    validation loss) to ``checkpoint_path`` if given.

    Only parameters with ``requires_grad=True`` are optimized, so freeze/
    unfreeze the desired layers (see ``model.py``) before calling this.
    Returns the per-epoch loss/accuracy history.
    """
    model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
        )
        if checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
    return history
