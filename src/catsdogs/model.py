"""Pretrained backbone loading, classification head replacement, and layer
freezing for transfer learning.
"""

from torch import nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet18 with its final fully-connected layer replaced for
    ``num_classes``-way classification.

    All backbone layers start frozen; use ``unfreeze_last_blocks`` for a
    second fine-tuning phase.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 1) -> None:
    """Unfreeze the last ``num_blocks`` residual blocks (``layer4``, then
    ``layer3``, ...) plus the classification head, for a short additional
    fine-tuning phase.
    """
    block_names = ["layer4", "layer3", "layer2", "layer1"][:num_blocks]
    for name in block_names:
        for param in getattr(model, name).parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
