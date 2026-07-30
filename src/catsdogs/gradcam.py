"""Grad-CAM: visualize which image regions drive the model's prediction."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Gradient-weighted Class Activation Mapping for a single target layer.

    Registers forward/backward hooks on ``target_layer`` to capture its
    activations and gradients, then combines them into a coarse heatmap
    over the input image on each call.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Compute a Grad-CAM heatmap for a single image, shape (1, C, H, W).

        Returns a (H, W) array in [0, 1], upsampled to the input's spatial size.
        """
        self.model.eval()
        image = image.clone().requires_grad_(True)
        logits = self.model(image)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam -= cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam /= max_val
        return cam


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a (H, W) Grad-CAM heatmap onto a (H, W, 3) RGB image in [0, 1].

    Returns an (H, W, 3) array in [0, 1] using the "jet" colormap for the heatmap.
    """
    import matplotlib.cm as cm

    colored = cm.jet(heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)
